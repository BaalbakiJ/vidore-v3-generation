import asyncio
import json
import logging
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from vidore_generation.dtos import Failed, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.utils import post_process_output

warnings.filterwarnings("ignore", category=UserWarning)


class APIGenerationHandler(GenerationHandler):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: Optional[logging.Logger] = None,
        retry_count: int = 3,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.retry_count = retry_count
        self.extra_kwargs = extra_kwargs or {}

        if logger is None:
            self.logger = logging.getLogger("LLM Call Tracker")
            self.logger.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        else:
            self.logger = logger

        self.cost = 0

    def _get_completion_kwargs(self, arguments: Dict[str, Any]) -> Dict:
        kwargs = {}
        kwargs.update(arguments)
        if "pydantic_schema" in arguments:
            # Pass the pydantic class directly — litellm translates to the correct
            # provider-specific format (json_schema for OpenAI, json_object for Fireworks, etc.)
            kwargs["response_format"] = arguments["pydantic_schema"]
            kwargs.pop("pydantic_schema")
        # Apply model-level extra kwargs; don't override prompt-level arguments
        for k, v in self.extra_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        # Ensure a token limit is always set when not provided by caller or extra_kwargs.
        # Use max_completion_tokens as the canonical default — litellm translates it to
        # max_tokens for providers that require it (e.g. Fireworks).
        if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_completion_tokens"] = 10_000
            kwargs["max_tokens"] = 10_000
        if "openai" in self.model_name.lower():
            # OpenAI models require max_tokens instead of max_completion_tokens
            kwargs.pop("max_tokens", None)
        return kwargs

    def _process_response(self, response, prompt: Prompt):
        self.cost += litellm.completion_cost(response, model=self.model_name)
        if "pydantic_schema" in prompt.arguments:
            output = prompt.arguments["pydantic_schema"].model_validate_json(
                post_process_output(response.choices[0].message.content)
            )
        else:
            output = post_process_output(response.choices[0].message.content)
        return output

    def _handle_error(self, val_error, prompt: str):
        logging.debug(
            f"Call to {self.model_name} with prompt: {prompt}\nreturned the following error:\n{val_error}"
        )
        return Failed(error=str(val_error))

    def generate_single_sample(self, prompt: Prompt) -> str:
        try:
            kwargs = self._get_completion_kwargs(prompt.arguments)
            response = litellm.completion(
                model=self.model_name,
                messages=prompt.messages,
                stream=False,
                **kwargs,
            )
            return self._process_response(response, prompt)

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            return self._handle_error(val_error, prompt)

    async def async_generate_single_sample(self, prompt: Prompt) -> str:
        try:
            kwargs = self._get_completion_kwargs(prompt.arguments)
            response = await litellm.acompletion(
                model=self.model_name,
                messages=prompt.messages,
                stream=False,
                **kwargs,
            )
            return self._process_response(response, prompt)

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            return self._handle_error(val_error, prompt)

    async def __summarize_section_with_semaphore(
        self, prompt: Prompt, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.async_generate_single_sample(prompt)

    async def async_generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = []
        for prompt in prompts:
            generation_tasks.append(
                self.__summarize_section_with_semaphore(prompt, semaphore)
            )
        results = await tqdm.gather(*generation_tasks, desc=desc)
        return results

    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        self.cost = 0
        retry_count = 0
        while retry_count < self.retry_count:
            try:
                results = asyncio.run(
                    self.async_generate_multiple_samples(prompts, semaphore_size, desc)
                )
                self.logger.info(f"Cost: {self.cost:.4f}$")
                return results
            except Exception as e:
                print(e)
                retry_count += 1
                self.logger.info("Sleeping for 60 seconds...")
                time.sleep(60)
                if retry_count == self.retry_count:
                    raise e
                self.logger.info(f"Retrying {retry_count} times")
