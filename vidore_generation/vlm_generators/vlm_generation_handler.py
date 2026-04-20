import asyncio
import json
import logging
import sys
import warnings
from typing import Any, Dict, List, Optional

import litellm
from pydantic import BaseModel
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from vidore_generation.dtos import Failed
from vidore_generation.utils import post_process_output

warnings.filterwarnings("ignore", category=UserWarning)


class Prompt(BaseModel):
    messages: List[Dict]
    arguments: Dict[str, Any]


class VLMGenerationHandler:
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        prompt_template: Optional[str] = None,
        pydantic_schema: Optional[BaseModel] = None,
        extra_kwargs: Optional[Dict] = None,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.pydantic_schema = pydantic_schema
        self.extra_kwargs = extra_kwargs or {}
        # if prompts_path is None:
        #     self.environment = Environment(
        #         loader=FileSystemLoader(files("vidore_generation").joinpath("prompts"))
        #     )
        # else:
        #     self.environment = Environment(loader=FileSystemLoader(prompts_path))

        self.logger = logging.getLogger("LLM Call Tracker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.cost = 0

    def create_prompt(self, prompt: Prompt) -> str:
        prompt = self.prompt_template.render(**prompt)
        return prompt

    def _get_completion_kwargs(self) -> Dict:
        return dict(self.extra_kwargs)

    def _get_response_format(self) -> Dict:
        return {
            "type": "json_object",
            "json_schema": {
                "name": self.pydantic_schema.__name__,
                "schema": self.pydantic_schema.model_json_schema(),
            },
        }

    def _process_response(self, response):
        self.cost += litellm.completion_cost(response, model=self.model_name)
        output = self.pydantic_schema.model_validate_json(
            post_process_output(response.choices[0].message.content)
        )
        return output

    def _handle_error(self, val_error, prompt: str):
        logging.debug(
            f"Call to {self.model_name} with prompt: {prompt}\n"
            f"returned the following error:\n{val_error}"
        )
        return Failed(error=str(val_error))

    async def generate_single_sample(self, prompt: Prompt) -> str:
        try:
            prompt = self.create_prompt(prompt)
            response = await litellm.acompletion(
                messages=prompt.messages,
                **prompt.arguments,
            )
            return self._process_response(response, prompt)

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            return self._handle_error(val_error, prompt)

    async def __generate_with_semaphore(
        self, prompt: Prompt, semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.generate_single_sample(prompt)

    async def async_generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = []
        for prompt in prompts:
            generation_tasks.append(self.__generate_with_semaphore(prompt, semaphore))
        results = await tqdm.gather(*generation_tasks, desc=desc)
        return results

    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        self.cost = 0
        try:
            results = asyncio.run(
                self.async_generate_multiple_samples(prompts, semaphore_size, desc)
            )
        except Exception as e:
            print("hello")
            print(e)
            raise e
        self.logger.info(f"Cost: {self.cost:.4f}$")
        return results
