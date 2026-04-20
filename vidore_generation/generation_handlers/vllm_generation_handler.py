import asyncio
import json
import logging
import sys
import warnings
from typing import Dict, List, Optional

import litellm
from pydantic import BaseModel
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from vidore_generation.dtos import Failed
from vidore_generation.utils import post_process_output

warnings.filterwarnings("ignore", category=UserWarning)


class VLLMGenerationHandler:
    def __init__(
        self,
        model_name: str = "Qwen3-235B-A22B-GPTQ-Int4",
        prompt_template: Optional[str] = None,
        pydantic_schema: Optional[BaseModel] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.pydantic_schema = pydantic_schema

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

    def create_prompt(self, input_instance: Dict[str, str]) -> str:
        prompt = self.prompt_template.render(**input_instance)
        self.logger.debug(f"Prompt: {prompt}")
        return prompt

    def _get_completion_kwargs(self) -> Dict:
        if "qwen3" in self.model_name:
            return {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
            }
        return {}

    def _get_response_format(self) -> Dict:
        return {
            "type": "json_object",
            "json_schema": {
                "name": self.pydantic_schema.__name__,
                "schema": self.pydantic_schema.model_json_schema(),
            },
        }

    def _process_response(self, response, prompt: str):
        self.cost += litellm.completion_cost(response, model=self.model_name)
        output = self.pydantic_schema.model_validate_json(
            post_process_output(response.choices[0].message.content)
        )
        return output

    def _handle_error(self, val_error, prompt: str):
        logging.debug(
            f"Call to {self.model_name} with prompt: {prompt}\nreturned the following error:\n{val_error}"
        )
        return Failed(error=str(val_error))

    def generate_single_sample(
        self, input_instance: Dict[str, str], max_tokens: int = 10_000
    ) -> str:
        try:
            prompt = self.create_prompt(input_instance)
            response = litellm.completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=self._get_response_format(),
                stream=False,
                max_tokens=max_tokens,
                **self._get_completion_kwargs(),
            )
            return self._process_response(response, prompt)

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            return self._handle_error(val_error, prompt)

    async def generate_single_sample(self, input_instance: Dict[str, str]) -> str:
        try:
            prompt = self.create_prompt(input_instance)
            response = await litellm.acompletion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=self._get_response_format(),
                stream=False,
                max_tokens=10_000,
                **self._get_completion_kwargs(),
            )
            return self._process_response(response, prompt)

        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as val_error:
            return self._handle_error(val_error, prompt)

    async def __summarize_section_with_semaphore(
        self, input_instance: Dict[str, str], semaphore: asyncio.Semaphore
    ):
        async with semaphore:
            return await self.generate_single_sample(input_instance)

    async def async_generate_multiple_samples(
        self,
        input_instances: List[Dict[str, str]],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = []
        for input_instance in input_instances:
            generation_tasks.append(
                self.__summarize_section_with_semaphore(input_instance, semaphore)
            )
        results = await tqdm.gather(*generation_tasks, desc=desc)
        return results

    def generate_multiple_samples(
        self,
        input_instances: List[Dict[str, str]],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[BaseModel]:
        self.cost = 0
        try:
            results = asyncio.run(
                self.async_generate_multiple_samples(
                    input_instances, semaphore_size, desc
                )
            )
        except Exception as e:
            print("hello")
            print(e)
            raise e
        self.logger.info(f"Cost: {self.cost:.4f}$")
        return results
