import asyncio
import base64
import binascii
import json
import logging
import sys
import time
import warnings
from typing import Any, Literal, TypedDict

import boto3
from botocore.config import Config
from pydantic import BaseModel
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from vidore_generation.dtos import Failed, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.utils import post_process_output

warnings.filterwarnings("ignore", category=UserWarning)


BedrockImageFormat = Literal["jpeg", "png"]


class BedrockTextBlock(TypedDict):
    text: str


class BedrockImageSource(TypedDict):
    bytes: bytes


class BedrockImage(TypedDict):
    format: BedrockImageFormat
    source: BedrockImageSource


class BedrockImageBlock(TypedDict):
    image: BedrockImage


BedrockContentBlock = BedrockTextBlock | BedrockImageBlock


class BedrockMessage(TypedDict):
    role: str
    content: list[BedrockContentBlock]


class BedrockSystemBlock(TypedDict):
    text: str


def _make_default_logger() -> logging.Logger:
    logger = logging.getLogger("Bedrock Call Tracker")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _decode_image_data_url(image_url: str) -> tuple[BedrockImageFormat, bytes]:
    header, separator, encoded_image = image_url.partition(",")
    if separator == "":
        raise ValueError("Image URL must be a base64 data URL with a comma separator")
    if not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError(
            f"Unsupported image URL header for Bedrock Converse: {header}"
        )

    image_format = header.removeprefix("data:image/").split(";")[0].lower()
    if image_format == "jpg":
        image_format = "jpeg"
    if image_format not in {"jpeg", "png"}:
        raise ValueError(
            f"Unsupported Bedrock image format: {image_format}. "
            "Supported formats are jpeg, jpg, and png."
        )

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except binascii.Error as error:
        raise ValueError("Invalid base64 image payload for Bedrock Converse") from error

    return image_format, image_bytes


def _map_text_content(content: str) -> BedrockTextBlock:
    return {"text": content}


def _map_image_content(content: dict[str, Any]) -> BedrockImageBlock:
    image_url = content.get("image_url")
    if not isinstance(image_url, str):
        raise ValueError("Bedrock image content requires image_url to be a string")
    image_format, image_bytes = _decode_image_data_url(image_url)
    return {
        "image": {
            "format": image_format,
            "source": {"bytes": image_bytes},
        }
    }


def _map_list_content(content: list[dict[str, Any]]) -> list[BedrockContentBlock]:
    mapped_content: list[BedrockContentBlock] = []
    for content_item in content:
        content_type = content_item.get("type")
        if content_type == "text":
            text = content_item.get("text")
            if not isinstance(text, str):
                raise ValueError("Bedrock text content requires text to be a string")
            mapped_content.append(_map_text_content(text))
        elif content_type == "image_url":
            mapped_content.append(_map_image_content(content_item))
        else:
            raise ValueError(f"Unsupported Bedrock content type: {content_type}")
    return mapped_content


def _map_message_content(
    content: str | list[dict[str, Any]],
) -> list[BedrockContentBlock]:
    if isinstance(content, str):
        return [_map_text_content(content)]
    if isinstance(content, list):
        return _map_list_content(content)
    raise ValueError(f"Unsupported Bedrock message content type: {type(content)}")


def _append_instruction_to_content(
    content: str | list[dict[str, Any]],
    instruction: str,
) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return f"{content}\n\n{instruction}"
    if isinstance(content, list):
        return [*content, {"type": "text", "text": instruction}]
    raise ValueError(f"Unsupported prompt content type: {type(content)}")


def _append_structured_output_instruction(
    messages: list[dict[str, Any]],
    pydantic_schema: type[BaseModel],
) -> list[dict[str, Any]]:
    schema_json = json.dumps(pydantic_schema.model_json_schema(), ensure_ascii=False)
    instruction = (
        "Return only valid JSON matching this JSON Schema. "
        "Do not include markdown fences, comments, or explanatory text.\n"
        f"JSON Schema: {schema_json}"
    )
    copied_messages = [message.copy() for message in messages]
    for message_index in range(len(copied_messages) - 1, -1, -1):
        if copied_messages[message_index].get("role") == "user":
            copied_messages[message_index]["content"] = _append_instruction_to_content(
                copied_messages[message_index]["content"],
                instruction,
            )
            return copied_messages

    return [*copied_messages, {"role": "user", "content": instruction}]


def _map_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[BedrockMessage], list[BedrockSystemBlock]]:
    bedrock_messages: list[BedrockMessage] = []
    system_blocks: list[BedrockSystemBlock] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            for content_block in _map_message_content(content):
                if "text" not in content_block:
                    raise ValueError(
                        "Bedrock system messages only support text content"
                    )
                system_blocks.append({"text": content_block["text"]})
        elif role in {"user", "assistant"}:
            bedrock_messages.append(
                {
                    "role": role,
                    "content": _map_message_content(content),
                }
            )
        else:
            raise ValueError(f"Unsupported Bedrock message role: {role}")
    return bedrock_messages, system_blocks


def _normalize_stop_sequences(stop: Any) -> list[str]:
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list) and all(isinstance(item, str) for item in stop):
        return stop
    raise ValueError("Bedrock stop must be a string or a list of strings")


def _build_inference_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    inference_config: dict[str, Any] = {}
    key_mapping = {
        "temperature": "temperature",
        "top_p": "topP",
        "stop": "stopSequences",
    }
    supported_keys = {
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
    }
    unsupported_keys = sorted(set(kwargs) - supported_keys)
    if unsupported_keys:
        raise ValueError(
            f"Unsupported Bedrock generation argument(s): {unsupported_keys}"
        )

    for source_key, target_key in key_mapping.items():
        if source_key not in kwargs:
            continue
        if source_key == "stop":
            inference_config[target_key] = _normalize_stop_sequences(kwargs[source_key])
        else:
            inference_config[target_key] = kwargs[source_key]

    if "max_tokens" in kwargs:
        inference_config["maxTokens"] = kwargs["max_tokens"]
    elif "max_completion_tokens" in kwargs:
        inference_config["maxTokens"] = kwargs["max_completion_tokens"]

    return inference_config


def _extract_response_text(response: dict[str, Any]) -> str:
    content = response.get("output", {}).get("message", {}).get("content", [])
    text_blocks = [
        content_block["text"]
        for content_block in content
        if isinstance(content_block, dict)
        and isinstance(content_block.get("text"), str)
    ]
    if not text_blocks:
        raise ValueError(f"Bedrock response did not contain text content: {response}")
    return "".join(text_blocks)


def _extract_json_payload(text: str) -> str:
    output = post_process_output(text).strip()
    if output.startswith("```"):
        output = output.removeprefix("```").strip()
        if output.endswith("```"):
            output = output[:-3].strip()
        if not output.startswith("{"):
            output_parts = output.split(maxsplit=1)
            if len(output_parts) == 2 and output_parts[0].isalpha():
                output = output_parts[1].strip()

    json_start_index = output.find("{")
    json_end_index = output.rfind("}")
    if json_start_index != -1 and json_end_index > json_start_index:
        return output[json_start_index : json_end_index + 1].strip()
    return output


class BedrockGenerationHandler(GenerationHandler):
    def __init__(
        self,
        model_name: str,
        region_name: str,
        profile_name: str | None = None,
        logger: logging.Logger | None = None,
        retry_count: int = 3,
        extra_kwargs: dict | None = None,
    ):
        self.model_name = model_name
        self.region_name = region_name
        self.profile_name = profile_name
        self.retry_count = retry_count
        self.extra_kwargs = extra_kwargs or {}
        self.logger = logger if logger is not None else _make_default_logger()
        self.cost = 0

        session_kwargs = {"region_name": region_name}
        if profile_name is not None:
            session_kwargs["profile_name"] = profile_name
        session = boto3.Session(**session_kwargs)
        client_config = Config(
            read_timeout=3600,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        self.client = session.client("bedrock-runtime", config=client_config)

    def _get_converse_kwargs(self, prompt: Prompt) -> dict[str, Any]:
        arguments = prompt.arguments.copy()
        pydantic_schema = arguments.pop("pydantic_schema", None)
        for key, value in self.extra_kwargs.items():
            if key not in arguments:
                arguments[key] = value

        messages = prompt.messages
        if pydantic_schema is not None:
            messages = _append_structured_output_instruction(
                prompt.messages,
                pydantic_schema,
            )

        bedrock_messages, system_blocks = _map_messages(messages)
        converse_kwargs: dict[str, Any] = {
            "modelId": self.model_name,
            "messages": bedrock_messages,
        }
        inference_config = _build_inference_config(arguments)
        if inference_config:
            converse_kwargs["inferenceConfig"] = inference_config
        if system_blocks:
            converse_kwargs["system"] = system_blocks
        return converse_kwargs

    def _process_response(self, response: dict[str, Any], prompt: Prompt) -> Any:
        response_text = _extract_response_text(response)
        if "pydantic_schema" in prompt.arguments:
            cleaned_output = _extract_json_payload(response_text)
            return prompt.arguments["pydantic_schema"].model_validate_json(
                cleaned_output
            )
        return post_process_output(response_text)

    def _handle_error(self, error: Exception, prompt: Prompt) -> Failed:
        self.logger.debug(
            "Bedrock call returned a parsing or validation error",
            extra={"model_name": self.model_name, "prompt": prompt.model_dump()},
        )
        return Failed(error=str(error))

    def generate_single_sample(self, prompt: Prompt) -> Any:
        response = self.client.converse(**self._get_converse_kwargs(prompt))
        try:
            return self._process_response(response, prompt)
        except (ValidationError, json.decoder.JSONDecodeError, ValueError) as error:
            return self._handle_error(error, prompt)

    async def async_generate_single_sample(self, prompt: Prompt) -> Any:
        return await asyncio.to_thread(self.generate_single_sample, prompt)

    async def __generate_single_sample_with_semaphore(
        self,
        prompt: Prompt,
        semaphore: asyncio.Semaphore,
    ) -> Any:
        async with semaphore:
            return await self.async_generate_single_sample(prompt)

    async def async_generate_multiple_samples(
        self,
        prompts: list[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> list[Any]:
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = [
            self.__generate_single_sample_with_semaphore(prompt, semaphore)
            for prompt in prompts
        ]
        return await tqdm.gather(*generation_tasks, desc=desc)

    def generate_multiple_samples(
        self,
        prompts: list[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> list[Any]:
        self.cost = 0
        retry_count = 0
        while retry_count < self.retry_count:
            try:
                results = asyncio.run(
                    self.async_generate_multiple_samples(prompts, semaphore_size, desc)
                )
                self.logger.info(
                    "Bedrock cost tracking not implemented",
                    extra={"model_name": self.model_name, "cost": self.cost},
                )
                return results
            except Exception as error:
                retry_count += 1
                self.logger.warning(
                    "Retrying Bedrock generation after error",
                    extra={
                        "model_name": self.model_name,
                        "retry_count": retry_count,
                        "retry_limit": self.retry_count,
                    },
                    exc_info=True,
                )
                if retry_count == self.retry_count:
                    raise error
                time.sleep(60)

        raise RuntimeError("Bedrock generation retry loop exited unexpectedly")
