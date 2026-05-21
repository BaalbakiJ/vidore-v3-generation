import asyncio
import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time
import warnings
from typing import Any, Literal, TypedDict

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from pydantic import BaseModel
from pydantic_core import ValidationError
from tqdm.asyncio import tqdm

from vidore_generation.dtos import BedrockModelPricing, Failed, Prompt
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


@dataclass(frozen=True)
class BedrockUsage:
    call_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = 0.0


THROTTLING_ERROR_CODES = {
    "ThrottlingException",
    "TooManyRequestsException",
    "Throttling",
    "RequestLimitExceeded",
    "ServiceQuotaExceededException",
}


def _is_throttling_error(error: Exception) -> bool:
    if not isinstance(error, ClientError):
        return False
    error_code = error.response.get("Error", {}).get("Code")
    return error_code in THROTTLING_ERROR_CODES


def _combine_costs(
    current_cost: float | None,
    added_cost: float | None,
) -> float | None:
    if current_cost is None or added_cost is None:
        return None
    return current_cost + added_cost


def _combine_usage(
    current_usage: BedrockUsage,
    added_usage: BedrockUsage,
) -> BedrockUsage:
    return BedrockUsage(
        call_count=current_usage.call_count + added_usage.call_count,
        input_tokens=current_usage.input_tokens + added_usage.input_tokens,
        output_tokens=current_usage.output_tokens + added_usage.output_tokens,
        total_tokens=current_usage.total_tokens + added_usage.total_tokens,
        estimated_cost_usd=_combine_costs(
            current_usage.estimated_cost_usd,
            added_usage.estimated_cost_usd,
        ),
    )


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
        max_concurrency: int = 20,
        retry_initial_sleep_seconds: float = 60.0,
        retry_backoff_multiplier: float = 2.0,
        retry_max_sleep_seconds: float = 300.0,
        usage_log_path: str | None = None,
        pricing_by_model: dict[str, BedrockModelPricing] | None = None,
        extra_kwargs: dict[str, Any] | None = None,
    ):
        self.model_name = model_name
        self.region_name = region_name
        self.profile_name = profile_name
        self.retry_count = retry_count
        self.max_concurrency = max_concurrency
        self.retry_initial_sleep_seconds = retry_initial_sleep_seconds
        self.retry_backoff_multiplier = retry_backoff_multiplier
        self.retry_max_sleep_seconds = retry_max_sleep_seconds
        self.usage_log_path = usage_log_path
        self.pricing_by_model = pricing_by_model or {}
        self.extra_kwargs = extra_kwargs or {}
        self.logger = logger if logger is not None else _make_default_logger()
        self.usage = BedrockUsage()
        self._batch_usage = BedrockUsage()
        self._usage_lock = threading.Lock()
        self.cost = 0.0

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

    def _extract_usage_token_count(
        self,
        usage_data: dict[str, Any],
        usage_key: str,
    ) -> int:
        raw_value = usage_data.get(usage_key)
        if raw_value is None:
            return 0
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            if raw_value >= 0:
                return raw_value
        self.logger.warning(
            "Invalid Bedrock usage token count",
            extra={
                "model_name": self.model_name,
                "usage_key": usage_key,
                "usage_value": raw_value,
            },
        )
        return 0

    def _extract_response_usage(self, response: dict[str, Any]) -> BedrockUsage:
        usage_data = response.get("usage")
        if usage_data is None:
            self.logger.debug(
                "Bedrock response missing usage",
                extra={"model_name": self.model_name},
            )
            return BedrockUsage(call_count=1)
        if not isinstance(usage_data, dict):
            self.logger.warning(
                "Invalid Bedrock usage payload",
                extra={"model_name": self.model_name, "usage_value": usage_data},
            )
            return BedrockUsage(call_count=1)

        input_tokens = self._extract_usage_token_count(usage_data, "inputTokens")
        output_tokens = self._extract_usage_token_count(usage_data, "outputTokens")
        if "totalTokens" in usage_data:
            total_tokens = self._extract_usage_token_count(usage_data, "totalTokens")
        else:
            total_tokens = input_tokens + output_tokens

        return BedrockUsage(
            call_count=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def _estimate_usage_cost(self, usage: BedrockUsage) -> float | None:
        pricing = self.pricing_by_model.get(self.model_name)
        if pricing is None:
            return None
        if (
            pricing.input_per_1k_tokens_usd is None
            or pricing.output_per_1k_tokens_usd is None
        ):
            return None
        input_cost = usage.input_tokens / 1000 * pricing.input_per_1k_tokens_usd
        output_cost = usage.output_tokens / 1000 * pricing.output_per_1k_tokens_usd
        return input_cost + output_cost

    def _model_pricing_is_available(self) -> bool:
        pricing = self.pricing_by_model.get(self.model_name)
        if pricing is None:
            return False
        return (
            pricing.input_per_1k_tokens_usd is not None
            and pricing.output_per_1k_tokens_usd is not None
        )

    def _record_response_usage(self, response: dict[str, Any]) -> BedrockUsage:
        extracted_usage = self._extract_response_usage(response)
        estimated_cost_usd = self._estimate_usage_cost(extracted_usage)
        response_usage = BedrockUsage(
            call_count=extracted_usage.call_count,
            input_tokens=extracted_usage.input_tokens,
            output_tokens=extracted_usage.output_tokens,
            total_tokens=extracted_usage.total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )
        with self._usage_lock:
            self.usage = _combine_usage(self.usage, response_usage)
            self._batch_usage = _combine_usage(self._batch_usage, response_usage)
            self.cost = self.usage.estimated_cost_usd or 0.0
        return response_usage

    def _get_current_batch_usage(self) -> BedrockUsage:
        with self._usage_lock:
            return self._batch_usage

    def _reset_current_batch_usage(self) -> None:
        with self._usage_lock:
            self._batch_usage = BedrockUsage()

    def start_usage_batch(self) -> None:
        self._reset_current_batch_usage()

    def finish_usage_batch(
        self,
        batch_description: str,
        effective_max_concurrency: int | None = None,
    ) -> BedrockUsage:
        batch_usage = self._get_current_batch_usage()
        concurrency = (
            effective_max_concurrency
            if effective_max_concurrency is not None
            else self.max_concurrency
        )
        self._log_usage_summary(batch_usage, batch_description)
        self._append_usage_log_record(batch_usage, batch_description, concurrency)
        return batch_usage

    def _get_retry_sleep_seconds(self, attempt_index: int) -> float:
        sleep_seconds = self.retry_initial_sleep_seconds * (
            self.retry_backoff_multiplier**attempt_index
        )
        return min(sleep_seconds, self.retry_max_sleep_seconds)

    def _call_converse_with_retries(self, prompt: Prompt) -> dict[str, Any]:
        converse_kwargs = self._get_converse_kwargs(prompt)
        for attempt_index in range(self.retry_count):
            try:
                return self.client.converse(**converse_kwargs)
            except Exception as error:
                if not _is_throttling_error(error):
                    raise
                if attempt_index == self.retry_count - 1:
                    raise
                sleep_seconds = self._get_retry_sleep_seconds(attempt_index)
                self.logger.warning(
                    "Retrying throttled Bedrock Converse call",
                    extra={
                        "model_name": self.model_name,
                        "attempt": attempt_index + 1,
                        "retry_limit": self.retry_count,
                        "sleep_seconds": sleep_seconds,
                    },
                    exc_info=True,
                )
                time.sleep(sleep_seconds)

        raise RuntimeError("Bedrock Converse retry loop exited unexpectedly")

    def _log_usage_summary(
        self,
        batch_usage: BedrockUsage,
        batch_description: str,
    ) -> None:
        if (
            not self._model_pricing_is_available()
            or batch_usage.estimated_cost_usd is None
        ):
            estimated_cost = "unavailable_missing_pricing"
        else:
            estimated_cost = f"{batch_usage.estimated_cost_usd:.4f}"
        self.logger.info(
            (
                "Bedrock usage summary: model=%s batch=%r calls=%s "
                "input_tokens=%s output_tokens=%s total_tokens=%s "
                "estimated_cost_usd=%s"
            ),
            self.model_name,
            batch_description,
            batch_usage.call_count,
            batch_usage.input_tokens,
            batch_usage.output_tokens,
            batch_usage.total_tokens,
            estimated_cost,
            extra={
                "model_name": self.model_name,
                "batch_description": batch_description,
                "call_count": batch_usage.call_count,
                "input_tokens": batch_usage.input_tokens,
                "output_tokens": batch_usage.output_tokens,
                "total_tokens": batch_usage.total_tokens,
                "estimated_cost_usd": batch_usage.estimated_cost_usd,
            },
        )

    def _append_usage_log_record(
        self,
        batch_usage: BedrockUsage,
        batch_description: str,
        effective_max_concurrency: int,
    ) -> None:
        if self.usage_log_path is None:
            return

        pricing_available = self._model_pricing_is_available()
        usage_log_path = Path(self.usage_log_path)
        if usage_log_path.parent != Path("."):
            usage_log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "model_name": self.model_name,
            "batch_description": batch_description,
            "call_count": batch_usage.call_count,
            "input_tokens": batch_usage.input_tokens,
            "output_tokens": batch_usage.output_tokens,
            "total_tokens": batch_usage.total_tokens,
            "estimated_cost_usd": batch_usage.estimated_cost_usd
            if pricing_available
            else None,
            "pricing_available": pricing_available,
            "max_concurrency": effective_max_concurrency,
        }
        with usage_log_path.open("a", encoding="utf-8") as usage_log_file:
            usage_log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_single_sample(self, prompt: Prompt) -> Any:
        response = self._call_converse_with_retries(prompt)
        self._record_response_usage(response)
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
        self.cost = 0.0
        self.start_usage_batch()
        effective_semaphore_size = min(semaphore_size, self.max_concurrency)
        self.logger.debug(
            "Bedrock request concurrency configured",
            extra={
                "model_name": self.model_name,
                "requested_semaphore_size": semaphore_size,
                "effective_semaphore_size": effective_semaphore_size,
                "max_concurrency": self.max_concurrency,
            },
        )
        results = asyncio.run(
            self.async_generate_multiple_samples(
                prompts,
                effective_semaphore_size,
                desc,
            )
        )
        self.finish_usage_batch(desc, effective_semaphore_size)
        return results
