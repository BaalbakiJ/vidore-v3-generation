import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from vidore_generation.dtos import BedrockModelPricing, Failed, Prompt
from vidore_generation.generation_handlers.bedrock_generation_handler import (
    BedrockGenerationHandler,
    BedrockUsage,
)
from vidore_generation.generation_schemas import Summary


class FakeBedrockClient:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses.copy()
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("No fake Bedrock responses remaining")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _make_response(
    response_text: str,
    usage: dict[str, Any] | None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "output": {
            "message": {
                "content": [
                    {
                        "text": response_text,
                    }
                ]
            }
        }
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _make_client_error(error_code: str) -> ClientError:
    return ClientError(
        {
            "Error": {
                "Code": error_code,
                "Message": "Fake Bedrock error",
            }
        },
        "Converse",
    )


def _make_handler(
    client: FakeBedrockClient,
    retry_count: int,
    max_concurrency: int,
    pricing_by_model: dict[str, BedrockModelPricing],
    usage_log_path: str | None,
) -> BedrockGenerationHandler:
    handler = BedrockGenerationHandler.__new__(BedrockGenerationHandler)
    handler.model_name = "test-model"
    handler.region_name = "eu-central-1"
    handler.profile_name = None
    handler.retry_count = retry_count
    handler.max_concurrency = max_concurrency
    handler.retry_initial_sleep_seconds = 0.0
    handler.retry_backoff_multiplier = 1.0
    handler.retry_max_sleep_seconds = 0.0
    handler.usage_log_path = usage_log_path
    handler.pricing_by_model = pricing_by_model
    handler.extra_kwargs = {}
    handler.logger = logging.getLogger("test-bedrock-generation-handler")
    handler.usage = BedrockUsage()
    handler._batch_usage = BedrockUsage()
    handler._usage_lock = threading.Lock()
    handler.cost = 0.0
    handler.client = client
    return handler


def _make_default_handler(client: FakeBedrockClient) -> BedrockGenerationHandler:
    return _make_handler(
        client=client,
        retry_count=1,
        max_concurrency=20,
        pricing_by_model={},
        usage_log_path=None,
    )


@pytest.mark.parametrize(
    "response_text",
    [
        '{"summary": "Bedrock works"}',
        '```json\n{"summary": "Bedrock works"}\n```',
        '```text\nHere is the JSON:\n{"summary": "Bedrock works"}\n```',
    ],
)
def test_structured_output_parses_json_payload(response_text: str) -> None:
    handler = _make_default_handler(
        FakeBedrockClient([_make_response(response_text, None)])
    )
    result = handler.generate_single_sample(
        Prompt(
            messages=[{"role": "user", "content": "Return JSON"}],
            arguments={"pydantic_schema": Summary},
        )
    )

    assert isinstance(result, Summary)
    assert result.summary == "Bedrock works"


def test_extracts_converse_usage() -> None:
    handler = _make_default_handler(FakeBedrockClient([]))
    usage = handler._extract_response_usage(
        {
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "totalTokens": 15,
            }
        }
    )

    assert usage == BedrockUsage(
        call_count=1,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
    )


def test_estimates_cost_from_configured_pricing() -> None:
    handler = _make_handler(
        client=FakeBedrockClient([]),
        retry_count=1,
        max_concurrency=20,
        pricing_by_model={
            "test-model": BedrockModelPricing(
                input_per_1k_tokens_usd=0.01,
                output_per_1k_tokens_usd=0.02,
            )
        },
        usage_log_path=None,
    )
    usage = BedrockUsage(call_count=1, input_tokens=1000, output_tokens=500)

    assert handler._estimate_usage_cost(usage) == pytest.approx(0.02)


def test_missing_pricing_returns_cost_unavailable() -> None:
    handler = _make_default_handler(FakeBedrockClient([]))
    usage = BedrockUsage(call_count=1, input_tokens=1000, output_tokens=500)

    assert handler._estimate_usage_cost(usage) is None


def test_generate_single_sample_records_usage_when_parse_succeeds() -> None:
    handler = _make_default_handler(
        FakeBedrockClient(
            [
                _make_response(
                    '{"summary": "Bedrock works"}',
                    {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                )
            ]
        )
    )
    result = handler.generate_single_sample(
        Prompt(
            messages=[{"role": "user", "content": "Return JSON"}],
            arguments={"pydantic_schema": Summary},
        )
    )

    assert isinstance(result, Summary)
    assert handler.usage.call_count == 1
    assert handler.usage.input_tokens == 10
    assert handler.usage.output_tokens == 5
    assert handler.usage.total_tokens == 15


def test_generate_single_sample_records_usage_when_parse_fails() -> None:
    handler = _make_default_handler(
        FakeBedrockClient(
            [
                _make_response(
                    "not json",
                    {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                )
            ]
        )
    )
    result = handler.generate_single_sample(
        Prompt(
            messages=[{"role": "user", "content": "Return JSON"}],
            arguments={"pydantic_schema": Summary},
        )
    )

    assert isinstance(result, Failed)
    assert handler.usage.call_count == 1
    assert handler.usage.input_tokens == 10
    assert handler.usage.output_tokens == 5
    assert handler.usage.total_tokens == 15


def test_max_concurrency_caps_semaphore_size(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler(
        client=FakeBedrockClient([]),
        retry_count=1,
        max_concurrency=2,
        pricing_by_model={},
        usage_log_path=None,
    )
    captured: dict[str, Any] = {}

    async def fake_async_generate_multiple_samples(
        prompts: list[Prompt],
        semaphore_size: int,
        desc: str,
    ) -> list[Any]:
        captured["prompts"] = prompts
        captured["semaphore_size"] = semaphore_size
        captured["desc"] = desc
        return []

    monkeypatch.setattr(
        handler,
        "async_generate_multiple_samples",
        fake_async_generate_multiple_samples,
    )

    results = handler.generate_multiple_samples([], semaphore_size=20, desc="batch")

    assert results == []
    assert captured["semaphore_size"] == 2


def test_throttling_retries_same_call(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = _make_handler(
        client=FakeBedrockClient(
            [
                _make_client_error("ThrottlingException"),
                _make_response(
                    '{"summary": "Bedrock works"}',
                    {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                ),
            ]
        ),
        retry_count=2,
        max_concurrency=20,
        pricing_by_model={},
        usage_log_path=None,
    )
    sleep_calls: list[float] = []
    monkeypatch.setattr("time.sleep", sleep_calls.append)

    result = handler.generate_single_sample(
        Prompt(
            messages=[{"role": "user", "content": "Return JSON"}],
            arguments={"pydantic_schema": Summary},
        )
    )

    assert isinstance(result, Summary)
    assert len(handler.client.calls) == 2
    assert sleep_calls == [0.0]


def test_non_throttling_client_error_raises() -> None:
    handler = _make_default_handler(
        FakeBedrockClient([_make_client_error("ValidationException")])
    )

    with pytest.raises(ClientError):
        handler.generate_single_sample(
            Prompt(
                messages=[{"role": "user", "content": "Return JSON"}],
                arguments={"pydantic_schema": Summary},
            )
        )


def test_usage_jsonl_log_records_batch_summary(tmp_path: Path) -> None:
    usage_log_path = tmp_path / "logs" / "bedrock_usage.jsonl"
    handler = _make_handler(
        client=FakeBedrockClient(
            [
                _make_response(
                    '{"summary": "Bedrock works"}',
                    {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                )
            ]
        ),
        retry_count=1,
        max_concurrency=2,
        pricing_by_model={
            "test-model": BedrockModelPricing(
                input_per_1k_tokens_usd=0.01,
                output_per_1k_tokens_usd=0.02,
            )
        },
        usage_log_path=str(usage_log_path),
    )
    prompt = Prompt(
        messages=[{"role": "user", "content": "Return JSON"}],
        arguments={"pydantic_schema": Summary},
    )

    handler.generate_multiple_samples([prompt], semaphore_size=20, desc="batch")

    records = [
        json.loads(line)
        for line in usage_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "timestamp": records[0]["timestamp"],
            "model_name": "test-model",
            "batch_description": "batch",
            "call_count": 1,
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "estimated_cost_usd": pytest.approx(0.0002),
            "pricing_available": True,
            "max_concurrency": 2,
        }
    ]


def test_finish_usage_batch_records_async_single_sample_usage(
    tmp_path: Path,
) -> None:
    usage_log_path = tmp_path / "logs" / "bedrock_usage.jsonl"
    handler = _make_handler(
        client=FakeBedrockClient(
            [
                _make_response(
                    '{"summary": "Bedrock works"}',
                    {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
                ),
                _make_response(
                    '{"summary": "Bedrock still works"}',
                    {"inputTokens": 20, "outputTokens": 15, "totalTokens": 35},
                ),
            ]
        ),
        retry_count=1,
        max_concurrency=2,
        pricing_by_model={
            "test-model": BedrockModelPricing(
                input_per_1k_tokens_usd=0.01,
                output_per_1k_tokens_usd=0.02,
            )
        },
        usage_log_path=str(usage_log_path),
    )
    prompt = Prompt(
        messages=[{"role": "user", "content": "Return JSON"}],
        arguments={"pydantic_schema": Summary},
    )

    handler.start_usage_batch()
    first_result = asyncio.run(handler.async_generate_single_sample(prompt))
    second_result = asyncio.run(handler.async_generate_single_sample(prompt))
    batch_usage = handler.finish_usage_batch("Generating queries", 1)

    assert isinstance(first_result, Summary)
    assert isinstance(second_result, Summary)
    assert batch_usage.call_count == 2
    assert batch_usage.input_tokens == 30
    assert batch_usage.output_tokens == 20
    assert batch_usage.total_tokens == 50
    assert batch_usage.estimated_cost_usd == pytest.approx(0.0007)
    records = [
        json.loads(line)
        for line in usage_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "timestamp": records[0]["timestamp"],
            "model_name": "test-model",
            "batch_description": "Generating queries",
            "call_count": 2,
            "input_tokens": 30,
            "output_tokens": 20,
            "total_tokens": 50,
            "estimated_cost_usd": pytest.approx(0.0007),
            "pricing_available": True,
            "max_concurrency": 1,
        }
    ]
