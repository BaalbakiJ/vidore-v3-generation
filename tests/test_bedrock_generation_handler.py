import logging
from typing import Any

import pytest

from vidore_generation.dtos import Prompt
from vidore_generation.generation_handlers.bedrock_generation_handler import (
    BedrockGenerationHandler,
)
from vidore_generation.generation_schemas import Summary


class FakeBedrockClient:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "output": {
                "message": {
                    "content": [
                        {
                            "text": self.response_text,
                        }
                    ]
                }
            }
        }


def _make_handler(client: FakeBedrockClient) -> BedrockGenerationHandler:
    handler = BedrockGenerationHandler.__new__(BedrockGenerationHandler)
    handler.model_name = "test-model"
    handler.region_name = "eu-central-1"
    handler.profile_name = None
    handler.retry_count = 1
    handler.extra_kwargs = {}
    handler.logger = logging.getLogger("test-bedrock-generation-handler")
    handler.cost = 0
    handler.client = client
    return handler


@pytest.mark.parametrize(
    "response_text",
    [
        '{"summary": "Bedrock works"}',
        '```json\n{"summary": "Bedrock works"}\n```',
        '```text\nHere is the JSON:\n{"summary": "Bedrock works"}\n```',
    ],
)
def test_structured_output_parses_json_payload(response_text: str) -> None:
    handler = _make_handler(FakeBedrockClient(response_text))
    result = handler.generate_single_sample(
        Prompt(
            messages=[{"role": "user", "content": "Return JSON"}],
            arguments={"pydantic_schema": Summary},
        )
    )

    assert isinstance(result, Summary)
    assert result.summary == "Bedrock works"
