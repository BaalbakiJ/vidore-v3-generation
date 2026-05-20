import logging
from typing import Any, Literal

from vidore_generation.dtos import LLMProviderConfig
from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
)
from vidore_generation.generation_handlers.generation_handler import GenerationHandler


GenerationRole = Literal["lm", "vl", "query_generation", "judge"]


def _require_model_name(model_name: str | None, role: GenerationRole) -> str:
    if model_name is None:
        raise ValueError(f"No model configured for generation role: {role}")
    return model_name


def _select_model_and_kwargs(
    llm_provider: LLMProviderConfig,
    role: GenerationRole,
) -> tuple[str, dict[str, Any]]:
    if role == "lm":
        return llm_provider.lm_model_name, llm_provider.lm_extra_kwargs
    if role == "vl":
        return (
            llm_provider.vl_model_name or llm_provider.lm_model_name,
            llm_provider.vl_extra_kwargs,
        )
    if role == "query_generation":
        return (
            _require_model_name(llm_provider.query_generation_model_name, role),
            llm_provider.query_generation_extra_kwargs,
        )
    if role == "judge":
        return (
            _require_model_name(llm_provider.judge_model_name, role),
            llm_provider.judge_extra_kwargs,
        )
    raise ValueError(f"Unsupported generation role: {role}")


def make_generation_handler(
    llm_provider: LLMProviderConfig,
    role: GenerationRole,
    logger: logging.Logger | None = None,
) -> GenerationHandler:
    model_name, extra_kwargs = _select_model_and_kwargs(llm_provider, role)
    if llm_provider.provider == "bedrock":
        if llm_provider.aws_region is None:
            raise ValueError("aws_region is required when provider is bedrock")
        from vidore_generation.generation_handlers.bedrock_generation_handler import (
            BedrockGenerationHandler,
        )

        return BedrockGenerationHandler(
            model_name=model_name,
            region_name=llm_provider.aws_region,
            profile_name=llm_provider.aws_profile,
            logger=logger,
            retry_count=llm_provider.bedrock_retry_count,
            max_concurrency=llm_provider.bedrock_max_concurrency,
            retry_initial_sleep_seconds=(
                llm_provider.bedrock_retry_initial_sleep_seconds
            ),
            retry_backoff_multiplier=llm_provider.bedrock_retry_backoff_multiplier,
            retry_max_sleep_seconds=llm_provider.bedrock_retry_max_sleep_seconds,
            usage_log_path=llm_provider.bedrock_usage_log_path,
            pricing_by_model=llm_provider.bedrock_pricing,
            extra_kwargs=extra_kwargs,
        )

    return APIGenerationHandler(
        model_name=model_name,
        logger=logger,
        extra_kwargs=extra_kwargs,
    )
