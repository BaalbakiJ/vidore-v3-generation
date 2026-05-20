from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

from vidore_generation.generation_schemas import Judgment


class BedrockModelPricing(BaseModel):
    input_per_1k_tokens_usd: Optional[float] = None
    output_per_1k_tokens_usd: Optional[float] = None

    @model_validator(mode="after")
    def _validate_pricing_values(self) -> "BedrockModelPricing":
        if (
            self.input_per_1k_tokens_usd is not None
            and self.input_per_1k_tokens_usd < 0
        ):
            raise ValueError("input_per_1k_tokens_usd must be greater than or equal to 0")
        if (
            self.output_per_1k_tokens_usd is not None
            and self.output_per_1k_tokens_usd < 0
        ):
            raise ValueError(
                "output_per_1k_tokens_usd must be greater than or equal to 0"
            )
        return self


class LLMProviderConfig(BaseModel):
    """Centralised LLM provider settings read from the config file's llm_provider key."""

    provider: Literal["litellm", "bedrock"] = "litellm"
    aws_region: Optional[str] = None
    aws_profile: Optional[str] = None

    lm_model_name: str
    vl_model_name: Optional[str] = None
    # Fall back to lm_model_name when not set explicitly
    query_generation_model_name: Optional[str] = None
    judge_model_name: Optional[str] = None

    # Extra kwargs forwarded verbatim to litellm for each role
    lm_extra_kwargs: Dict[str, Any] = Field(default_factory=dict)
    vl_extra_kwargs: Dict[str, Any] = Field(default_factory=dict)
    query_generation_extra_kwargs: Dict[str, Any] = Field(default_factory=dict)
    judge_extra_kwargs: Dict[str, Any] = Field(default_factory=dict)

    bedrock_max_concurrency: int = Field(default=20, ge=1)
    bedrock_retry_count: int = Field(default=3, ge=1)
    bedrock_retry_initial_sleep_seconds: float = Field(default=60.0, ge=0)
    bedrock_retry_backoff_multiplier: float = Field(default=2.0, ge=1)
    bedrock_retry_max_sleep_seconds: float = Field(default=300.0, ge=0)
    bedrock_usage_log_path: Optional[str] = None
    bedrock_pricing: Dict[str, BedrockModelPricing] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _set_defaults(self) -> "LLMProviderConfig":
        if self.provider == "bedrock" and self.aws_region is None:
            raise ValueError("aws_region is required when provider is bedrock")
        if (
            self.bedrock_retry_max_sleep_seconds
            < self.bedrock_retry_initial_sleep_seconds
        ):
            raise ValueError(
                "bedrock_retry_max_sleep_seconds must be greater than or equal to "
                "bedrock_retry_initial_sleep_seconds"
            )
        if self.query_generation_model_name is None:
            self.query_generation_model_name = self.lm_model_name
        if self.judge_model_name is None:
            self.judge_model_name = self.lm_model_name
        return self


class DocumentDescription(BaseModel):
    document_id: UUID
    description: str


class CorpusDescription(BaseModel):
    description: str


class Document(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    content: str
    document_description: Optional[DocumentDescription] = None


class Section(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    filename: str
    document_description: str
    section: str
    page_numbers: List[int]


class TOCCheck(BaseModel):
    explanation: str
    has_table_of_contents: bool


class PageQualityCheck(BaseModel):
    explanation: str
    has_table_of_contents: bool
    is_title_only: bool
    is_blank_or_meaningless: bool


class FinalSummary(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    summary: str
    document_ids: List[UUID]
    filenames: List[str]
    page_numbers: List[List[int]]
    original_summaries: Optional[List[str]] = None
    judgments: Optional[List[Judgment]] = None
    addition_reason: Optional[str] = None


class PairSummaryCombination(BaseModel):
    summary_1_document_id: int
    summary_1_id: int
    # summary_1: str
    summary_2_document_id: int
    summary_2_id: int
    # summary_2: str
    combined_summary: str


class PairSummaryCombinations(BaseModel):
    combinations: List[PairSummaryCombination]


class TripletSummaryCombination(BaseModel):
    summary_1_document_id: int
    summary_1_id: int
    # summary_1: str
    summary_2_document_id: int
    summary_2_id: int
    # summary_2: str
    summary_3_document_id: int
    summary_3_id: int
    # summary_3: str
    combined_summary: str


class TripletSummaryCombinations(BaseModel):
    combinations: List[TripletSummaryCombination]


class IndexedSummary(BaseModel):
    summary: str
    document_id: UUID
    filename: str
    page_numbers: List[int]
    summary_id: UUID


class CombinedSummary(BaseModel):
    summaries: List[IndexedSummary]
    combined_summary: str


class ImageSection(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    filename: str
    document_description: str
    images: list[Any]
    image_paths: list[str]
    page_numbers: list[int]


class Summary(BaseModel):
    summary: str


class Failed(BaseModel):
    """
    A sentinel class used to distinguish failed request results from results
    with the value None (which may have different behavior).
    """

    error: Optional[str] = None

    def __bool__(self) -> Literal[False]:
        return False

    def __int__(self) -> int:
        return 0

    @override
    def __repr__(self) -> str:
        return "FAILED"


class Prompt(BaseModel):
    messages: List[Dict]
    arguments: Dict[str, Any]
