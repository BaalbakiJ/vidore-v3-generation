from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

from vidore_generation.generation_schemas import Judgment


class LLMProviderConfig(BaseModel):
    """Centralised LLM provider settings read from the config file's llm_provider key."""

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

    @model_validator(mode="after")
    def _set_defaults(self) -> "LLMProviderConfig":
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
