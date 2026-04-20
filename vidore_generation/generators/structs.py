from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, Field


class SectionSummary(BaseModel):
    summary: str = Field(description="Summary of the section")


class Query(BaseModel):
    query_type: str = Field(description="Type of query")
    query_format: str = Field(description="Format of the query")
    language: str = Field(description="Language of the query")
    query: str = Field(description="Query")


class Query(BaseModel):
    query_type: str = Field(description="Type of query")
    query_format: str = Field(description="Format of the query")
    language: str = Field(description="Language of the query")
    query: str = Field(description="Query")


class Answer(BaseModel):
    page_id: str = Field(description="ID of the page related to the query answer")
    is_answerable: str = Field(
        description=(
            "Whether the query is answerable or not. Can be 'fully answerable', 'partially answerable'"
            " or 'unanswerable'."
        )
    )
    answer: str = Field(description="Answer of the query with respect to the page")


class QRels(BaseModel):
    query: str = Field(description="The query")
    affiliation: Answer = Field(
        description="Page affiliation of the query and its answer."
    )


class PageTag(BaseModel):
    page_id: str = Field(description="ID of the page")
    page_content: str = Field(description="Content of the page")


class PageRel(BaseModel):
    page_id: str = Field(description="ID of the page")
    base_path: str = Field(description="Base path of the page")
    relevance: str = Field(
        description="Relevance of the page, can either be 'fully answerable', 'partially answerable' or 'unanswerable'."
    )
    query: str = Field(description="The query")
    answer: str = Field(description="Answer of the query with respect to the page")


class QueryIded(BaseModel):
    query_id: str = Field(description="ID of the query")
    query: str = Field(description="Query")


class DatasetQrels(BaseModel):
    query_id: str = Field(description="ID of the query")
    page_id: int = Field(description="ID of the page")
    is_answerable: str = Field(
        description=(
            "Whether the query is answerable or not. Can be 'fully answerable', 'partially answerable'"
            " or 'unanswerable'."
        )
    )
    answer: str = Field(description="Answer of the query with respect to the page")
    score: int = Field(description="Score of the query")


class QueryType(Enum):
    EXTRACTIVE = "extractive"
    OPEN_ENDED = "open-ended"
    COMPARE_CONTRAST = "compare-contrast"
    NUMERICAL = "numerical"
    BOOLEAN = "boolean"
    ENUMERATIVE = "enumerative"
    MULTI_HOP = "multi-hop"
    ANY = "any"


class QueryFormat(Enum):
    QUESTION = "question"
    KEYWORD = "keyword"
    INSTRUCTION = "instruction"
    ANY = "any"


class Answerability(Enum):
    FULL = "fully answerable"
    PARTIAL = "partially answerable"
    ADVERSARIAL = "unanswerable"


class Modality(Enum):
    TEXT = "text"
    FIGURE = "figure"
    TABLE = "table"
    OTHER = "other"
    ANY = "any"


@dataclass(kw_only=True)
class QueryModule:
    type: QueryType
    format: QueryFormat
    answerability: Answerability
    modality: Modality
    instruction: str
