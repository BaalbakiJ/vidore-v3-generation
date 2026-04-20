from typing import List

from pydantic import BaseModel


class Description(BaseModel):
    description: str


class Summary(BaseModel):
    summary: str


class TOCCheck(BaseModel):
    explanation: str
    has_table_of_contents: bool


class Score(BaseModel):
    grade: int
    explanation: str


class Judgment(BaseModel):
    information_richness: Score
    persona_relevance: Score
    query_generation_potential: Score
    conceptual_clarity: Score


class PairSummaryCombination(BaseModel):
    summary_1_document_id: int
    summary_1_id: int
    summary_2_document_id: int
    summary_2_id: int
    combined_summary: str


class PairSummaryCombinations(BaseModel):
    combinations: List[PairSummaryCombination]


class TripletSummaryCombination(BaseModel):
    summary_1_document_id: int
    summary_1_id: int
    summary_2_document_id: int
    summary_2_id: int
    summary_3_document_id: int
    summary_3_id: int
    combined_summary: str


class TripletSummaryCombinations(BaseModel):
    combinations: List[TripletSummaryCombination]


class CombinedSummaryGeneration(BaseModel):
    combined_summary: str


class QueryFilter(BaseModel):
    reasoning: str
    has_answer: bool


class QueryRephrase(BaseModel):
    new_query: str
