import random
from typing import List, Optional

import jinja2
from pydantic import BaseModel, Field

from vidore_generation.dtos import Prompt
from vidore_generation.generation_handlers.api_generation_handler import APIGenerationHandler
from vidore_generation.query_generation.vidore_juicer.query_samplers import (
    FigureModuleSampler,
    TableModuleSampler,
    TextModuleSampler,
)
from vidore_generation.query_generation.vidore_juicer.structs import (
    Query,
    SectionSummary,
)
from vidore_generation.query_generation.vidore_juicer.utils import (
    assign_number_of_questions,
)


class Queries(BaseModel):
    queries: List[Query]


async def generate_queries(
    summary: SectionSummary,
    handler: APIGenerationHandler,
    prompt_environment: jinja2.Environment,
    languages: Optional[List[str]] = ["fr"],
    max_queries_per_page: Optional[int] = 5,
    max_textual_queries_per_page: Optional[int] = 3,
    max_tabular_queries_per_page: Optional[int] = 1,
    max_visual_queries_per_page: Optional[int] = 1,
    max_adversarial_queries_per_page: Optional[int] = 1,
    seed: Optional[int] = 42,
) -> List[Query]:
    """Generates queries based on the given summary."""
    query_counts = assign_number_of_questions(
        max_queries_per_page,
        max_textual_queries_per_page,
        max_tabular_queries_per_page,
        max_visual_queries_per_page,
        max_adversarial_queries_per_page,
        seed,
    )
    query_modules = []
    query_modules.extend(TextModuleSampler.sample(query_counts["textual"], seed=seed))
    query_modules.extend(FigureModuleSampler.sample(query_counts["visual"], seed=seed))
    query_modules.extend(TableModuleSampler.sample(query_counts["tabular"], seed=seed))
    assert len(query_modules) > 0
    language = random.choice(languages)
    prompt_text = prompt_environment.get_template("query_doc.j2").render(
        language=language,
        query_modules=query_modules,
        summary=summary.summary,
    )
    prompt = Prompt(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
        arguments={"pydantic_schema": Queries},
    )
    result = await handler.async_generate_single_sample(prompt)
    if not result:
        return Queries(queries=[])
    return result


class DocumentRelevance(BaseModel):
    reasoning: str = Field(
        default="",
        description="The reasoning for the relevance of the document to the query.",
    )
    relevancy: int = Field(
        default=1,
        description="The score for the relevance of the document to the query.",
        ge=1,
        le=5,
    )  # between 1 and 5


class SelfSufficiency(BaseModel):
    reasoning: str = Field(
        default="", description="The reasoning for the self-sufficiency of the query."
    )
    self_sufficiency: int = Field(
        default=1,
        description="The score for the self-sufficiency of the query.",
        ge=1,
        le=5,
    )


class PersonaAdaptation(BaseModel):
    reasoning: str = Field(
        default="",
        description="The reasoning for the adaptation of the query to the persona.",
    )
    persona_adaptation: int = Field(
        default=1,
        description="How well the query is adapted to the persona.",
        ge=1,
        le=5,
    )


class QueryTypeFollowing(BaseModel):
    reasoning: str = Field(
        default="",
        description="The reasoning for the query type following of the query.",
    )
    follows_query_type: bool = Field(
        default=False,
        description="True if the query is following the query type, False otherwise.",
    )


class QueryFormatFollowing(BaseModel):
    reasoning: str = Field(
        default="",
        description="The reasoning for the query format following of the query.",
    )
    follows_query_format: bool = Field(
        default=False,
        description="True if the query is following the query format, False otherwise.",
    )


class Judgment(BaseModel):
    relevancy: DocumentRelevance
    self_sufficiency: SelfSufficiency
    persona_adaptation: PersonaAdaptation
    query_type_following: QueryTypeFollowing
    query_format_following: QueryFormatFollowing


async def get_judgment(
    base_model_class: BaseModel,
    summary: str,
    query: str,
    query_type: str,
    query_format: str,
    persona: str,
    prompt_environment: jinja2.Environment,
    prompt_path: str,
    handler: APIGenerationHandler,
) -> BaseModel:
    prompt_text = prompt_environment.get_template(prompt_path).render(
        summary=summary,
        query=query,
        query_type=query_type,
        query_format=query_format,
        persona=persona,
    )
    prompt = Prompt(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            }
        ],
        arguments={"pydantic_schema": base_model_class},
    )
    try:
        result = await handler.async_generate_single_sample(prompt)
    except Exception:
        return base_model_class(reasoning="error")
    if not result:
        return base_model_class(reasoning="error")
    return result


async def judge_query(
    summary: str,
    query: str,
    query_type: str,
    query_format: str,
    persona: str,
    prompt_environment: jinja2.Environment,
    handler: APIGenerationHandler,
) -> bool:
    relevancy_result = await get_judgment(
        DocumentRelevance,
        summary,
        query,
        query_type,
        query_format,
        persona,
        prompt_environment,
        "judgment/relevancy.j2",
        handler,
    )
    self_sufficiency_result = await get_judgment(
        SelfSufficiency,
        summary,
        query,
        query_type,
        query_format,
        persona,
        prompt_environment,
        "judgment/self_sufficiency.j2",
        handler,
    )
    persona_adaptation_result = await get_judgment(
        PersonaAdaptation,
        summary,
        query,
        query_type,
        query_format,
        persona,
        prompt_environment,
        "judgment/persona_adaptation.j2",
        handler,
    )
    query_type_following_result = await get_judgment(
        QueryTypeFollowing,
        summary,
        query,
        query_type,
        query_format,
        persona,
        prompt_environment,
        "judgment/type_following.j2",
        handler,
    )
    query_format_following_result = await get_judgment(
        QueryFormatFollowing,
        summary,
        query,
        query_type,
        query_format,
        persona,
        prompt_environment,
        "judgment/format_following.j2",
        handler,
    )
    return Judgment(
        relevancy=relevancy_result,
        self_sufficiency=self_sufficiency_result,
        persona_adaptation=persona_adaptation_result,
        query_type_following=query_type_following_result,
        query_format_following=query_format_following_result,
    )
