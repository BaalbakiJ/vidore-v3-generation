import json
from typing import Optional

from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
    Prompt,
)
from vidore_generation.generation_schemas import QueryRephrase


def rephrase_queries(
    environment,
    model_name,
    queries_path,
    language,
    extra_kwargs: Optional[dict] = None,
):
    with open(queries_path, "r") as f:
        queries = json.load(f)
    api_generation_handler = APIGenerationHandler(
        model_name=model_name,
        extra_kwargs=extra_kwargs or {},
    )
    template = environment.get_template("query_rephrase.j2")
    prompts = []
    for query in queries:
        prompts.append(
            Prompt(
                messages=[
                    {
                        "role": "user",
                        "content": template.render(
                            context=query["origin_summary"],
                            query=query["query"],
                            query_type=query["supposed_query_type"],
                            query_format=query["supposed_query_format"],
                            language=language,
                        ),
                    },
                ],
                arguments={"pydantic_schema": QueryRephrase},
            )
        )
    responses = api_generation_handler.generate_multiple_samples(
        prompts, desc="Rephrasing queries"
    )
    new_queries = []
    for query, response in zip(queries, responses):
        new_query = query.copy()
        new_query["rephrased_query"] = response.new_query
        new_queries.append(new_query)
    return new_queries
