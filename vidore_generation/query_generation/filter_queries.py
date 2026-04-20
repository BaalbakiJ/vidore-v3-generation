import json
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
    Prompt,
)
from vidore_generation.generation_schemas import QueryFilter


def filter_queries(
    environment,
    model_name,
    queries_path,
    debug: bool = False,
    extra_kwargs: Optional[dict] = None,
):
    with open(queries_path, "r") as f:
        queries = json.load(f)
    api_generation_handler = APIGenerationHandler(
        model_name=model_name,
        extra_kwargs=extra_kwargs or {},
    )
    template = environment.get_template("query_filter.j2")
    prompts = []
    filtered_queries = []
    for query in queries:
        prompts.append(
            Prompt(
                messages=[
                    {"role": "user", "content": template.render(query=query["query"])},
                ],
                arguments={"pydantic_schema": QueryFilter},
            )
        )
    responses = api_generation_handler.generate_multiple_samples(
        prompts, desc="Filtering queries"
    )
    for query, response in zip(queries, responses):
        if not response.has_answer:
            filtered_queries.append(query)
        elif debug:
            print(f"Query: {query['query']}")
            print(f"Response: {response}")
            print("-" * 100)
    print(f"Kept {len(filtered_queries)} queries out of {len(queries)}")
    with open(queries_path.replace(".json", "_filtered.json"), "w") as f:
        json.dump(filtered_queries, f)
    return filtered_queries


if __name__ == "__main__":
    environment = Environment(loader=FileSystemLoader("vidore_generation/prompts"))
    filter_queries(
        environment,
        "fireworks_ai/kimi-k2p5",
        "data/vidore_v3/healthcare_fda_en/queries/vidore_juicer_healthcare_fda_en_queries.json",
        debug=True,
    )
