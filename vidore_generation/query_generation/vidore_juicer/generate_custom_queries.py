import asyncio
import json
import logging
import os
import sys
import time
from typing import Callable, List, Optional, cast

import jinja2
import litellm
import yaml
from pydantic import BaseModel
from tqdm.asyncio import tqdm as async_tqdm

from vidore_generation.dtos import LLMProviderConfig
from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
)
from vidore_generation.generation_handlers.factory import make_generation_handler
from vidore_generation.query_generation.vidore_juicer.generate_queries import (
    Judgment,
    Queries,
    generate_queries,
    judge_query,
)
from vidore_generation.query_generation.vidore_juicer.structs import (
    SectionSummary,
)


class QueryToJudge(BaseModel):
    query: str
    summary: str
    supposed_query_type: str
    supposed_query_format: str
    document_ids: List[str]
    filenames: List[str]
    page_numbers: List[List[int]]
    judgment: Optional[Judgment] = None


def _start_usage_batch_if_supported(handler: object) -> None:
    start_usage_batch = getattr(handler, "start_usage_batch", None)
    if callable(start_usage_batch):
        typed_start_usage_batch = cast(Callable[[], None], start_usage_batch)
        typed_start_usage_batch()


def _get_effective_batch_concurrency(
    handler: object,
    requested_concurrency: int,
) -> int:
    max_concurrency = getattr(handler, "max_concurrency", requested_concurrency)
    if not isinstance(max_concurrency, int):
        raise TypeError("Batch usage max_concurrency must be an integer")
    return min(requested_concurrency, max_concurrency)


def _finish_usage_batch_if_supported(
    handler: object,
    batch_description: str,
    requested_concurrency: int,
) -> None:
    finish_usage_batch = getattr(handler, "finish_usage_batch", None)
    if not callable(finish_usage_batch):
        return
    effective_concurrency = _get_effective_batch_concurrency(
        handler,
        requested_concurrency,
    )
    typed_finish_usage_batch = cast(
        Callable[[str, int], object],
        finish_usage_batch,
    )
    typed_finish_usage_batch(batch_description, effective_concurrency)


class QueryGenerator:
    def __init__(
        self,
        prompt_environment: jinja2.Environment,
        persona: str,
        lm_model_name: str,
        judge_model_name: str,
        retry_count: int = 3,
        language: str = "english",
        query_generation_extra_kwargs: Optional[dict] = None,
        judge_extra_kwargs: Optional[dict] = None,
        llm_provider: Optional[LLMProviderConfig] = None,
    ):
        self.prompt_environment = prompt_environment
        self.persona = persona
        self.retry_count = retry_count
        self.language = language

        if llm_provider is not None:
            self.query_handler = make_generation_handler(
                llm_provider,
                "query_generation",
            )
            self.judge_handler = make_generation_handler(
                llm_provider,
                "judge",
            )
        else:
            self.query_handler = APIGenerationHandler(
                model_name=lm_model_name,
                extra_kwargs=query_generation_extra_kwargs or {},
            )
            self.judge_handler = APIGenerationHandler(
                model_name=judge_model_name,
                extra_kwargs=judge_extra_kwargs or {},
            )

        self.logger = logging.getLogger("LLM Call Tracker")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    async def __async_generate_queries(
        self,
        summary: SectionSummary,
        semaphore: asyncio.Semaphore,
        seed: int = 42,
    ):
        async with semaphore:
            return await generate_queries(
                summary,
                self.query_handler,
                self.prompt_environment,
                [self.language],
                max_queries_per_page=3,
                seed=seed,
            )

    async def async_generate_queries(
        self,
        summaries: List[SectionSummary],
        semaphore_size: int,
        desc: str,
    ):
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = []
        for i, summary in enumerate(summaries):
            generation_tasks.append(
                self.__async_generate_queries(summary, semaphore, seed=i)
            )
        results = await async_tqdm.gather(*generation_tasks, desc=desc)
        return results

    def generate_queries(
        self,
        summaries: List[str],
    ):
        requested_concurrency = 50
        batch_description = "Generating queries"
        retry_count = 0
        while retry_count < self.retry_count:
            try:
                _start_usage_batch_if_supported(self.query_handler)
                results = asyncio.run(
                    self.async_generate_queries(
                        [SectionSummary(summary=summary) for summary in summaries],
                        requested_concurrency,
                        batch_description,
                    )
                )
                _finish_usage_batch_if_supported(
                    self.query_handler,
                    batch_description,
                    requested_concurrency,
                )
                self.logger.info(f"Cost: {self.query_handler.cost:.4f}$")
                return results
            except Exception as e:
                print(e)
                retry_count += 1
                self.logger.info("Sleeping for 60 seconds...")
                time.sleep(60)
                if retry_count == self.retry_count:
                    raise e
                self.logger.info(f"Retrying {retry_count} times")

    async def __async_generate_judgments(
        self,
        query_to_judge: QueryToJudge,
        semaphore: asyncio.Semaphore,
        seed: int = 42,
    ):
        async with semaphore:
            return await judge_query(
                query_to_judge.summary,
                query_to_judge.query,
                query_to_judge.supposed_query_type,
                query_to_judge.supposed_query_format,
                self.persona,
                self.prompt_environment,
                self.judge_handler,
            )

    async def async_generate_judgments(
        self,
        queries_to_judge: List[QueryToJudge],
        semaphore_size: int,
        desc: str,
    ):
        semaphore = asyncio.Semaphore(semaphore_size)
        generation_tasks = []
        for i, query_to_judge in enumerate(queries_to_judge):
            generation_tasks.append(
                self.__async_generate_judgments(query_to_judge, semaphore, seed=i)
            )
        results = await async_tqdm.gather(*generation_tasks, desc=desc)
        return results

    def judge_queries(
        self,
        queries_to_judge: List[QueryToJudge],
    ):
        requested_concurrency = 2
        batch_description = "Generating judgments"
        retry_count = 0
        while retry_count < self.retry_count:
            try:
                _start_usage_batch_if_supported(self.judge_handler)
                results = asyncio.run(
                    self.async_generate_judgments(
                        queries_to_judge,
                        requested_concurrency,
                        batch_description,
                    )
                )
                _finish_usage_batch_if_supported(
                    self.judge_handler,
                    batch_description,
                    requested_concurrency,
                )
                self.logger.info(f"Cost: {self.judge_handler.cost:.4f}$")
                return results
            except Exception as e:
                retry_count += 1
                self.logger.info(f"Retrying {retry_count}/{self.retry_count} after error: {e}")
                time.sleep(60)
                if retry_count == self.retry_count:
                    raise e


def run_vidore_juicer_generation(input_file, generation_config_path):
    litellm.drop_params = True
    litellm.enable_cache("disk", disk_cache_dir=".litellm_cache/")

    prompt_environment = jinja2.Environment(
        loader=jinja2.FileSystemLoader("vidore_generation/query_generation/vidore_juicer/prompts")
    )
    with open(generation_config_path, "r") as f:
        generation_config = yaml.safe_load(f)
    llm_provider = LLMProviderConfig(**generation_config["llm_provider"])
    persona = generation_config["persona"]
    language = generation_config["language"]
    query_generator = QueryGenerator(
        prompt_environment,
        persona,
        lm_model_name=llm_provider.query_generation_model_name,
        judge_model_name=llm_provider.judge_model_name,
        language=language,
        query_generation_extra_kwargs=llm_provider.query_generation_extra_kwargs,
        judge_extra_kwargs=llm_provider.judge_extra_kwargs,
        llm_provider=llm_provider,
    )
    with open(input_file, "r") as f:
        summaries = json.load(f)
    summarie_strs = [summary["summary"] for summary in summaries]
    prefix = generation_config_path.split("/")[-1].split(".")[0]
    output_dir = os.path.join(generation_config["documents_dir"], generation_config["dataset_name"], "queries")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"vidore_juicer_{generation_config['dataset_name']}_queries.json")
    queries_path_before_filtering = os.path.join(
        output_dir,
        f"{prefix}_all_queries.json"
    )
    if os.path.exists(queries_path_before_filtering):
        with open(queries_path_before_filtering, "r", encoding="utf-8") as f:
            query_list = json.load(f)
        queries = [Queries(**query_group) for query_group in query_list]
    else:
        queries = query_generator.generate_queries(summarie_strs)
        with open(queries_path_before_filtering, "w", encoding="utf-8") as file:
            json.dump([query.model_dump() for query in queries], file)
    queries_to_judge = []
    for summary, queries_ in zip(summaries, queries):
        for query in queries_.queries:
            queries_to_judge.append(
                QueryToJudge(
                    query=query.query,
                    summary=summary["summary"],
                    supposed_query_type=query.query_type,
                    supposed_query_format=query.query_format,
                    document_ids=summary["document_ids"],
                    filenames=summary["filenames"],
                    page_numbers=summary["page_numbers"],
                )
            )

    judgments = query_generator.judge_queries(
        queries_to_judge,
    )
    queries_to_keep = []
    discarded_queries = []
    error_nb = 0
    for query_to_judge, judgment in zip(queries_to_judge, judgments):
        query_to_judge.judgment = judgment
        if (
            judgment.relevancy.reasoning == "error"
            or judgment.self_sufficiency.reasoning == "error"
            or judgment.persona_adaptation.reasoning == "error"
            or judgment.query_type_following.reasoning == "error"
            or judgment.query_format_following.reasoning == "error"
        ):
            error_nb += 1
            continue
        if (
            judgment.relevancy.relevancy >= 4
            and judgment.self_sufficiency.self_sufficiency >= 4
            and judgment.persona_adaptation.persona_adaptation >= 4
            and judgment.query_type_following.follows_query_type
            and judgment.query_format_following.follows_query_format
            and (
                query_to_judge.supposed_query_type,
                query_to_judge.supposed_query_format,
            )
            not in [
                ("multi-hop", "keyword"),
                ("enumerative", "keyword"),
                ("boolean", "keyword"),
                ("boolean", "instruction"),
            ]
        ):
            queries_to_keep.append(query_to_judge)
        else:
            discarded_queries.append(query_to_judge)
    print(f"Proportion of errors: {error_nb}/{len(queries_to_judge)}")
    print(f"Proportion of good queries: {len(queries_to_keep) / len(queries_to_judge)}")

    queries_to_keep = [query.model_dump() for query in queries_to_keep]
    for query_to_keep in queries_to_keep:
        query_to_keep["origin_summary"] = query_to_keep.pop("summary")
    with open(output_file, "w") as file:
        json.dump(queries_to_keep, file)
