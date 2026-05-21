import logging

from vidore_generation.query_generation.vidore_juicer.generate_custom_queries import (
    QueryGenerator,
    QueryToJudge,
    _finish_usage_batch_if_supported,
    _start_usage_batch_if_supported,
)


class HandlerWithoutBatchUsage:
    cost: float = 0.0


class FakeBatchUsageHandler:
    def __init__(self, max_concurrency: int) -> None:
        self.max_concurrency = max_concurrency
        self.cost = 0.0
        self.started_count = 0
        self.finished: list[tuple[str, int | None]] = []

    def start_usage_batch(self) -> None:
        self.started_count += 1

    def finish_usage_batch(
        self,
        batch_description: str,
        effective_max_concurrency: int | None = None,
    ) -> object:
        self.finished.append((batch_description, effective_max_concurrency))
        return object()


def _make_query_generator(
    query_handler: object,
    judge_handler: object,
) -> QueryGenerator:
    generator = QueryGenerator.__new__(QueryGenerator)
    generator.retry_count = 1
    generator.query_handler = query_handler
    generator.judge_handler = judge_handler
    generator.logger = logging.getLogger("test-vidore-juicer-bedrock-usage")
    return generator


def test_batch_usage_helpers_noop_without_batch_methods() -> None:
    handler = HandlerWithoutBatchUsage()

    _start_usage_batch_if_supported(handler)
    _finish_usage_batch_if_supported(handler, "Generating queries", 50)


def test_batch_usage_helpers_call_supported_handler_methods() -> None:
    handler = FakeBatchUsageHandler(max_concurrency=2)

    _start_usage_batch_if_supported(handler)
    _finish_usage_batch_if_supported(handler, "Generating queries", 50)

    assert handler.started_count == 1
    assert handler.finished == [("Generating queries", 2)]


def test_judge_batch_usage_helper_caps_concurrency() -> None:
    handler = FakeBatchUsageHandler(max_concurrency=1)

    _finish_usage_batch_if_supported(handler, "Generating judgments", 2)

    assert handler.finished == [("Generating judgments", 1)]


def test_generate_queries_finishes_supported_batch_usage() -> None:
    query_handler = FakeBatchUsageHandler(max_concurrency=2)
    generator = _make_query_generator(
        query_handler=query_handler,
        judge_handler=HandlerWithoutBatchUsage(),
    )
    captured: dict[str, object] = {}

    async def fake_async_generate_queries(
        summaries: list[object],
        semaphore_size: int,
        desc: str,
    ) -> list[str]:
        captured["summaries"] = summaries
        captured["semaphore_size"] = semaphore_size
        captured["desc"] = desc
        return ["query"]

    generator.async_generate_queries = fake_async_generate_queries

    result = generator.generate_queries(["summary"])

    assert result == ["query"]
    assert captured["semaphore_size"] == 50
    assert captured["desc"] == "Generating queries"
    assert query_handler.started_count == 1
    assert query_handler.finished == [("Generating queries", 2)]


def test_judge_queries_finishes_supported_batch_usage() -> None:
    judge_handler = FakeBatchUsageHandler(max_concurrency=1)
    generator = _make_query_generator(
        query_handler=HandlerWithoutBatchUsage(),
        judge_handler=judge_handler,
    )
    query_to_judge = QueryToJudge(
        query="What is shown?",
        summary="A document summary.",
        supposed_query_type="textual",
        supposed_query_format="question",
        document_ids=[],
        filenames=[],
        page_numbers=[],
    )
    captured: dict[str, object] = {}

    async def fake_async_generate_judgments(
        queries_to_judge: list[QueryToJudge],
        semaphore_size: int,
        desc: str,
    ) -> list[str]:
        captured["queries_to_judge"] = queries_to_judge
        captured["semaphore_size"] = semaphore_size
        captured["desc"] = desc
        return ["judgment"]

    generator.async_generate_judgments = fake_async_generate_judgments

    result = generator.judge_queries([query_to_judge])

    assert result == ["judgment"]
    assert captured["semaphore_size"] == 2
    assert captured["desc"] == "Generating judgments"
    assert judge_handler.started_count == 1
    assert judge_handler.finished == [("Generating judgments", 1)]
