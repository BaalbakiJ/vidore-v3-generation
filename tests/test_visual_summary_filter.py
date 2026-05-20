import pytest

from vidore_generation.dtos import FinalSummary
from vidore_generation.filters.visual_summary_filter import VisualSummaryFilter
from vidore_generation.generation_schemas import Judgment, Score
from vidore_generation.pipelines.visual_summary_pipeline import get_document_id


def make_judgment(
    information_richness: int,
    persona_relevance: int,
    query_generation_potential: int,
    conceptual_clarity: int,
) -> Judgment:
    return Judgment(
        information_richness=Score(
            grade=information_richness,
            explanation="x",
        ),
        persona_relevance=Score(
            grade=persona_relevance,
            explanation="x",
        ),
        query_generation_potential=Score(
            grade=query_generation_potential,
            explanation="x",
        ),
        conceptual_clarity=Score(
            grade=conceptual_clarity,
            explanation="x",
        ),
    )


def make_uniform_judgment(grade: int) -> Judgment:
    return make_judgment(grade, grade, grade, grade)


def make_summary(
    summary: str,
    filenames: list[str],
    page_numbers: list[list[int]],
    addition_reason: str | None,
    original_summaries: list[str] | None,
) -> FinalSummary:
    return FinalSummary(
        summary=summary,
        document_ids=[get_document_id(filename) for filename in filenames],
        filenames=filenames,
        page_numbers=page_numbers,
        original_summaries=original_summaries,
        addition_reason=addition_reason,
    )


def test_passes_threshold_requires_all_grades_at_minimum() -> None:
    summary_filter = VisualSummaryFilter()

    assert summary_filter.passes_threshold(make_judgment(4, 4, 4, 4), 4)
    assert not summary_filter.passes_threshold(make_judgment(5, 4, 4, 3), 4)


def test_filter_prioritizes_combined_summary_buckets() -> None:
    summary_filter = VisualSummaryFilter()
    single_summary = make_summary(
        "single",
        ["doc_a"],
        [[0]],
        "visual summary from page images",
        None,
    )
    same_document_combined = make_summary(
        "same document combined",
        ["doc_a", "doc_a"],
        [[0], [1]],
        "visual combined summary",
        ["single 0", "single 1"],
    )
    multi_document_combined = make_summary(
        "multi document combined",
        ["doc_a", "doc_b"],
        [[0], [0]],
        "visual combined summary",
        ["single 0", "single 1"],
    )

    selected = summary_filter.filter_summaries(
        candidate_summaries=[
            single_summary,
            same_document_combined,
            multi_document_combined,
        ],
        judgments=[
            make_uniform_judgment(5),
            make_uniform_judgment(5),
            make_uniform_judgment(5),
        ],
        filtered_summaries_nb=3,
        min_grade=4,
    )

    assert [summary.summary for summary in selected] == [
        "multi document combined",
        "same document combined",
        "single",
    ]
    assert [summary.addition_reason for summary in selected] == [
        "visual judged multi-document combined summary",
        "visual judged same-document combined summary",
        "visual judged single summary",
    ]


def test_filter_fills_with_best_rejected_fallbacks() -> None:
    summary_filter = VisualSummaryFilter()
    candidates = [
        make_summary("rejected low", ["doc_a"], [[0]], None, None),
        make_summary("passing", ["doc_a"], [[1]], None, None),
        make_summary("rejected high", ["doc_a"], [[2]], None, None),
    ]

    selected = summary_filter.filter_summaries(
        candidate_summaries=candidates,
        judgments=[
            make_uniform_judgment(2),
            make_uniform_judgment(4),
            make_uniform_judgment(3),
        ],
        filtered_summaries_nb=3,
        min_grade=4,
    )

    assert [summary.summary for summary in selected] == [
        "passing",
        "rejected high",
        "rejected low",
    ]
    assert [summary.addition_reason for summary in selected] == [
        "visual judged single summary",
        "visual fallback selected summary",
        "visual fallback selected summary",
    ]


def test_filter_preserves_page_numbers_and_original_summaries() -> None:
    summary_filter = VisualSummaryFilter()
    original_summary = make_summary(
        "same document combined",
        ["doc_a", "doc_a"],
        [[0, 1], [2, 3]],
        "visual combined summary",
        ["first summary", "second summary"],
    )
    judgment = make_uniform_judgment(5)

    selected = summary_filter.filter_summaries(
        candidate_summaries=[original_summary],
        judgments=[judgment],
        filtered_summaries_nb=1,
        min_grade=4,
    )

    assert selected[0].page_numbers == [[0, 1], [2, 3]]
    assert selected[0].original_summaries == ["first summary", "second summary"]
    assert selected[0].judgments == [judgment]
    assert original_summary.page_numbers == [[0, 1], [2, 3]]
    assert original_summary.original_summaries == ["first summary", "second summary"]
    assert original_summary.judgments is None
    assert original_summary.addition_reason == "visual combined summary"


def test_filter_raises_for_summary_judgment_length_mismatch() -> None:
    summary_filter = VisualSummaryFilter()

    with pytest.raises(ValueError, match="does not match"):
        summary_filter.filter_summaries(
            candidate_summaries=[
                make_summary("single", ["doc_a"], [[0]], None, None),
            ],
            judgments=[],
            filtered_summaries_nb=1,
            min_grade=4,
        )
