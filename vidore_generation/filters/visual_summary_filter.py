from statistics import mean
from typing import Literal

from vidore_generation.dtos import FinalSummary
from vidore_generation.generation_schemas import Judgment


SummaryBucket = Literal[
    "multi_document_combined",
    "same_document_combined",
    "single",
]


class VisualSummaryFilter:
    def passes_threshold(
        self,
        judgment: Judgment,
        min_grade: int = 4,
    ) -> bool:
        return all(
            self._score_grade(score) >= min_grade
            for score in judgment.model_dump().values()
        )

    def count_passing(
        self,
        judgments: list[Judgment],
        min_grade: int = 4,
    ) -> int:
        return sum(
            1 for judgment in judgments if self.passes_threshold(judgment, min_grade)
        )

    def filter_summaries(
        self,
        candidate_summaries: list[FinalSummary],
        judgments: list[Judgment],
        filtered_summaries_nb: int,
        min_grade: int = 4,
    ) -> list[FinalSummary]:
        if len(candidate_summaries) != len(judgments):
            raise ValueError(
                "Visual summary count does not match judgment count: "
                f"{len(candidate_summaries)} summaries for {len(judgments)} judgments"
            )
        if filtered_summaries_nb <= 0:
            return []

        passing_buckets: dict[
            SummaryBucket,
            list[tuple[FinalSummary, Judgment]],
        ] = {
            "multi_document_combined": [],
            "same_document_combined": [],
            "single": [],
        }
        rejected: list[tuple[int, FinalSummary, Judgment]] = []

        for index, (summary, judgment) in enumerate(
            zip(candidate_summaries, judgments)
        ):
            if self.passes_threshold(judgment, min_grade):
                bucket = self._bucket_summary(summary)
                passing_buckets[bucket].append((summary, judgment))
                continue
            rejected.append((index, summary, judgment))

        selected_summaries: list[FinalSummary] = []
        for bucket in (
            "multi_document_combined",
            "same_document_combined",
            "single",
        ):
            for summary, judgment in passing_buckets[bucket]:
                if len(selected_summaries) >= filtered_summaries_nb:
                    return selected_summaries
                selected_summaries.append(
                    self._select_summary(
                        summary,
                        judgment,
                        self._addition_reason(bucket),
                    )
                )

        ranked_rejections = sorted(
            rejected,
            key=lambda item: (-self._average_grade(item[2]), item[0]),
        )
        for _index, summary, judgment in ranked_rejections:
            if len(selected_summaries) >= filtered_summaries_nb:
                break
            selected_summaries.append(
                self._select_summary(
                    summary,
                    judgment,
                    "visual fallback selected summary",
                )
            )

        return selected_summaries

    def _bucket_summary(self, summary: FinalSummary) -> SummaryBucket:
        if not self._is_combined_summary(summary):
            return "single"
        if len(set(summary.filenames)) > 1:
            return "multi_document_combined"
        return "same_document_combined"

    def _is_combined_summary(self, summary: FinalSummary) -> bool:
        return (
            summary.addition_reason == "visual combined summary"
            or summary.original_summaries is not None
            or len(summary.filenames) > 1
            or len(summary.page_numbers) > 1
        )

    def _addition_reason(self, bucket: SummaryBucket) -> str:
        if bucket == "multi_document_combined":
            return "visual judged multi-document combined summary"
        if bucket == "same_document_combined":
            return "visual judged same-document combined summary"
        return "visual judged single summary"

    def _select_summary(
        self,
        summary: FinalSummary,
        judgment: Judgment,
        addition_reason: str,
    ) -> FinalSummary:
        selected_summary = summary.model_copy(deep=True)
        selected_summary.addition_reason = addition_reason
        selected_summary.judgments = [judgment]
        return selected_summary

    def _average_grade(self, judgment: Judgment) -> float:
        grades = [
            self._score_grade(score) for score in judgment.model_dump().values()
        ]
        if not grades:
            raise ValueError("Judgment must contain at least one score")
        return mean(grades)

    def _score_grade(self, score: object) -> int:
        if isinstance(score, dict):
            grade = score.get("grade")
        else:
            grade = getattr(score, "grade", None)
        if not isinstance(grade, int) or isinstance(grade, bool):
            raise TypeError(f"Judgment score grade must be an integer, got {grade}")
        return grade
