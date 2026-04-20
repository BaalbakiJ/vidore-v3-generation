import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from vidore_generation.dtos import Document, FinalSummary
from vidore_generation.filters.summary_filter import SummaryFilter
from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
)
from vidore_generation.generators.judge import Judge
from vidore_generation.generators.summary_combinator import SummaryCombinator


class SummaryPipeline:
    def __init__(
        self,
        documents_dir: Path,
        model_name: str,
        persona: str,
        debug: bool = False,
        combination_iteration_nb: int = 20,
        inference_method: Literal["api", "vllm"] = "api",
        sampling_multi_doc_ratio: float = 0.5,
        language: str = "english",
        filtered_summaries_nb: int = 400,
        extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.documents_dir = documents_dir
        self.markdowns_dir = Path(os.path.join(documents_dir, "markdowns"))
        self.model_name = model_name
        self.persona = persona
        self.debug = debug
        self.combination_iteration_nb = combination_iteration_nb
        self.language = language
        self.filtered_summaries_nb = filtered_summaries_nb
        self.init_logger()

        if inference_method == "api":
            self.generation_handler = APIGenerationHandler(
                model_name=model_name, logger=self.logger, extra_kwargs=extra_kwargs or {}
            )
        # elif inference_method == "vllm":
        #     self.generation_handler = VLLMGenerationHandler(model_name=model_name, logger=self.logger)
        else:
            raise ValueError(f"Invalid inference method: {inference_method}")

        self.summary_combinator = SummaryCombinator(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            combination_iteration_nb=self.combination_iteration_nb,
            save_folder=self.documents_dir,
            debug=self.debug,
            sampling_multi_doc_ratio=sampling_multi_doc_ratio,
            language=language,
        )
        self.judge = Judge(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            language=language,
        )
        self.summary_filter = SummaryFilter()

    def init_logger(self):
        logs_dir = os.path.join(self.documents_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(
            logging.DEBUG
        )  # Capture all messages at DEBUG level and above

        # ---- File Handler: logs everything ----
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f"llm_pipeline_{date_str}.log")
        )
        file_handler.setLevel(logging.DEBUG)  # Log everything

        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # ---- Console Handler: logs only INFO and above ----
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only log INFO and above

        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def combine_summaries(
        self, documents: List[Document], summaries: List[FinalSummary]
    ):
        combined_summaries_dir = os.path.join(self.documents_dir, "combined_summaries")
        if os.path.exists(combined_summaries_dir):
            combined_summaries = self.summary_combinator.import_combined_summaries(
                combined_summaries_dir
            )
        else:
            combined_summaries = self.summary_combinator.combine_summaries(
                documents,
                summaries,
                random_seeds=list(range(self.combination_iteration_nb)),
            )

            self.logger.debug(f"Combined summary: {combined_summaries[0]}")
            combined_summaries = [
                FinalSummary(
                    summary=combined_summary.combined_summary,
                    document_ids=[
                        summary.document_id for summary in combined_summary.summaries
                    ],
                    filenames=[
                        summary.filename for summary in combined_summary.summaries
                    ],
                    page_numbers=[
                        summary.page_numbers for summary in combined_summary.summaries
                    ],
                    original_summaries=[
                        summary.summary for summary in combined_summary.summaries
                    ],
                )
                for combined_summary in combined_summaries
            ]
            self.summary_combinator.export(combined_summaries_dir, combined_summaries)
        return combined_summaries

    def judge_summaries(self, summaries):
        judgments_dir = os.path.join(self.documents_dir, "judgments")
        if os.path.exists(judgments_dir):
            judgments = self.judge.import_judgments(judgments_dir)
            assert len(judgments) == len(summaries)
        else:
            judgments = self.judge.judge_summaries(summaries, self.persona)
            self.logger.debug(f"Judgment: {judgments[0]}")
            self.judge.export(judgments_dir, judgments, "judgments")
        return judgments

    def filter_summaries(
        self, single_session_summaries, combined_summaries, sections, judgments
    ):
        filtered_summaries = self.summary_filter.filter_summaries(
            single_session_summaries,
            combined_summaries,
            sections,
            judgments,
            self.filtered_summaries_nb,
        )
        self.logger.debug(f"Filtered summary: {filtered_summaries[0]}")
        self.summary_filter.export(
            os.path.join(self.documents_dir, "filtered_summaries"), filtered_summaries
        )
        return filtered_summaries

    def run(self):
        raise NotImplementedError("Subclasses must implement this method")
