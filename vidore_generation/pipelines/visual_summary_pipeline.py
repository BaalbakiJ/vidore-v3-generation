import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from uuid import UUID, uuid5

from vidore_generation.dtos import Failed, FinalSummary, ImageSection, LLMProviderConfig
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Summary
from vidore_generation.generators.visual_summarizer import VisualSummarizer


VISUAL_SUMMARY_DOCUMENT_NAMESPACE = UUID("4a7c56cf-5d73-4b36-a2f4-31ffcf258962")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def extract_page_number(image_path: Path) -> int:
    match = re.match(r"^(.+)_([0-9]+)$", image_path.stem)
    if match is None:
        raise ValueError(
            f"Image filename must end with an underscore and page number: {image_path}"
        )
    return int(match.group(2))


def get_document_id(filename: str) -> UUID:
    return uuid5(VISUAL_SUMMARY_DOCUMENT_NAMESPACE, filename)


def dump_final_summaries(output_path: Path, summaries: List[FinalSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            [json.loads(summary.model_dump_json()) for summary in summaries],
            file,
            indent=4,
        )


def load_final_summaries(input_path: Path) -> List[FinalSummary]:
    with open(input_path, "r", encoding="utf-8") as file:
        return [FinalSummary(**item) for item in json.load(file)]


def make_generation_handler(
    llm_provider: LLMProviderConfig,
    role: str,
    logger: logging.Logger | None = None,
) -> GenerationHandler:
    from vidore_generation.generation_handlers.factory import (
        make_generation_handler as make_factory_generation_handler,
    )

    return make_factory_generation_handler(llm_provider, role, logger=logger)


class VisualSummaryPipeline:
    def __init__(
        self,
        dataset_dir: Path,
        llm_provider: LLMProviderConfig,
        language: str = "english",
        debug: bool = False,
        section_size: int = 1,
        stride: int = 1,
        max_windows: int | None = None,
        max_windows_per_document: int | None = None,
        max_documents: int | None = None,
        max_summary_words: int = 250,
        filtered_summaries_nb: int = 50,
        overwrite_existing: bool = False,
        document_description: str | None = None,
    ):
        if section_size < 1:
            raise ValueError(f"section_size must be at least 1, got {section_size}")
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")
        if max_windows is not None and max_windows < 1:
            raise ValueError(f"max_windows must be at least 1, got {max_windows}")
        if max_windows_per_document is not None and max_windows_per_document < 1:
            raise ValueError(
                "max_windows_per_document must be at least 1, "
                f"got {max_windows_per_document}"
            )
        if max_documents is not None and max_documents < 1:
            raise ValueError(f"max_documents must be at least 1, got {max_documents}")
        if max_summary_words < 1:
            raise ValueError(
                f"max_summary_words must be at least 1, got {max_summary_words}"
            )
        if filtered_summaries_nb < 1:
            raise ValueError(
                f"filtered_summaries_nb must be at least 1, got {filtered_summaries_nb}"
            )

        self.dataset_dir = dataset_dir
        self.imgs_dir = dataset_dir / "imgs"
        self.llm_provider = llm_provider
        self.language = language
        self.debug = debug
        self.section_size = section_size
        self.stride = stride
        self.max_windows = max_windows
        self.max_windows_per_document = max_windows_per_document
        self.max_documents = max_documents
        self.max_summary_words = max_summary_words
        self.filtered_summaries_nb = filtered_summaries_nb
        self.overwrite_existing = overwrite_existing
        self.document_description = document_description or ""
        self.visual_summaries_path = (
            self.dataset_dir / "visual_summaries" / "visual_summaries.json"
        )
        self.summaries_dir = self.dataset_dir / "summaries"
        self.filtered_summaries_path = (
            self.dataset_dir / "filtered_summaries" / "filtered_summaries.json"
        )

        self.init_logger()
        self.vl_generation_handler: GenerationHandler | None = None
        self.visual_summarizer: VisualSummarizer | None = None

    def get_visual_summarizer(self) -> VisualSummarizer:
        if self.visual_summarizer is not None:
            return self.visual_summarizer

        self.vl_generation_handler = make_generation_handler(
            self.llm_provider,
            "vl",
            logger=self.logger,
        )
        self.visual_summarizer = VisualSummarizer(
            model_name=self.llm_provider.vl_model_name
            or self.llm_provider.lm_model_name,
            logger=self.logger,
            generation_handler=self.vl_generation_handler,
            language=self.language,
            max_summary_words=self.max_summary_words,
        )
        return self.visual_summarizer

    def init_logger(self) -> None:
        logs_dir = self.dataset_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(f"{self.__class__.__name__}.{id(self)}")
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.propagate = False

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            logs_dir / f"visual_summary_pipeline_{date_str}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if self.debug else logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def close_logger(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)

    def require_imgs_dir(self) -> None:
        if not self.imgs_dir.exists():
            raise FileNotFoundError(
                f"Rendered page images directory does not exist: {self.imgs_dir}"
            )
        if not self.imgs_dir.is_dir():
            raise NotADirectoryError(
                f"Rendered page images path is not a directory: {self.imgs_dir}"
            )

    def discover_document_dirs(self) -> List[Path]:
        self.require_imgs_dir()
        document_dirs = [
            path for path in sorted(self.imgs_dir.iterdir()) if path.is_dir()
        ]
        if self.max_documents is not None:
            return document_dirs[: self.max_documents]
        return document_dirs

    def get_document_image_paths(self, document_dir: Path) -> List[Path]:
        image_paths = [
            path
            for path in document_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ]
        return sorted(image_paths, key=extract_page_number)

    def create_image_sections(self) -> List[ImageSection]:
        document_dirs = self.discover_document_dirs()
        image_sections: List[ImageSection] = []

        self.logger.info("Documents found: %d", len(document_dirs))
        for document_dir in document_dirs:
            image_paths = self.get_document_image_paths(document_dir)
            document_window_count = 0
            for start_index in range(0, len(image_paths), self.stride):
                end_index = start_index + self.section_size
                window_paths = image_paths[start_index:end_index]
                if not window_paths:
                    continue
                image_sections.append(
                    ImageSection(
                        filename=document_dir.name,
                        document_description=self.document_description,
                        images=[],
                        image_paths=[str(path) for path in window_paths],
                        page_numbers=[
                            extract_page_number(path) for path in window_paths
                        ],
                    )
                )
                document_window_count += 1
                if (
                    self.max_windows is not None
                    and len(image_sections) >= self.max_windows
                ):
                    self.logger.info(
                        "Image windows created: %d",
                        len(image_sections),
                    )
                    return image_sections
                if (
                    self.max_windows_per_document is not None
                    and document_window_count >= self.max_windows_per_document
                ):
                    break

        self.logger.info("Image windows created: %d", len(image_sections))
        return image_sections

    def create_final_summaries(
        self,
        image_sections: List[ImageSection],
        summary_results: List[Summary | Failed],
    ) -> List[FinalSummary]:
        if len(image_sections) != len(summary_results):
            raise ValueError(
                "Visual summary result count does not match image window count: "
                f"{len(summary_results)} results for {len(image_sections)} windows"
            )

        final_summaries: List[FinalSummary] = []
        for image_section, summary_result in zip(image_sections, summary_results):
            if isinstance(summary_result, Failed):
                continue
            final_summaries.append(
                FinalSummary(
                    summary=summary_result.summary,
                    document_ids=[get_document_id(image_section.filename)],
                    filenames=[image_section.filename],
                    page_numbers=[image_section.page_numbers],
                    addition_reason="visual summary from page images",
                )
            )
        return final_summaries

    def export_document_summaries(self, summaries: List[FinalSummary]) -> None:
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        summaries_by_filename: Dict[str, List[FinalSummary]] = {}
        for summary in summaries:
            filename = summary.filenames[0]
            if filename not in summaries_by_filename:
                summaries_by_filename[filename] = []
            summaries_by_filename[filename].append(summary)

        for filename, document_summaries in summaries_by_filename.items():
            output_path = self.summaries_dir / f"{filename}.json"
            dump_final_summaries(output_path, document_summaries)
            self.logger.info("Wrote summaries: %s", output_path)

    def export_outputs(self, summaries: List[FinalSummary]) -> List[FinalSummary]:
        filtered_summaries = summaries[: self.filtered_summaries_nb]

        dump_final_summaries(self.visual_summaries_path, summaries)
        self.logger.info("Wrote visual summaries: %s", self.visual_summaries_path)
        self.export_document_summaries(summaries)
        dump_final_summaries(self.filtered_summaries_path, filtered_summaries)
        self.logger.info("Wrote filtered summaries: %s", self.filtered_summaries_path)

        return filtered_summaries

    def run(self) -> List[FinalSummary]:
        try:
            return self._run()
        finally:
            self.close_logger()

    def _run(self) -> List[FinalSummary]:
        self.require_imgs_dir()
        if self.visual_summaries_path.exists() and not self.overwrite_existing:
            document_dirs = self.discover_document_dirs()
            self.logger.info("Documents found: %d", len(document_dirs))
            summaries = load_final_summaries(self.visual_summaries_path)
            self.logger.info(
                "Image windows created: %d (reused existing visual summaries)",
                len(summaries),
            )
            self.logger.info(
                "Loaded existing visual summaries: %d",
                len(summaries),
            )
            filtered_summaries = self.export_outputs(summaries)
            self.logger.info(
                "Successful summaries: %d",
                len(summaries),
            )
            return filtered_summaries

        image_sections = self.create_image_sections()
        if not image_sections:
            self.logger.info("Successful summaries: 0")
            return self.export_outputs([])

        summary_results = self.get_visual_summarizer().summarize_sections(
            image_sections
        )
        summaries = self.create_final_summaries(image_sections, summary_results)
        self.logger.info("Successful summaries: %d", len(summaries))
        return self.export_outputs(summaries)
