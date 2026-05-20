import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, cast
from uuid import UUID, uuid5

from vidore_generation.dtos import (
    CombinedSummary,
    Document,
    DocumentDescription,
    Failed,
    FinalSummary,
    ImageSection,
    IndexedSummary,
    LLMProviderConfig,
)
from vidore_generation.filters.visual_summary_filter import VisualSummaryFilter
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Judgment, Score, Summary
from vidore_generation.generators.judge import Judge
from vidore_generation.generators.visual_document_descriptor import (
    VisualDocumentDescriptor,
    VisualDocumentSample,
)
from vidore_generation.generators.visual_summarizer import VisualSummarizer
from vidore_generation.page_filtering.page_manifest import (
    get_excluded_image_page_numbers_by_filename,
    load_page_manifest,
)


if TYPE_CHECKING:
    from vidore_generation.generators.summary_combinator import SummaryCombinator


VISUAL_SUMMARY_DOCUMENT_NAMESPACE = UUID("4a7c56cf-5d73-4b36-a2f4-31ffcf258962")
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}

# Page-numbering convention for the visual pipeline:
# - page_manifest.page_index is the 0-based PDF library index.
# - page_manifest.image_page_number is the 0-based rendered image suffix/index.
# - page_manifest.page_number is the 1-based human/PDF display page number.
# - FinalSummary.page_numbers stores 0-based image_page_number values so it
#   matches rendered image filenames like <filename>_<image_page_number>.png
#   and can be used directly for retrieval/QREL page IDs.


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


def failed_judgment(error: str) -> Judgment:
    explanation = f"Visual summary judgment failed: {error}"
    return Judgment(
        information_richness=Score(grade=1, explanation=explanation),
        persona_relevance=Score(grade=1, explanation=explanation),
        query_generation_potential=Score(grade=1, explanation=explanation),
        conceptual_clarity=Score(grade=1, explanation=explanation),
    )


class VisualSummaryPipeline:
    def __init__(
        self,
        dataset_dir: Path,
        llm_provider: LLMProviderConfig,
        language: str = "english",
        persona: str = "",
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
        respect_page_manifest: bool = False,
        use_visual_document_descriptions: bool = False,
        max_description_pages: int = 3,
        max_description_words: int = 150,
        overwrite_descriptions: bool = False,
        use_visual_combined_summaries: bool = False,
        overwrite_combined_summaries: bool = False,
        combination_iteration_nb: int = 20,
        sampling_multi_doc_ratio: float = 0.5,
        use_visual_summary_judging: bool = False,
        overwrite_judgments: bool = False,
        visual_judgment_min_grade: int = 4,
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
        if max_description_pages < 1:
            raise ValueError(
                f"max_description_pages must be at least 1, got {max_description_pages}"
            )
        if max_description_words < 1:
            raise ValueError(
                f"max_description_words must be at least 1, got {max_description_words}"
            )
        if combination_iteration_nb < 0:
            raise ValueError(
                "combination_iteration_nb must be at least 0, "
                f"got {combination_iteration_nb}"
            )
        if sampling_multi_doc_ratio < 0.0 or sampling_multi_doc_ratio > 1.0:
            raise ValueError(
                "sampling_multi_doc_ratio must be between 0.0 and 1.0, "
                f"got {sampling_multi_doc_ratio}"
            )
        if visual_judgment_min_grade < 1 or visual_judgment_min_grade > 5:
            raise ValueError(
                "visual_judgment_min_grade must be between 1 and 5, "
                f"got {visual_judgment_min_grade}"
            )

        self.dataset_dir = dataset_dir
        self.imgs_dir = dataset_dir / "imgs"
        self.llm_provider = llm_provider
        self.language = language
        self.persona = persona
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
        self.respect_page_manifest = respect_page_manifest
        self.use_visual_document_descriptions = use_visual_document_descriptions
        self.max_description_pages = max_description_pages
        self.max_description_words = max_description_words
        self.overwrite_descriptions = overwrite_descriptions
        self.use_visual_combined_summaries = use_visual_combined_summaries
        self.overwrite_combined_summaries = overwrite_combined_summaries
        self.combination_iteration_nb = combination_iteration_nb
        self.sampling_multi_doc_ratio = sampling_multi_doc_ratio
        self.use_visual_summary_judging = use_visual_summary_judging
        self.overwrite_judgments = overwrite_judgments
        self.visual_judgment_min_grade = visual_judgment_min_grade
        self.excluded_visual_summary_page_numbers: dict[str, set[int]] | None = None
        self.visual_summaries_path = (
            self.dataset_dir / "visual_summaries" / "visual_summaries.json"
        )
        self.summaries_dir = self.dataset_dir / "summaries"
        self.combined_summaries_dir = self.dataset_dir / "combined_summaries"
        self.judgments_dir = self.dataset_dir / "judgments"
        self.filtered_summaries_path = (
            self.dataset_dir / "filtered_summaries" / "filtered_summaries.json"
        )

        self.init_logger()
        self.vl_generation_handler: GenerationHandler | None = None
        self.lm_generation_handler: GenerationHandler | None = None
        self.judge_generation_handler: GenerationHandler | None = None
        self.visual_summarizer: VisualSummarizer | None = None
        self.visual_document_descriptor: VisualDocumentDescriptor | None = None
        self.summary_combinator: "SummaryCombinator | None" = None
        self.judge: Judge | None = None
        self.visual_summary_filter = VisualSummaryFilter()

    @property
    def descriptions_dir(self) -> Path:
        return self.dataset_dir / "descriptions"

    def get_vl_generation_handler(self) -> GenerationHandler:
        if self.vl_generation_handler is None:
            self.vl_generation_handler = make_generation_handler(
                self.llm_provider,
                "vl",
                logger=self.logger,
            )
        return self.vl_generation_handler

    def get_lm_generation_handler(self) -> GenerationHandler:
        if self.lm_generation_handler is None:
            self.lm_generation_handler = make_generation_handler(
                self.llm_provider,
                "lm",
                logger=self.logger,
            )
        return self.lm_generation_handler

    def get_judge_generation_handler(self) -> GenerationHandler:
        if self.judge_generation_handler is None:
            self.judge_generation_handler = make_generation_handler(
                self.llm_provider,
                "judge",
                logger=self.logger,
            )
        return self.judge_generation_handler

    def get_visual_summarizer(self) -> VisualSummarizer:
        if self.visual_summarizer is not None:
            return self.visual_summarizer

        self.visual_summarizer = VisualSummarizer(
            model_name=self.llm_provider.vl_model_name
            or self.llm_provider.lm_model_name,
            logger=self.logger,
            generation_handler=self.get_vl_generation_handler(),
            language=self.language,
            max_summary_words=self.max_summary_words,
        )
        return self.visual_summarizer

    def get_visual_document_descriptor(self) -> VisualDocumentDescriptor:
        if self.visual_document_descriptor is not None:
            return self.visual_document_descriptor

        self.visual_document_descriptor = VisualDocumentDescriptor(
            model_name=self.llm_provider.vl_model_name
            or self.llm_provider.lm_model_name,
            logger=self.logger,
            generation_handler=self.get_vl_generation_handler(),
            language=self.language,
            max_description_words=self.max_description_words,
        )
        return self.visual_document_descriptor

    def get_judge(self) -> Judge:
        if self.judge is not None:
            return self.judge

        self.judge = Judge(
            model_name=self.llm_provider.judge_model_name
            or self.llm_provider.lm_model_name,
            logger=self.logger,
            generation_handler=self.get_judge_generation_handler(),
            language=self.language,
        )
        return self.judge

    def get_summary_combinator(self) -> "SummaryCombinator":
        if self.summary_combinator is not None:
            return self.summary_combinator

        from vidore_generation.generators.summary_combinator import SummaryCombinator

        self.summary_combinator = SummaryCombinator(
            model_name=self.llm_provider.lm_model_name,
            logger=self.logger,
            generation_handler=self.get_lm_generation_handler(),
            combination_iteration_nb=self.combination_iteration_nb,
            sampling_multi_doc_ratio=self.sampling_multi_doc_ratio,
            save_folder=str(self.dataset_dir),
            debug=self.debug,
            language=self.language,
        )
        return self.summary_combinator

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

    def get_excluded_visual_summary_page_numbers(self) -> dict[str, set[int]]:
        if not self.respect_page_manifest:
            return {}
        if self.excluded_visual_summary_page_numbers is not None:
            return self.excluded_visual_summary_page_numbers

        manifest_path = self.dataset_dir / "page_manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "Page manifest not found. Run "
                "'vidore-generation build-page-manifest --config ...' first: "
                f"{manifest_path}"
            )
        manifest_rows = load_page_manifest(manifest_path)
        self.excluded_visual_summary_page_numbers = (
            get_excluded_image_page_numbers_by_filename(
                manifest_rows,
                "exclude_from_visual_summaries",
            )
        )
        return self.excluded_visual_summary_page_numbers

    def get_document_image_paths(self, document_dir: Path) -> List[Path]:
        excluded_page_numbers = self.get_excluded_visual_summary_page_numbers().get(
            document_dir.name,
            set(),
        )
        image_paths = [
            path
            for path in document_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            and extract_page_number(path) not in excluded_page_numbers
        ]
        return sorted(image_paths, key=extract_page_number)

    def get_description_path(self, filename: str) -> Path:
        return self.descriptions_dir / f"{filename}.json"

    def get_combined_summaries_path(self) -> Path:
        return self.combined_summaries_dir / "combined_summaries.json"

    def get_judgments_path(self) -> Path:
        return self.judgments_dir / "judgments.json"

    def load_document_description(self, filename: str) -> DocumentDescription | None:
        description_path = self.get_description_path(filename)
        if not description_path.exists():
            return None

        with open(description_path, "r", encoding="utf-8") as file:
            return DocumentDescription(**json.load(file))

    def export_document_description(
        self,
        filename: str,
        description: DocumentDescription,
    ) -> None:
        self.descriptions_dir.mkdir(parents=True, exist_ok=True)
        with open(self.get_description_path(filename), "w", encoding="utf-8") as file:
            json.dump(json.loads(description.model_dump_json()), file, indent=4)

    def create_visual_document_samples(
        self,
        document_dirs: List[Path],
    ) -> List[VisualDocumentSample]:
        samples: List[VisualDocumentSample] = []
        for document_dir in document_dirs:
            image_paths = self.get_document_image_paths(document_dir)[
                : self.max_description_pages
            ]
            if not image_paths:
                continue
            samples.append(
                VisualDocumentSample(
                    filename=document_dir.name,
                    image_paths=[str(path) for path in image_paths],
                    page_numbers=[extract_page_number(path) for path in image_paths],
                )
            )
        return samples

    def get_document_description_for_filename(
        self,
        filename: str,
        descriptions_by_filename: dict[str, str],
    ) -> str:
        description = descriptions_by_filename.get(filename)
        if description:
            return description
        if self.document_description:
            return self.document_description
        return "Document from the visual summary dataset."

    def describe_documents(self, document_dirs: List[Path]) -> dict[str, str]:
        descriptions_by_filename: dict[str, str] = {}
        document_dirs_to_generate: List[Path] = []
        loaded_count = 0

        for document_dir in document_dirs:
            if not self.overwrite_descriptions:
                existing_description = self.load_document_description(
                    document_dir.name
                )
                if existing_description is not None:
                    descriptions_by_filename[document_dir.name] = (
                        existing_description.description
                    )
                    loaded_count += 1
                    continue
            document_dirs_to_generate.append(document_dir)

        self.logger.info("Document descriptions loaded: %d", loaded_count)

        samples = self.create_visual_document_samples(document_dirs_to_generate)
        sampled_filenames = {sample.filename for sample in samples}
        generated_count = 0
        written_count = 0

        for document_dir in document_dirs_to_generate:
            if document_dir.name in sampled_filenames:
                continue
            description_text = self.get_document_description_for_filename(
                document_dir.name,
                {},
            )
            descriptions_by_filename[document_dir.name] = description_text
            self.export_document_description(
                document_dir.name,
                DocumentDescription(
                    document_id=get_document_id(document_dir.name),
                    description=description_text,
                ),
            )
            written_count += 1

        if samples:
            description_results = (
                self.get_visual_document_descriptor().describe_documents(samples)
            )
            if len(description_results) != len(samples):
                raise ValueError(
                    "Visual document description result count does not match "
                    "document sample count: "
                    f"{len(description_results)} results for {len(samples)} samples"
                )

            for sample, description_result in zip(samples, description_results):
                if isinstance(description_result, Failed):
                    description_text = self.get_document_description_for_filename(
                        sample.filename,
                        {},
                    )
                    self.logger.warning(
                        "Visual document description failed for %s",
                        sample.filename,
                    )
                else:
                    description_text = description_result.description
                    generated_count += 1

                descriptions_by_filename[sample.filename] = description_text
                self.export_document_description(
                    sample.filename,
                    DocumentDescription(
                        document_id=get_document_id(sample.filename),
                        description=description_text,
                    ),
                )
                written_count += 1

        self.logger.info("Document descriptions generated: %d", generated_count)
        if written_count:
            self.logger.info("Document descriptions written: %s", self.descriptions_dir)

        return descriptions_by_filename

    def create_image_sections(self) -> List[ImageSection]:
        document_dirs = self.discover_document_dirs()
        descriptions_by_filename: dict[str, str] = {}
        if self.use_visual_document_descriptions:
            descriptions_by_filename = self.describe_documents(document_dirs)

        image_sections: List[ImageSection] = []

        self.logger.info("Documents found: %d", len(document_dirs))
        for document_dir in document_dirs:
            image_paths = self.get_document_image_paths(document_dir)
            document_description = self.document_description
            if self.use_visual_document_descriptions:
                document_description = self.get_document_description_for_filename(
                    document_dir.name,
                    descriptions_by_filename,
                )
            document_window_count = 0
            for start_index in range(0, len(image_paths), self.stride):
                end_index = start_index + self.section_size
                window_paths = image_paths[start_index:end_index]
                if not window_paths:
                    continue
                image_sections.append(
                    ImageSection(
                        filename=document_dir.name,
                        document_description=document_description,
                        images=[],
                        image_paths=[str(path) for path in window_paths],
                        # Store 0-based rendered image page numbers, matching
                        # image filenames and page_manifest.image_page_number.
                        # Human-facing 1-based page numbers can be recovered
                        # from page_manifest.page_number or by adding 1.
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

    def create_combination_documents(
        self,
        summaries: List[FinalSummary],
    ) -> List[Document]:
        filenames = sorted(
            {
                filename
                for summary in summaries
                for filename in summary.filenames
            }
        )
        return [
            Document(
                id=get_document_id(filename),
                filename=filename,
                content="",
                document_description=self.load_document_description(filename),
            )
            for filename in filenames
        ]

    def create_final_combined_summaries(
        self,
        combined_summaries: List[CombinedSummary],
    ) -> List[FinalSummary]:
        final_summaries: List[FinalSummary] = []
        for combined_summary in combined_summaries:
            indexed_summaries: List[IndexedSummary] = combined_summary.summaries
            final_summaries.append(
                FinalSummary(
                    summary=combined_summary.combined_summary,
                    document_ids=[
                        indexed_summary.document_id
                        for indexed_summary in indexed_summaries
                    ],
                    filenames=[
                        indexed_summary.filename
                        for indexed_summary in indexed_summaries
                    ],
                    page_numbers=[
                        indexed_summary.page_numbers
                        for indexed_summary in indexed_summaries
                    ],
                    original_summaries=[
                        indexed_summary.summary
                        for indexed_summary in indexed_summaries
                    ],
                    addition_reason="visual combined summary",
                )
            )
        return final_summaries

    def load_visual_combined_summaries(self) -> List[FinalSummary]:
        return load_final_summaries(self.get_combined_summaries_path())

    def load_visual_judgments(self) -> List[Judgment]:
        judgments_path = self.get_judgments_path()
        with open(judgments_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(
                f"Visual summary judgments must be a list: {judgments_path}"
            )
        return [Judgment(**item) for item in data]

    def export_visual_judgments(self, judgments: List[Judgment]) -> None:
        judgments_path = self.get_judgments_path()
        judgments_path.parent.mkdir(parents=True, exist_ok=True)
        with open(judgments_path, "w", encoding="utf-8") as file:
            json.dump(
                [json.loads(judgment.model_dump_json()) for judgment in judgments],
                file,
                indent=4,
            )
        self.logger.info("Wrote visual summary judgments: %s", judgments_path)

    def export_visual_combined_summaries(
        self,
        combined_summaries: List[FinalSummary],
    ) -> None:
        dump_final_summaries(self.get_combined_summaries_path(), combined_summaries)
        self.logger.info(
            "Wrote combined visual summaries: %s",
            self.get_combined_summaries_path(),
        )

    def combine_visual_summaries(
        self,
        summaries: List[FinalSummary],
    ) -> List[FinalSummary]:
        if not self.use_visual_combined_summaries:
            return []
        if len(summaries) < 2:
            self.logger.warning(
                "Skipping visual summary combination: at least 2 summaries are "
                "required, got %d",
                len(summaries),
            )
            return []

        combined_summaries_path = self.get_combined_summaries_path()
        if combined_summaries_path.exists() and not self.overwrite_combined_summaries:
            combined_summaries = self.load_visual_combined_summaries()
            self.logger.info(
                "Loaded existing combined visual summaries: %d",
                len(combined_summaries),
            )
            return combined_summaries

        if self.combination_iteration_nb == 0:
            self.logger.info(
                "Skipping visual summary combination: combination_iteration_nb is 0"
            )
            return []

        documents = self.create_combination_documents(summaries)
        try:
            generated_combined_summaries = (
                self.get_summary_combinator().combine_summaries(
                    documents,
                    summaries,
                    random_seeds=list(range(self.combination_iteration_nb)),
                )
            )
        except (ValueError, AssertionError, IndexError) as error:
            self.logger.warning(
                "Skipping visual summary combination because there was not enough "
                "compatible summary data: %s",
                error,
            )
            return []

        combined_summaries = self.create_final_combined_summaries(
            generated_combined_summaries
        )
        self.export_visual_combined_summaries(combined_summaries)
        return combined_summaries

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

    def export_visual_summaries(self, summaries: List[FinalSummary]) -> None:
        dump_final_summaries(self.visual_summaries_path, summaries)
        self.logger.info("Wrote visual summaries: %s", self.visual_summaries_path)
        self.export_document_summaries(summaries)

    def export_filtered_summaries(
        self,
        candidate_summaries: List[FinalSummary],
    ) -> List[FinalSummary]:
        filtered_summaries = candidate_summaries[: self.filtered_summaries_nb]
        dump_final_summaries(self.filtered_summaries_path, filtered_summaries)
        self.logger.info("Wrote filtered summaries: %s", self.filtered_summaries_path)
        return filtered_summaries

    def normalize_judgment_result(
        self,
        judgment_result: Judgment | Failed,
    ) -> Judgment:
        if isinstance(judgment_result, Judgment):
            return judgment_result
        if isinstance(judgment_result, Failed):
            error = judgment_result.error or "unknown judgment generation error"
            self.logger.warning(
                "Visual summary judgment failed; using low-score judgment: %s",
                error,
            )
            return failed_judgment(error)
        raise TypeError(
            "Unexpected visual summary judgment result type: "
            f"{type(judgment_result).__name__}"
        )

    def judge_visual_summaries(
        self,
        candidate_summaries: List[FinalSummary],
    ) -> List[Judgment]:
        if not candidate_summaries:
            return []

        judgments_path = self.get_judgments_path()
        if judgments_path.exists() and not self.overwrite_judgments:
            judgments = self.load_visual_judgments()
            if len(judgments) == len(candidate_summaries):
                self.logger.info(
                    "Loaded existing visual summary judgments: %d",
                    len(judgments),
                )
                return judgments
            self.logger.warning(
                "Existing visual summary judgments count does not match candidate "
                "summary count; regenerating: %d judgments for %d summaries",
                len(judgments),
                len(candidate_summaries),
            )

        raw_judgments = self.get_judge().judge_summaries(
            cast(List[Summary], candidate_summaries),
            self.persona,
        )
        if not isinstance(raw_judgments, list):
            raise TypeError(
                "Judge returned unexpected visual summary judgments container: "
                f"{type(raw_judgments).__name__}"
            )
        if len(raw_judgments) != len(candidate_summaries):
            raise ValueError(
                "Visual summary judgment result count does not match candidate "
                "summary count: "
                f"{len(raw_judgments)} judgments for {len(candidate_summaries)} "
                "summaries"
            )

        judgments = [
            self.normalize_judgment_result(judgment_result)
            for judgment_result in raw_judgments
        ]
        self.export_visual_judgments(judgments)
        return judgments

    def filter_visual_summaries_with_judgments(
        self,
        candidate_summaries: List[FinalSummary],
    ) -> List[FinalSummary]:
        if not self.use_visual_summary_judging:
            return self.export_filtered_summaries(candidate_summaries)

        judgments = self.judge_visual_summaries(candidate_summaries)
        filtered_summaries = self.visual_summary_filter.filter_summaries(
            candidate_summaries=candidate_summaries,
            judgments=judgments,
            filtered_summaries_nb=self.filtered_summaries_nb,
            min_grade=self.visual_judgment_min_grade,
        )
        dump_final_summaries(self.filtered_summaries_path, filtered_summaries)
        passing_count = self.visual_summary_filter.count_passing(
            judgments,
            self.visual_judgment_min_grade,
        )
        self.logger.info("Candidate visual summaries: %d", len(candidate_summaries))
        self.logger.info("Visual summary judgments: %d", len(judgments))
        self.logger.info("Passing visual summary judgments: %d", passing_count)
        self.logger.info("Filtered visual summaries: %d", len(filtered_summaries))
        self.logger.info("Wrote filtered summaries: %s", self.filtered_summaries_path)
        return filtered_summaries

    def export_outputs(
        self,
        summaries: List[FinalSummary],
        candidate_summaries: List[FinalSummary] | None = None,
    ) -> List[FinalSummary]:
        final_candidate_summaries = (
            summaries if candidate_summaries is None else candidate_summaries
        )
        self.export_visual_summaries(summaries)
        return self.filter_visual_summaries_with_judgments(final_candidate_summaries)

    def create_candidate_summaries(
        self,
        summaries: List[FinalSummary],
        combined_summaries: List[FinalSummary],
    ) -> List[FinalSummary]:
        if self.use_visual_combined_summaries:
            return combined_summaries + summaries
        return summaries

    def export_visual_summary_outputs(
        self,
        summaries: List[FinalSummary],
    ) -> List[FinalSummary]:
        combined_summaries = self.combine_visual_summaries(summaries)
        candidate_summaries = self.create_candidate_summaries(
            summaries,
            combined_summaries,
        )
        self.export_visual_summaries(summaries)
        filtered_summaries = self.filter_visual_summaries_with_judgments(
            candidate_summaries
        )
        self.logger.info("Single visual summaries: %d", len(summaries))
        self.logger.info("Combined visual summaries: %d", len(combined_summaries))
        self.logger.info("Candidate summaries: %d", len(candidate_summaries))
        self.logger.info("Filtered summaries: %d", len(filtered_summaries))
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
            filtered_summaries = self.export_visual_summary_outputs(summaries)
            self.logger.info(
                "Successful summaries: %d",
                len(summaries),
            )
            return filtered_summaries

        image_sections = self.create_image_sections()
        if not image_sections:
            self.logger.info("Successful summaries: 0")
            return self.export_visual_summary_outputs([])

        summary_results = self.get_visual_summarizer().summarize_sections(
            image_sections
        )
        summaries = self.create_final_summaries(image_sections, summary_results)
        self.logger.info("Successful summaries: %d", len(summaries))
        return self.export_visual_summary_outputs(summaries)
