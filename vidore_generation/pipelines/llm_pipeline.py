import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from vidore_generation.dtos import Document, Failed, FinalSummary, LLMProviderConfig
from vidore_generation.generators.corpus_describer import CorpusDescriber
from vidore_generation.generators.document_descriptor import DocumentDescriptor
from vidore_generation.generators.section_extractor import SectionExtractor
from vidore_generation.generators.summarizer import Summarizer
from vidore_generation.pipelines.summary_pipeline import SummaryPipeline


class LLMPipeline(SummaryPipeline):
    def __init__(
        self,
        documents_dir: Path,
        model_name: str,
        persona: str,
        debug: bool = False,
        combination_iteration_nb: int = 20,
        sampling_multi_doc_ratio: float = 0.5,
        language: str = "english",
        filtered_summaries_nb: int = 400,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        llm_provider: Optional[LLMProviderConfig] = None,
    ):
        super().__init__(
            documents_dir=documents_dir,
            model_name=model_name,
            persona=persona,
            debug=debug,
            combination_iteration_nb=combination_iteration_nb,
            sampling_multi_doc_ratio=sampling_multi_doc_ratio,
            language=language,
            filtered_summaries_nb=filtered_summaries_nb,
            extra_kwargs=extra_kwargs,
            llm_provider=llm_provider,
        )

        self.document_descriptor = DocumentDescriptor(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            language=language,
        )
        self.corpus_describer = CorpusDescriber(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            language=language,
        )
        self.section_extractor = SectionExtractor(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            language=language,
        )
        self.summarizer = Summarizer(
            model_name=model_name,
            logger=self.logger,
            generation_handler=self.generation_handler,
            language=language,
        )
        self.corpus_stats = {}

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

    def get_documents(self) -> List[Document]:
        documents = []
        self.docid2filename = {}
        self.document_descriptions = {}
        for filepath in sorted(self.markdowns_dir.glob("*.md")):
            with open(filepath, "r") as f:
                document_markdown = f.read()
            documents.append(
                Document(
                    filename=filepath.stem,
                    content=document_markdown,
                    document_description=None,
                )
            )
            self.docid2filename[documents[-1].id] = filepath.stem
        document_descriptions_list = self.document_descriptor.describe_documents(
            documents
        )
        self.logger.debug(
            f"Document description: {document_descriptions_list[0].description}"
        )
        for document, document_description in zip(
            documents, document_descriptions_list
        ):
            document.document_description = document_description
        self.document_descriptor.export(
            os.path.join(self.documents_dir, "descriptions"),
            {
                document.filename: document.document_description
                for document in documents
            },
        )
        return documents

    def initialize_stats(self, documents: List[Document]):
        for document in documents:
            self.corpus_stats[document.filename] = {
                "nb_pages": len(document.content.split("<!-- page break -->")),
                "nb_sections": 0,
                "nb_summaries": 0,
                "nb_combined_summaries": 0,
                "nb_filtered_summaries": 0,
                "nb_filtered_combined_summaries": 0,
            }

    def describe_corpus(self, documents: List[Document]):
        if not os.path.exists(
            os.path.join(
                self.documents_dir, "corpus_description", "corpus_description.json"
            )
        ):
            corpus_description = self.corpus_describer.describe_corpus(documents)
            self.corpus_describer.export(
                os.path.join(self.documents_dir, "corpus_description"),
                corpus_description,
            )

    def extract_sections(self, documents):
        all_sections = []
        for document in tqdm(documents, desc="Extracting sections"):
            sections = self.section_extractor.extract_sections(document)
            all_sections.extend(sections)
            self.corpus_stats[document.filename]["nb_sections"] = len(sections)
            self.section_extractor.export(
                os.path.join(self.documents_dir, "sections"),
                sections,
                document.filename,
            )
        return all_sections

    def summarize_sections(self, sections):
        summaries_dir = os.path.join(self.documents_dir, "summaries")
        if os.path.exists(summaries_dir):
            summaries = self.summarizer.import_summaries(
                summaries_dir, self.docid2filename
            )
            self.logger.info(
                f"Loaded {len(summaries)} summaries for {len(sections)} sections"
            )
        else:
            section_summaries = self.summarizer.summarize_sections(sections)
            assert len(section_summaries) == len(sections)
            self.logger.debug(f"Section summary: {section_summaries[0]}")
            summaries = [
                FinalSummary(
                    summary=summary.summary,
                    document_ids=[section.document_id],
                    filenames=[section.filename],
                    page_numbers=[section.page_numbers],
                )
                for summary, section in zip(section_summaries, sections)
                if not isinstance(summary, Failed)
            ]
            for summary in summaries:
                self.corpus_stats[summary.filenames[0]]["nb_summaries"] += 1
            self.summarizer.export(summaries_dir, summaries, self.docid2filename)
        return summaries

    def run(self) -> List[Document]:
        self.logger.info("Running LLM Pipeline")
        documents = self.get_documents()
        self.describe_corpus(documents)
        self.initialize_stats(documents)
        sections = self.extract_sections(documents)
        self.logger.info(f"Stats: {self.corpus_stats}")
        summaries = self.summarize_sections(sections)
        combined_summaries = self.combine_summaries(documents, summaries)
        judgments = self.judge_summaries(summaries + combined_summaries)
        self.filter_summaries(summaries, combined_summaries, sections, judgments)
        self.logger.info(f"Stats: {self.corpus_stats}")
