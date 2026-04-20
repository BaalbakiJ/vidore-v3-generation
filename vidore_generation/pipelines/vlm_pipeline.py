import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from vidore_generation.dtos import DocumentDescription, ImageSection
from vidore_generation.filters.summary_filter import SummaryFilter
from vidore_generation.generation_handlers.api_generation_handler import (
    APIGenerationHandler,
)
from vidore_generation.generators.judge import Judge
from vidore_generation.generators.vlm_query_generator import VLMQueryGenerator


class VLMPipeline:
    def __init__(
        self,
        dataset_dir: Path,
        lm_model_name: str = "fireworks_ai/kimi-k2p5",
        vl_model_name: str = "fireworks_ai/kimi-k2p5",
        persona: Optional[str] = None,
        debug: bool = False,
        section_size: int = 5,
        combination_iteration_nb: int = 20,
        inference_method: Literal["api", "vllm"] = "api",
        sampling_multi_doc_ratio: float = 0.5,
        language: str = "english",
        lm_extra_kwargs: Optional[Dict[str, Any]] = None,
        vl_extra_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_dir = dataset_dir
        self.lm_model_name = lm_model_name
        self.vl_model_name = vl_model_name
        self.persona = persona
        self.debug = debug
        self.combination_iteration_nb = combination_iteration_nb
        self.language = language
        self.section_size = section_size
        self.init_logger()

        if inference_method == "api":
            self.vl_generation_handler = APIGenerationHandler(
                model_name=vl_model_name, logger=self.logger, extra_kwargs=vl_extra_kwargs or {}
            )
            self.lm_generation_handler = APIGenerationHandler(
                model_name=lm_model_name, logger=self.logger, extra_kwargs=lm_extra_kwargs or {}
            )
        # elif inference_method == "vllm":
        #     self.generation_handler = VLLMGenerationHandler(model_name=model_name, logger=self.logger)
        else:
            raise ValueError(f"Invalid inference method: {inference_method}")

        self.vlm_query_generator = VLMQueryGenerator(
            model_name=self.vl_model_name,
            generation_handler=self.vl_generation_handler,
            language=language,
        )
        self.judge = Judge(
            model_name=lm_model_name,
            logger=self.logger,
            generation_handler=self.lm_generation_handler,
            language=language,
        )
        self.summary_filter = SummaryFilter()

        self.descriptions: Dict[str, DocumentDescription] = {}
        self.combinations: List[Dict[str, List[Any]]] = []
        self.image_sections_list: Dict[str, List[ImageSection]] = []

    def init_logger(self):
        logs_dir = os.path.join(self.dataset_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        self.logger = logging.getLogger("VLM Pipeline")
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

    def load_descriptions(self):
        self.descriptions = {}
        for filename in sorted(
            os.listdir(os.path.join(self.dataset_dir, "descriptions"))
        ):
            with open(
                os.path.join(self.dataset_dir, "descriptions", filename), "r"
            ) as file:
                description = DocumentDescription(**json.load(file))
            self.descriptions[filename[:-5]] = description

    def load_image_sections(self):
        self.image_sections_list = []
        imgs_dir = os.path.join(self.dataset_dir, "imgs")
        for directory in sorted(os.listdir(imgs_dir)):
            img_paths = [
                os.path.join(imgs_dir, directory, img_path)
                for img_path in sorted(
                    os.listdir(os.path.join(imgs_dir, directory)),
                    key=lambda x: int(x.split(".")[0].split("_")[-1]),
                )
            ]
            for i in range(0, len(img_paths), self.section_size):
                images = []
                section_image_paths = []
                for img_path in img_paths[i : i + self.section_size]:
                    images.append(Image.open(img_path))
                    section_image_paths.append(img_path)
                self.image_sections_list.append(
                    ImageSection(
                        filename=directory,
                        document_description=self.descriptions[directory].description,
                        images=images,
                        image_paths=section_image_paths,
                        page_numbers=[
                            i + j
                            for j in range(self.section_size)
                            if i + j < len(img_paths)
                        ],
                    )
                )

    def load_combinations(self):
        self.combinations = []
        imgs_dir = os.path.join(self.dataset_dir, "imgs")
        with open(
            os.path.join(
                self.dataset_dir, "combined_summaries", "combined_summaries.json"
            ),
            "r",
        ) as file:
            summary_combinations = json.load(file)
        for combination in summary_combinations:
            sections = []
            for filename, page_numbers in zip(
                combination["filenames"], combination["page_numbers"]
            ):
                images = []
                image_paths = []
                for page_number in page_numbers:
                    img_path = os.path.join(
                        imgs_dir, filename, f"{filename}_{page_number}.png"
                    )
                    images.append(Image.open(img_path))
                    image_paths.append(img_path)
                sections.append(
                    ImageSection(
                        filename=filename,
                        document_description=self.descriptions[filename].description,
                        images=images,
                        image_paths=image_paths,
                        page_numbers=page_numbers,
                    )
                )
            self.combinations.append(sections)

    def run(self):
        # Get document descriptions
        self.load_descriptions()
        self.load_image_sections()
        self.load_combinations()
        random.seed(42)

        print("Number of image sections: ", len(self.image_sections_list))
        print(
            "Number of pages per section: ",
            [
                len(image_section.image_paths)
                for image_section in self.image_sections_list
            ],
        )
        # raise Exception("test")
        # Generate queries on single sections
        single_section_queries = (
            self.vlm_query_generator.generate_single_section_queries(
                self.image_sections_list[:10]  # DEBUG
            )
        )
        exported_queries = []
        for query, image_section in zip(
            single_section_queries, self.image_sections_list[:10]
        ):
            exported_query = query.model_dump()
            exported_query["filenames"] = [image_section.filename]
            exported_query["page_numbers"] = [image_section.page_numbers]
            exported_queries.append(exported_query)
        queries_dir = os.path.join(self.dataset_dir, "queries")
        os.makedirs(queries_dir, exist_ok=True)
        with open(
            os.path.join(queries_dir, "vlm_single_section_queries.json"), "w"
        ) as file:
            json.dump(exported_queries, file, indent=4)

        # Generate queries on multi sections and multi-doc sections
        multi_section_queries = self.vlm_query_generator.generate_multi_section_queries(
            self.combinations[:10]
        )
        exported_queries = []
        for query, image_sections in zip(multi_section_queries, self.combinations[:10]):
            exported_query = query.model_dump()
            exported_query["filenames"] = [
                image_section.filename for image_section in image_sections
            ]
            exported_query["page_numbers"] = [
                image_section.page_numbers for image_section in image_sections
            ]
            exported_queries.append(exported_query)
        queries_dir = os.path.join(self.dataset_dir, "queries")
        os.makedirs(queries_dir, exist_ok=True)
        with open(
            os.path.join(queries_dir, "vlm_multi_section_queries.json"), "w"
        ) as file:
            json.dump(exported_queries, file, indent=4)

        # Generate queries on multi sections and multi-doc sections
        # all_sections = []
        # for image_sections in image_sections_list:
        #     summaries = self.vlm_summarizer.summarize_sections(image_sections)
        #     all_sections.append(summaries)
        # final_summaries = [
        #     FinalSummary(
        #         summary=summary.summary,
        #         document_ids=[image_section.document_id],
        #         filenames=[image_section.filename],
        #         page_numbers=[image_section.page_numbers],
        #     )
        #     for summary, image_section in zip(summaries, image_sections)
        # ]
        # self.vlm_summarizer.export(
        #     os.path.join(self.dataset_dir, "vlm_summaries"), final_summaries
        # )

        # # TODO: fix input variables
        # combined_summaries = self.combine_summaries(descriptions, final_summaries)
        # judgments = self.judge_summaries(combined_summaries)
        # self.filter_summaries(
        #     final_summaries, combined_summaries, image_sections_list, judgments
        # )
