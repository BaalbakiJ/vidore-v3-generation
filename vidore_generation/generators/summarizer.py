import json
import logging
import os
import warnings
from pathlib import Path
from typing import List

from vidore_generation.dtos import FinalSummary, Prompt, Section
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Summary
from vidore_generation.generators.base_generator import BaseGenerator

warnings.filterwarnings("ignore", category=UserWarning)


class Summarizer(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("section_summary.j2")
        self.generation_handler = generation_handler
        self.language = language

    def summarize_sections(self, sections: List[Section]) -> List[Summary]:
        return self.generation_handler.generate_multiple_samples(
            [
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": self.create_prompt(
                                {
                                    "document_description": section.document_description,
                                    "section": section.section,
                                    "language": self.language,
                                }
                            ),
                        }
                    ],
                    arguments={"pydantic_schema": Summary},
                )
                for section in sections
            ],
            semaphore_size=5,
            desc="Summarizing sections",
        )

    def export(
        self, output_dir: Path, summaries: List[FinalSummary], docid2filename: dict
    ):
        os.makedirs(output_dir, exist_ok=True)
        for docid in docid2filename:
            filename = docid2filename[docid]
            print(filename)
            sub_summaries = [
                summary for summary in summaries if summary.filenames[0] == filename
            ]
            with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
                json.dump(
                    [
                        json.loads(summary.model_dump_json())
                        for summary in sub_summaries
                    ],
                    f,
                    indent=4,
                )

    def import_summaries(
        self, output_dir: Path, docid2filename: dict
    ) -> List[FinalSummary]:
        summaries = []
        for docid in docid2filename:
            filename = docid2filename[docid]
            with open(os.path.join(output_dir, f"{filename}.json"), "r") as f:
                summaries.extend([FinalSummary(**item) for item in json.load(f)])
        return summaries
