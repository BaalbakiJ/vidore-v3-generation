import json
import logging
import os
import warnings
from pathlib import Path
from typing import List

from vidore_generation.dtos import Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Judgment, Summary
from vidore_generation.generators.base_generator import BaseGenerator

warnings.filterwarnings("ignore", category=UserWarning)


class Judge(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("judgment.j2")
        self.generation_handler = generation_handler
        self.language = language

    def judge_summaries(self, summaries: List[Summary], persona: str) -> List[Judgment]:
        return self.generation_handler.generate_multiple_samples(
            [
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": self.create_prompt(
                                {
                                    "summary": summary.summary,
                                    "persona": persona,
                                    "language": self.language,
                                }
                            ),
                        }
                    ],
                    arguments={"pydantic_schema": Judgment},
                )
                for summary in summaries
            ],
            semaphore_size=10,
            desc="Judging summaries",
        )

    def export(self, output_dir: Path, judgments: List[Judgment], filename: str):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "judgments.json"), "w") as f:
            json.dump(
                [json.loads(judgment.model_dump_json()) for judgment in judgments],
                f,
                indent=4,
            )

    def import_judgments(self, output_dir: Path) -> List[Judgment]:
        with open(os.path.join(output_dir, "judgments.json"), "r") as f:
            data = json.load(f)
        return [Judgment(**item) for item in data]
