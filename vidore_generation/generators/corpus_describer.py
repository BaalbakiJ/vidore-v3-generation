import json
import logging
import os
from pathlib import Path
from typing import List

from vidore_generation.dtos import CorpusDescription, Document, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generators.base_generator import BaseGenerator


class CorpusDescriber(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("corpus_description.j2")
        self.generation_handler = generation_handler
        self.language = language

    def describe_corpus(self, documents: List[Document]) -> List[CorpusDescription]:
        document_descriptions = [
            document.document_description.description for document in documents
        ]
        prompt = self.create_prompt(
            {"descriptions": document_descriptions, "language": self.language},
        )
        corpus_description = self.generation_handler.generate_single_sample(
            Prompt(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                arguments={"pydantic_schema": CorpusDescription},
            )
        )
        return corpus_description

    def export(self, output_dir: Path, corpus_description: CorpusDescription):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "corpus_description.json"), "w") as file:
            json.dump(corpus_description.model_dump(), file)
