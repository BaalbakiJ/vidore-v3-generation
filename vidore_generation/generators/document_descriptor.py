import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import litellm

from vidore_generation.dtos import Document, DocumentDescription, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Description
from vidore_generation.generators.base_generator import BaseGenerator


class DocumentDescriptor(BaseGenerator):
    def __init__(
        self,
        model_name: str = "fireworks_ai/kimi-k2p5",
        logger: logging.Logger = None,
        generation_handler: GenerationHandler = None,
        language: str = "english",
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("document_description.j2")
        self.generation_handler = generation_handler
        self.language = language

    def describe_documents(
        self, documents: List[Document]
    ) -> List[DocumentDescription]:
        shortened_documents = []
        for document in documents:
            document_content = document.content
            token_count = litellm.token_counter(
                messages=[{"role": "user", "content": document_content}],
                model=self.model_name,
            )
            while token_count > 100_000:
                document_content = document_content[:-50_000]
                token_count = litellm.token_counter(
                    messages=[{"role": "user", "content": document_content}],
                    model=self.model_name,
                )
            shortened_documents.append(
                Document(
                    id=document.id,
                    content=document_content,
                    filename=document.filename,
                    document_description=document.document_description,
                )
            )

        results = self.generation_handler.generate_multiple_samples(
            [
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": self.create_prompt(
                                {"content": document.content, "language": self.language}
                            ),
                        }
                    ],
                    arguments={"pydantic_schema": Description},
                )
                for document in shortened_documents
            ],
            semaphore_size=2,
            desc="Describing documents",
        )
        return [
            DocumentDescription(document_id=document.id, description=result.description)
            for document, result in zip(documents, results)
        ]

    def export(
        self, output_dir: Path, document_descriptions: Dict[str, DocumentDescription]
    ):
        os.makedirs(output_dir, exist_ok=True)
        for filename, document_description in document_descriptions.items():
            with open(os.path.join(output_dir, f"{filename}.json"), "w") as f:
                json.dump(
                    json.loads(document_description.model_dump_json()), f, indent=4
                )
