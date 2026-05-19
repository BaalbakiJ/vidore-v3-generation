import logging
from typing import List

from pydantic import BaseModel

from vidore_generation.dtos import Failed, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Description
from vidore_generation.generators.base_generator import BaseGenerator
from vidore_generation.generators.visual_summarizer import make_image_data_url


class VisualDocumentSample(BaseModel):
    filename: str
    image_paths: list[str]
    page_numbers: list[int]


class VisualDocumentDescriptor(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        logger: logging.Logger | None,
        generation_handler: GenerationHandler,
        language: str = "english",
        max_description_words: int = 150,
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("visual_document_description.j2")
        self.logger = logger
        self.generation_handler = generation_handler
        self.language = language
        self.max_description_words = max_description_words

    def describe_documents(
        self,
        samples: List[VisualDocumentSample],
    ) -> List[Description | Failed]:
        prompts: list[Prompt] = []
        for sample in samples:
            content = [
                {
                    "type": "text",
                    "text": self.create_prompt(
                        {
                            "language": self.language,
                            "max_description_words": self.max_description_words,
                        }
                    ),
                }
            ]
            for image_path in sample.image_paths:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": make_image_data_url(image_path),
                    }
                )
            prompts.append(
                Prompt(
                    messages=[
                        {
                            "role": "user",
                            "content": content,
                        }
                    ],
                    arguments={"pydantic_schema": Description},
                )
            )

        return self.generation_handler.generate_multiple_samples(
            prompts,
            semaphore_size=2,
            desc="Describing visual documents",
        )
