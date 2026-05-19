import base64
import logging
from pathlib import Path
from typing import List

from vidore_generation.dtos import Failed, ImageSection, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Summary
from vidore_generation.generators.base_generator import BaseGenerator


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    raise ValueError(f"Unsupported image type for visual summary: {image_path}")


def make_image_data_url(image_path: str) -> str:
    return f"data:{get_image_media_type(image_path)};base64,{encode_image(image_path)}"


class VisualSummarizer(BaseGenerator):
    def __init__(
        self,
        model_name: str,
        logger: logging.Logger | None,
        generation_handler: GenerationHandler,
        language: str = "english",
        max_summary_words: int = 250,
    ):
        super().__init__(model_name)
        self.template = self.environment.get_template("visual_section_summary.j2")
        self.logger = logger
        self.generation_handler = generation_handler
        self.language = language
        self.max_summary_words = max_summary_words

    def summarize_sections(
        self,
        sections: List[ImageSection],
    ) -> List[Summary | Failed]:
        prompts = []
        for section in sections:
            content = [
                {
                    "type": "text",
                    "text": self.create_prompt(
                        {
                            "document_description": section.document_description,
                            "language": self.language,
                            "max_summary_words": self.max_summary_words,
                        }
                    ),
                }
            ]
            for image_path in section.image_paths:
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
                    arguments={"pydantic_schema": Summary},
                )
            )

        return self.generation_handler.generate_multiple_samples(
            prompts,
            semaphore_size=2,
            desc="Generating visual summaries",
        )
