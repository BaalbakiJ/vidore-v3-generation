import base64
from pathlib import Path
from typing import List

from vidore_generation.dtos import Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Description
from vidore_generation.generators.visual_document_descriptor import (
    VisualDocumentDescriptor,
    VisualDocumentSample,
)
from vidore_generation.generators.visual_summarizer import make_image_data_url


class FakeGenerationHandler(GenerationHandler):
    def __init__(self) -> None:
        self.prompts: List[Prompt] = []

    def generate_single_sample(self, prompt: Prompt) -> Description:
        self.prompts.append(prompt)
        return Description(description="Generated document context")

    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[Description]:
        self.prompts = prompts
        return [Description(description="Generated document context") for _ in prompts]


def write_fake_image(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_visual_document_descriptor_builds_text_and_image_blocks(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "manual_0.png"
    jpeg_path = tmp_path / "manual_1.jpeg"
    write_fake_image(png_path, b"png bytes")
    write_fake_image(jpeg_path, b"jpeg bytes")
    fake_handler = FakeGenerationHandler()
    descriptor = VisualDocumentDescriptor(
        model_name="fake-vl",
        logger=None,
        generation_handler=fake_handler,
        language="english",
        max_description_words=42,
    )

    descriptor.describe_documents(
        [
            VisualDocumentSample(
                filename="manual",
                image_paths=[str(png_path), str(jpeg_path)],
                page_numbers=[0, 1],
            )
        ]
    )

    assert len(fake_handler.prompts) == 1
    prompt = fake_handler.prompts[0]
    content = prompt.messages[0]["content"]
    assert [block["type"] for block in content] == [
        "text",
        "image_url",
        "image_url",
    ]
    assert "Write the description in english." in content[0]["text"]
    assert "Keep it under 42 words." in content[0]["text"]
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert content[2]["image_url"].startswith("data:image/jpeg;base64,")


def test_visual_document_descriptor_requests_description_schema(
    tmp_path: Path,
) -> None:
    png_path = tmp_path / "manual_0.png"
    write_fake_image(png_path, b"png bytes")
    fake_handler = FakeGenerationHandler()
    descriptor = VisualDocumentDescriptor(
        model_name="fake-vl",
        logger=None,
        generation_handler=fake_handler,
        language="english",
        max_description_words=150,
    )

    descriptor.describe_documents(
        [
            VisualDocumentSample(
                filename="manual",
                image_paths=[str(png_path)],
                page_numbers=[0],
            )
        ]
    )

    assert fake_handler.prompts[0].arguments["pydantic_schema"] is Description


def test_make_image_data_url_supports_png_and_jpeg(tmp_path: Path) -> None:
    png_path = tmp_path / "manual_0.png"
    jpg_path = tmp_path / "manual_1.jpg"
    write_fake_image(png_path, b"png bytes")
    write_fake_image(jpg_path, b"jpg bytes")

    assert make_image_data_url(str(png_path)) == (
        "data:image/png;base64,"
        f"{base64.b64encode(b'png bytes').decode('utf-8')}"
    )
    assert make_image_data_url(str(jpg_path)) == (
        "data:image/jpeg;base64,"
        f"{base64.b64encode(b'jpg bytes').decode('utf-8')}"
    )
