import json
from pathlib import Path
from typing import Any, List
from uuid import UUID

from pytest import MonkeyPatch

from vidore_generation.dtos import ImageSection, LLMProviderConfig, Prompt
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Summary
from vidore_generation.pipelines import visual_summary_pipeline
from vidore_generation.pipelines.visual_summary_pipeline import VisualSummaryPipeline


class FakeGenerationHandler(GenerationHandler):
    def __init__(self) -> None:
        self.prompts: List[Prompt] = []

    def generate_single_sample(self, prompt: Prompt) -> Summary:
        self.prompts.append(prompt)
        return Summary(summary="visual summary")

    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[Summary]:
        self.prompts = prompts
        return [
            Summary(summary=f"visual summary {index}")
            for index, _prompt in enumerate(prompts)
        ]


def write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image")


def create_sections(pipeline: VisualSummaryPipeline) -> List[ImageSection]:
    try:
        return pipeline.create_image_sections()
    finally:
        pipeline.close_logger()


def make_pipeline(
    dataset_dir: Path,
    fake_handler: FakeGenerationHandler,
    monkeypatch: MonkeyPatch,
    **kwargs: Any,
) -> VisualSummaryPipeline:
    llm_provider = LLMProviderConfig(
        lm_model_name="fake-lm",
        vl_model_name="fake-vl",
    )

    def fake_make_generation_handler(
        llm_provider: LLMProviderConfig,
        role: str,
        logger: Any = None,
    ) -> FakeGenerationHandler:
        assert role == "vl"
        return fake_handler

    monkeypatch.setattr(
        visual_summary_pipeline,
        "make_generation_handler",
        fake_make_generation_handler,
    )
    return VisualSummaryPipeline(
        dataset_dir=dataset_dir,
        llm_provider=llm_provider,
        **kwargs,
    )


def test_max_windows_per_document_creates_one_window_per_document(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "doc_a" / "doc_a_0.png")
    write_fake_image(dataset_dir / "imgs" / "doc_a" / "doc_a_1.png")
    write_fake_image(dataset_dir / "imgs" / "doc_b" / "doc_b_0.png")
    write_fake_image(dataset_dir / "imgs" / "doc_b" / "doc_b_1.png")
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        max_windows_per_document=1,
    )

    sections = create_sections(pipeline)

    assert [(section.filename, section.page_numbers) for section in sections] == [
        ("doc_a", [0]),
        ("doc_b", [0]),
    ]


def test_max_windows_is_global_cap_with_max_windows_per_document(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    for document_name in ["doc_a", "doc_b", "doc_c"]:
        for page_number in range(4):
            write_fake_image(
                dataset_dir
                / "imgs"
                / document_name
                / f"{document_name}_{page_number}.png"
            )
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        max_windows_per_document=3,
        max_windows=4,
    )

    sections = create_sections(pipeline)

    assert [(section.filename, section.page_numbers) for section in sections] == [
        ("doc_a", [0]),
        ("doc_a", [1]),
        ("doc_a", [2]),
        ("doc_b", [0]),
    ]


def test_image_windows_use_numeric_page_sorting(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_10.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_2.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=2,
        stride=1,
    )

    sections = create_sections(pipeline)

    assert [section.page_numbers for section in sections] == [
        [1, 2],
        [2, 10],
        [10],
    ]


def test_run_exports_vidore_compatible_visual_summaries_without_markdowns(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        max_windows=2,
        filtered_summaries_nb=1,
        document_description="Internal technical documents.",
    )

    filtered_summaries = pipeline.run()

    assert not (dataset_dir / "markdowns").exists()
    assert len(filtered_summaries) == 1
    assert len(fake_handler.prompts) == 2

    visual_summaries_path = (
        dataset_dir / "visual_summaries" / "visual_summaries.json"
    )
    document_summaries_path = dataset_dir / "summaries" / "manual.json"
    filtered_summaries_path = (
        dataset_dir / "filtered_summaries" / "filtered_summaries.json"
    )
    assert visual_summaries_path.exists()
    assert document_summaries_path.exists()
    assert filtered_summaries_path.exists()

    exported = json.loads(filtered_summaries_path.read_text(encoding="utf-8"))
    assert len(exported) == 1
    assert exported[0]["summary"] == "visual summary 0"
    assert exported[0]["filenames"] == ["manual"]
    assert exported[0]["page_numbers"] == [[0]]
    assert exported[0]["addition_reason"] == "visual summary from page images"
    assert UUID(exported[0]["document_ids"][0])


def test_visual_pipeline_has_no_docling_or_markdown_pipeline_dependency() -> None:
    source = Path(visual_summary_pipeline.__file__).read_text(encoding="utf-8")

    assert "docling" not in source.lower()
    assert "parse_pdfs" not in source
    assert "LLMPipeline" not in source
    assert "SectionExtractor" not in source
