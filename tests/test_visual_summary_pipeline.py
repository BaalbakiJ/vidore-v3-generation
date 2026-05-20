import json
from pathlib import Path
from typing import Any, List
from uuid import UUID

from pytest import MonkeyPatch

from vidore_generation.dtos import (
    CombinedSummary,
    DocumentDescription,
    Failed,
    FinalSummary,
    ImageSection,
    IndexedSummary,
    LLMProviderConfig,
    Prompt,
)
from vidore_generation.generation_handlers.generation_handler import GenerationHandler
from vidore_generation.generation_schemas import Description, Judgment, Score, Summary
from vidore_generation.pipelines import visual_summary_pipeline
from vidore_generation.pipelines.visual_summary_pipeline import (
    VisualSummaryPipeline,
    dump_final_summaries,
    get_document_id,
)


class FakeGenerationHandler(GenerationHandler):
    def __init__(self, results: List[Any] | None = None) -> None:
        self.prompts: List[Prompt] = []
        self.results = results
        self.call_count = 0

    def generate_single_sample(self, prompt: Prompt) -> Any:
        self.prompts.append(prompt)
        if self.results is not None:
            return self.results[0]
        return Summary(summary="visual summary")

    def generate_multiple_samples(
        self,
        prompts: List[Prompt],
        semaphore_size: int = 20,
        desc: str = "Generating samples",
    ) -> List[Any]:
        self.prompts = prompts
        self.call_count += 1
        if self.results is not None:
            return self.results[: len(prompts)]
        return [
            Summary(summary=f"visual summary {index}")
            for index, _prompt in enumerate(prompts)
        ]


class FakeSummaryCombinator:
    def __init__(self) -> None:
        self.call_count = 0
        self.documents: List[Any] = []
        self.random_seeds: List[int] = []

    def combine_summaries(
        self,
        documents: List[Any],
        summaries: List[FinalSummary],
        random_seeds: List[int],
    ) -> List[CombinedSummary]:
        self.call_count += 1
        self.documents = documents
        self.random_seeds = random_seeds
        return [
            CombinedSummary(
                summaries=[
                    IndexedSummary(
                        summary=summaries[0].summary,
                        document_id=summaries[0].document_ids[0],
                        filename=summaries[0].filenames[0],
                        page_numbers=summaries[0].page_numbers[0],
                        summary_id=summaries[0].id,
                    ),
                    IndexedSummary(
                        summary=summaries[1].summary,
                        document_id=summaries[1].document_ids[0],
                        filename=summaries[1].filenames[0],
                        page_numbers=summaries[1].page_numbers[0],
                        summary_id=summaries[1].id,
                    ),
                ],
                combined_summary="combined visual summary",
            )
        ]


class FakeJudge:
    def __init__(self, judgments: List[Any]) -> None:
        self.judgments = judgments
        self.call_count = 0
        self.summaries: List[FinalSummary] = []
        self.persona = ""

    def judge_summaries(
        self,
        summaries: List[FinalSummary],
        persona: str,
    ) -> List[Any]:
        self.call_count += 1
        self.summaries = summaries
        self.persona = persona
        return self.judgments[: len(summaries)]


def make_judgment(grade: int) -> Judgment:
    return Judgment(
        information_richness=Score(grade=grade, explanation="x"),
        persona_relevance=Score(grade=grade, explanation="x"),
        query_generation_potential=Score(grade=grade, explanation="x"),
        conceptual_clarity=Score(grade=grade, explanation="x"),
    )


def write_fake_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real image")


def create_final_summary(
    filename: str,
    summary: str,
    page_numbers: List[int],
) -> FinalSummary:
    return FinalSummary(
        summary=summary,
        document_ids=[get_document_id(filename)],
        filenames=[filename],
        page_numbers=[page_numbers],
        addition_reason="visual summary from page images",
    )


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


def test_image_sections_use_fallback_document_description_by_default(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        document_description="Fallback document context.",
    )

    sections = create_sections(pipeline)

    assert [section.document_description for section in sections] == [
        "Fallback document context."
    ]
    assert fake_handler.call_count == 0


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


def test_respect_page_manifest_excludes_visual_summary_images(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_2.png")
    manifest_rows = [
        {
            "filename": "manual",
            "page_number": 2,
            "image_page_number": 1,
            "exclude_from_visual_summaries": True,
        }
    ]
    (dataset_dir / "page_manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        respect_page_manifest=True,
    )

    sections = create_sections(pipeline)

    assert [section.page_numbers for section in sections] == [[0], [2]]


def test_generated_visual_document_descriptions_are_used_when_enabled(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    fake_handler = FakeGenerationHandler(
        results=[Description(description="Generated doc context")]
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        use_visual_document_descriptions=True,
        document_description="Fallback document context.",
    )

    sections = create_sections(pipeline)

    assert [section.document_description for section in sections] == [
        "Generated doc context",
        "Generated doc context",
    ]
    description_path = dataset_dir / "descriptions" / "manual.json"
    exported_description = json.loads(description_path.read_text(encoding="utf-8"))
    assert exported_description == {
        "document_id": str(get_document_id("manual")),
        "description": "Generated doc context",
    }


def test_existing_visual_document_descriptions_are_loaded_when_present(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    description_path = dataset_dir / "descriptions" / "manual.json"
    description_path.parent.mkdir(parents=True, exist_ok=True)
    description_path.write_text(
        DocumentDescription(
            document_id=get_document_id("manual"),
            description="Existing doc context",
        ).model_dump_json(),
        encoding="utf-8",
    )
    fake_handler = FakeGenerationHandler(
        results=[Description(description="Generated doc context")]
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        use_visual_document_descriptions=True,
        overwrite_descriptions=False,
        document_description="Fallback document context.",
    )

    sections = create_sections(pipeline)

    assert [section.document_description for section in sections] == [
        "Existing doc context"
    ]
    assert fake_handler.call_count == 0


def test_visual_document_description_samples_respect_page_manifest(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_2.png")
    manifest_rows = [
        {
            "filename": "manual",
            "page_number": 2,
            "image_page_number": 1,
            "exclude_from_visual_summaries": True,
        }
    ]
    (dataset_dir / "page_manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )
    fake_handler = FakeGenerationHandler()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        respect_page_manifest=True,
        max_description_pages=3,
    )

    try:
        samples = pipeline.create_visual_document_samples(
            [dataset_dir / "imgs" / "manual"]
        )
    finally:
        pipeline.close_logger()

    assert len(samples) == 1
    assert samples[0].page_numbers == [0, 2]
    assert [
        Path(image_path).name for image_path in samples[0].image_paths
    ] == [
        "manual_0.png",
        "manual_2.png",
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
    assert not (dataset_dir / "combined_summaries").exists()
    assert not (dataset_dir / "judgments").exists()

    exported = json.loads(filtered_summaries_path.read_text(encoding="utf-8"))
    assert len(exported) == 1
    assert exported[0]["summary"] == "visual summary 0"
    assert exported[0]["filenames"] == ["manual"]
    assert exported[0]["page_numbers"] == [[0]]
    assert exported[0]["addition_reason"] == "visual summary from page images"
    assert UUID(exported[0]["document_ids"][0])


def test_combined_visual_summaries_are_generated_when_enabled(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    fake_handler = FakeGenerationHandler()
    fake_combinator = FakeSummaryCombinator()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        max_windows=2,
        filtered_summaries_nb=1,
        use_visual_combined_summaries=True,
        combination_iteration_nb=3,
    )
    monkeypatch.setattr(
        pipeline,
        "get_summary_combinator",
        lambda: fake_combinator,
    )

    filtered_summaries = pipeline.run()

    combined_summaries_path = (
        dataset_dir / "combined_summaries" / "combined_summaries.json"
    )
    visual_summaries_path = (
        dataset_dir / "visual_summaries" / "visual_summaries.json"
    )
    document_summaries_path = dataset_dir / "summaries" / "manual.json"
    assert combined_summaries_path.exists()
    assert len(filtered_summaries) == 1
    assert filtered_summaries[0].summary == "combined visual summary"
    assert fake_combinator.call_count == 1
    assert fake_combinator.random_seeds == [0, 1, 2]
    assert [
        (
            document.id,
            document.filename,
            document.content,
            document.document_description,
        )
        for document in fake_combinator.documents
    ] == [(get_document_id("manual"), "manual", "", None)]

    exported_combined = json.loads(
        combined_summaries_path.read_text(encoding="utf-8")
    )
    assert exported_combined == [
        {
            "id": exported_combined[0]["id"],
            "summary": "combined visual summary",
            "document_ids": [
                str(get_document_id("manual")),
                str(get_document_id("manual")),
            ],
            "filenames": ["manual", "manual"],
            "page_numbers": [[0], [1]],
            "original_summaries": ["visual summary 0", "visual summary 1"],
            "judgments": None,
            "addition_reason": "visual combined summary",
        }
    ]

    exported_visual = json.loads(visual_summaries_path.read_text(encoding="utf-8"))
    exported_document = json.loads(
        document_summaries_path.read_text(encoding="utf-8")
    )
    assert [item["summary"] for item in exported_visual] == [
        "visual summary 0",
        "visual summary 1",
    ]
    assert [item["summary"] for item in exported_document] == [
        "visual summary 0",
        "visual summary 1",
    ]


def test_visual_summary_judging_writes_judgments_and_filtered_summaries(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_0.png")
    write_fake_image(dataset_dir / "imgs" / "manual" / "manual_1.png")
    fake_handler = FakeGenerationHandler()
    fake_combinator = FakeSummaryCombinator()
    fake_judge = FakeJudge(
        [
            make_judgment(5),
            make_judgment(5),
            make_judgment(5),
        ]
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        section_size=1,
        stride=1,
        max_windows=2,
        filtered_summaries_nb=2,
        use_visual_combined_summaries=True,
        use_visual_summary_judging=True,
        persona="A quality engineer.",
    )
    monkeypatch.setattr(
        pipeline,
        "get_summary_combinator",
        lambda: fake_combinator,
    )
    monkeypatch.setattr(pipeline, "get_judge", lambda: fake_judge)

    filtered_summaries = pipeline.run()

    judgments_path = dataset_dir / "judgments" / "judgments.json"
    filtered_summaries_path = (
        dataset_dir / "filtered_summaries" / "filtered_summaries.json"
    )
    assert judgments_path.exists()
    assert filtered_summaries_path.exists()
    assert fake_judge.call_count == 1
    assert fake_judge.persona == "A quality engineer."
    assert [summary.summary for summary in fake_judge.summaries] == [
        "combined visual summary",
        "visual summary 0",
        "visual summary 1",
    ]
    assert [summary.addition_reason for summary in filtered_summaries] == [
        "visual judged same-document combined summary",
        "visual judged single summary",
    ]
    assert [summary.page_numbers for summary in filtered_summaries] == [
        [[0], [1]],
        [[0]],
    ]

    exported_judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    exported_filtered = json.loads(
        filtered_summaries_path.read_text(encoding="utf-8")
    )
    assert len(exported_judgments) == 3
    assert [item["addition_reason"] for item in exported_filtered] == [
        "visual judged same-document combined summary",
        "visual judged single summary",
    ]
    assert exported_filtered[0]["page_numbers"] == [[0], [1]]


def test_existing_visual_judgments_are_reused(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_judge = FakeJudge([make_judgment(1)])
    judgments_path = dataset_dir / "judgments" / "judgments.json"
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    judgments_path.write_text(
        json.dumps([json.loads(make_judgment(5).model_dump_json())]),
        encoding="utf-8",
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_summary_judging=True,
        overwrite_judgments=False,
        filtered_summaries_nb=1,
    )
    monkeypatch.setattr(pipeline, "get_judge", lambda: fake_judge)
    candidates = [create_final_summary("manual", "visual summary 0", [0])]

    try:
        filtered_summaries = pipeline.filter_visual_summaries_with_judgments(
            candidates
        )
    finally:
        pipeline.close_logger()

    assert fake_judge.call_count == 0
    assert len(filtered_summaries) == 1
    assert filtered_summaries[0].addition_reason == "visual judged single summary"


def test_mismatched_existing_visual_judgments_are_regenerated(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_judge = FakeJudge([make_judgment(5), make_judgment(5)])
    judgments_path = dataset_dir / "judgments" / "judgments.json"
    judgments_path.parent.mkdir(parents=True, exist_ok=True)
    judgments_path.write_text(
        json.dumps([json.loads(make_judgment(5).model_dump_json())]),
        encoding="utf-8",
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_summary_judging=True,
        overwrite_judgments=False,
        filtered_summaries_nb=2,
    )
    monkeypatch.setattr(pipeline, "get_judge", lambda: fake_judge)
    candidates = [
        create_final_summary("manual", "visual summary 0", [0]),
        create_final_summary("manual", "visual summary 1", [1]),
    ]

    try:
        filtered_summaries = pipeline.filter_visual_summaries_with_judgments(
            candidates
        )
    finally:
        pipeline.close_logger()

    exported_judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert fake_judge.call_count == 1
    assert len(exported_judgments) == 2
    assert [summary.summary for summary in filtered_summaries] == [
        "visual summary 0",
        "visual summary 1",
    ]


def test_failed_visual_judgment_becomes_low_score_judgment(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_judge = FakeJudge(
        [
            Failed(error="judge timeout"),
            make_judgment(5),
        ]
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_summary_judging=True,
        filtered_summaries_nb=2,
    )
    monkeypatch.setattr(pipeline, "get_judge", lambda: fake_judge)
    candidates = [
        create_final_summary("manual", "failed judgment summary", [0]),
        create_final_summary("manual", "passing judgment summary", [1]),
    ]

    try:
        filtered_summaries = pipeline.filter_visual_summaries_with_judgments(
            candidates
        )
    finally:
        pipeline.close_logger()

    judgments_path = dataset_dir / "judgments" / "judgments.json"
    exported_judgments = json.loads(judgments_path.read_text(encoding="utf-8"))
    assert fake_judge.call_count == 1
    assert [summary.summary for summary in filtered_summaries] == [
        "passing judgment summary",
        "failed judgment summary",
    ]
    assert [summary.addition_reason for summary in filtered_summaries] == [
        "visual judged single summary",
        "visual fallback selected summary",
    ]
    assert exported_judgments[0]["information_richness"]["grade"] == 1
    assert "judge timeout" in exported_judgments[0]["information_richness"][
        "explanation"
    ]


def test_existing_combined_visual_summaries_are_reused(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_combinator = FakeSummaryCombinator()
    existing_combined_summary = FinalSummary(
        summary="existing combined",
        document_ids=[get_document_id("manual"), get_document_id("manual")],
        filenames=["manual", "manual"],
        page_numbers=[[0], [1]],
        original_summaries=["visual summary 0", "visual summary 1"],
        addition_reason="visual combined summary",
    )
    dump_final_summaries(
        dataset_dir / "combined_summaries" / "combined_summaries.json",
        [existing_combined_summary],
    )
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_combined_summaries=True,
        overwrite_combined_summaries=False,
    )
    monkeypatch.setattr(
        pipeline,
        "get_summary_combinator",
        lambda: fake_combinator,
    )
    summaries = [
        create_final_summary("manual", "visual summary 0", [0]),
        create_final_summary("manual", "visual summary 1", [1]),
    ]

    try:
        combined_summaries = pipeline.combine_visual_summaries(summaries)
    finally:
        pipeline.close_logger()

    assert [summary.summary for summary in combined_summaries] == [
        "existing combined"
    ]
    assert fake_combinator.call_count == 0


def test_overwrite_combined_visual_summaries_regenerates(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_combinator = FakeSummaryCombinator()
    existing_combined_summary = FinalSummary(
        summary="existing combined",
        document_ids=[get_document_id("manual"), get_document_id("manual")],
        filenames=["manual", "manual"],
        page_numbers=[[0], [1]],
        original_summaries=["visual summary 0", "visual summary 1"],
        addition_reason="visual combined summary",
    )
    combined_summaries_path = (
        dataset_dir / "combined_summaries" / "combined_summaries.json"
    )
    dump_final_summaries(combined_summaries_path, [existing_combined_summary])
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_combined_summaries=True,
        overwrite_combined_summaries=True,
    )
    monkeypatch.setattr(
        pipeline,
        "get_summary_combinator",
        lambda: fake_combinator,
    )
    summaries = [
        create_final_summary("manual", "visual summary 0", [0]),
        create_final_summary("manual", "visual summary 1", [1]),
    ]

    try:
        combined_summaries = pipeline.combine_visual_summaries(summaries)
    finally:
        pipeline.close_logger()

    assert [summary.summary for summary in combined_summaries] == [
        "combined visual summary"
    ]
    assert fake_combinator.call_count == 1
    exported_combined = json.loads(combined_summaries_path.read_text(encoding="utf-8"))
    assert [item["summary"] for item in exported_combined] == [
        "combined visual summary"
    ]


def test_not_enough_visual_summaries_does_not_crash_combination(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "dataset"
    fake_handler = FakeGenerationHandler()
    fake_combinator = FakeSummaryCombinator()
    pipeline = make_pipeline(
        dataset_dir,
        fake_handler,
        monkeypatch,
        use_visual_combined_summaries=True,
    )
    monkeypatch.setattr(
        pipeline,
        "get_summary_combinator",
        lambda: fake_combinator,
    )
    summaries = [create_final_summary("manual", "visual summary 0", [0])]

    try:
        combined_summaries = pipeline.combine_visual_summaries(summaries)
        filtered_summaries = pipeline.export_outputs(summaries, summaries)
    finally:
        pipeline.close_logger()

    assert combined_summaries == []
    assert fake_combinator.call_count == 0
    assert len(filtered_summaries) == 1
    assert (
        dataset_dir / "filtered_summaries" / "filtered_summaries.json"
    ).exists()
    assert not (dataset_dir / "combined_summaries").exists()


def test_visual_pipeline_has_no_docling_or_markdown_pipeline_dependency() -> None:
    source = Path(visual_summary_pipeline.__file__).read_text(encoding="utf-8")

    assert "docling" not in source.lower()
    assert "parse_pdfs" not in source
    assert "LLMPipeline" not in source
    assert "SectionExtractor" not in source
