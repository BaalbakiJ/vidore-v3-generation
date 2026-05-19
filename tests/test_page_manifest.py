import json
from pathlib import Path

from vidore_generation.page_filtering import page_manifest
from vidore_generation.page_filtering.page_manifest import (
    get_excluded_image_page_numbers_by_filename,
    load_page_manifest,
)


def test_load_page_manifest_reads_jsonl_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "page_manifest.jsonl"
    rows = [
        {"filename": "manual", "image_page_number": 0},
        {"filename": "manual", "image_page_number": 1},
    ]
    manifest_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded_rows = load_page_manifest(manifest_path)

    assert loaded_rows == rows


def test_get_excluded_image_page_numbers_groups_by_filename() -> None:
    manifest_rows = [
        {
            "filename": "manual",
            "page_number": 99,
            "image_page_number": 1,
            "exclude_from_visual_summaries": True,
        },
        {
            "filename": "manual",
            "page_number": 100,
            "image_page_number": 2,
            "exclude_from_visual_summaries": False,
        },
        {
            "filename": "guide",
            "page_number": 42,
            "image_page_number": 0,
            "exclude_from_visual_summaries": True,
        },
    ]

    excluded = get_excluded_image_page_numbers_by_filename(
        manifest_rows,
        "exclude_from_visual_summaries",
    )

    assert excluded == {"manual": {1}, "guide": {0}}


def test_page_manifest_module_does_not_require_pandas() -> None:
    source = Path(page_manifest.__file__).read_text(encoding="utf-8")

    assert "pandas" not in source
