import csv
import json
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from vidore_generation.page_filtering.toc_detection import (
    extract_page_lines_with_status,
    score_toc_page,
)


PAGE_MANIFEST_FIELDS = [
    "filename",
    "pdf_filename",
    "pdf_path",
    "page_index",
    "page_number",
    "image_page_number",
    "is_toc_page",
    "exclude_from_visual_summaries",
    "exclude_from_image_rendering",
    "exclusion_reason",
    "text_preview",
    "parse_status",
    "toc_needs_manual_review",
    "toc_score",
    "toc_has_title",
    "toc_is_early_page",
    "toc_eligible_line_count",
    "toc_like_line_count",
    "toc_like_ratio",
    "toc_dotted_count",
    "toc_trailing_page_number_count",
    "toc_section_number_count",
    "toc_reasons",
    "toc_sample_lines",
]
TOC_SUMMARY_FIELDS = [
    "filename",
    "pdf_filename",
    "pdf_path",
    "total_pages",
    "toc_detected",
    "toc_pages",
    "toc_page_count",
    "toc_manual_review_pages",
    "toc_manual_review_page_count",
    "status",
    "error",
]
EXCLUSION_FIELDS = {
    "exclude_from_image_rendering",
    "exclude_from_visual_summaries",
}

ManifestRow = dict[str, Any]


def _make_error_message(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _serialize_csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, set)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _write_jsonl(path: Path, rows: list[ManifestRow]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, fieldnames: list[str], rows: list[ManifestRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    fieldname: _serialize_csv_value(row.get(fieldname, ""))
                    for fieldname in fieldnames
                }
            )


def _close_pdf_document(document: Any) -> None:
    if document is not None:
        document.close()


def _build_page_row(
    pdf_path: Path,
    page_index: int,
    total_pages: int,
    lines: list[str],
    parse_status: str,
) -> ManifestRow:
    toc_score = score_toc_page(lines, page_index, total_pages)
    is_toc_page = toc_score["is_toc_page"]

    return {
        "filename": pdf_path.stem,
        "pdf_filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "page_index": page_index,
        "page_number": page_index + 1,
        "image_page_number": page_index,
        "is_toc_page": is_toc_page,
        "exclude_from_visual_summaries": is_toc_page,
        "exclude_from_image_rendering": is_toc_page,
        "exclusion_reason": "toc_page" if is_toc_page else "",
        "text_preview": "\n".join(lines)[:500],
        "parse_status": parse_status,
        **toc_score,
    }


def _build_error_summary_row(pdf_path: Path, error: Exception) -> ManifestRow:
    return {
        "filename": pdf_path.stem,
        "pdf_filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "total_pages": 0,
        "toc_detected": False,
        "toc_pages": [],
        "toc_page_count": 0,
        "toc_manual_review_pages": [],
        "toc_manual_review_page_count": 0,
        "status": "error",
        "error": _make_error_message(error),
    }


def _build_summary_row(
    pdf_path: Path,
    total_pages: int,
    toc_pages: list[int],
    toc_manual_review_pages: list[int],
) -> ManifestRow:
    return {
        "filename": pdf_path.stem,
        "pdf_filename": pdf_path.name,
        "pdf_path": str(pdf_path),
        "total_pages": total_pages,
        "toc_detected": bool(toc_pages),
        "toc_pages": toc_pages,
        "toc_page_count": len(toc_pages),
        "toc_manual_review_pages": toc_manual_review_pages,
        "toc_manual_review_page_count": len(toc_manual_review_pages),
        "status": "ok",
        "error": "",
    }


def _validate_pdfs_dir(pdfs_dir: Path) -> None:
    if not pdfs_dir.exists():
        raise FileNotFoundError(f"PDF directory does not exist: {pdfs_dir}")
    if not pdfs_dir.is_dir():
        raise NotADirectoryError(f"PDF path is not a directory: {pdfs_dir}")
    if pdfs_dir.name != "pdfs":
        raise ValueError(f"PDF directory must be named 'pdfs': {pdfs_dir}")


def build_page_manifest(
    pdfs_dir: Path,
    output_dir: Path,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    _validate_pdfs_dir(pdfs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_rows: list[ManifestRow] = []
    summary_rows: list[ManifestRow] = []

    for pdf_path in sorted(pdfs_dir.glob("*.pdf")):
        document = None
        try:
            document = pdfium.PdfDocument(pdf_path)
            total_pages = len(document)
            toc_pages: list[int] = []
            toc_manual_review_pages: list[int] = []

            for page_index in range(total_pages):
                try:
                    page = document[page_index]
                    lines, parse_status = extract_page_lines_with_status(page)
                except Exception as error:
                    lines = []
                    parse_status = f"error:{type(error).__name__}:{error}"

                page_row = _build_page_row(
                    pdf_path=pdf_path,
                    page_index=page_index,
                    total_pages=total_pages,
                    lines=lines,
                    parse_status=parse_status,
                )
                page_rows.append(page_row)
                if page_row["is_toc_page"]:
                    toc_pages.append(page_row["page_number"])
                if page_row["toc_needs_manual_review"]:
                    toc_manual_review_pages.append(page_row["page_number"])

            summary_rows.append(
                _build_summary_row(
                    pdf_path=pdf_path,
                    total_pages=total_pages,
                    toc_pages=toc_pages,
                    toc_manual_review_pages=toc_manual_review_pages,
                )
            )
        except Exception as error:
            summary_rows.append(_build_error_summary_row(pdf_path, error))
        finally:
            _close_pdf_document(document)

    _write_jsonl(output_dir / "page_manifest.jsonl", page_rows)
    _write_csv(output_dir / "page_manifest.csv", PAGE_MANIFEST_FIELDS, page_rows)
    _write_csv(
        output_dir / "toc_detection_summary.csv",
        TOC_SUMMARY_FIELDS,
        summary_rows,
    )

    return page_rows, summary_rows


def load_page_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with open(path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            row = json.loads(stripped_line)
            if not isinstance(row, dict):
                raise ValueError(
                    "Page manifest JSONL rows must be JSON objects: "
                    f"{path}:{line_number}"
                )
            rows.append(row)
    return rows


def get_excluded_image_page_numbers_by_filename(
    manifest_rows: list[ManifestRow],
    exclusion_field: str,
) -> dict[str, set[int]]:
    if exclusion_field not in EXCLUSION_FIELDS:
        raise ValueError(
            "Unsupported page manifest exclusion field: "
            f"{exclusion_field}. Expected one of {sorted(EXCLUSION_FIELDS)}"
        )

    excluded_page_numbers: dict[str, set[int]] = {}
    for row in manifest_rows:
        if not row.get(exclusion_field, False):
            continue
        filename = str(row["filename"])
        image_page_number = int(row["image_page_number"])
        if filename not in excluded_page_numbers:
            excluded_page_numbers[filename] = set()
        excluded_page_numbers[filename].add(image_page_number)
    return excluded_page_numbers
