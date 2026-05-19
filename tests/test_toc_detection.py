from vidore_generation.page_filtering.toc_detection import score_toc_page


def test_table_of_contents_with_trailing_page_numbers_is_detected() -> None:
    lines = [
        "Table of Contents",
        "Introduction 1",
        "Applicable regulations and standards 3",
        "4.1 Purpose 6",
        "4.2 Scope 7",
        "Risk management 12",
    ]

    result = score_toc_page(lines, page_index=1, total_pages=80)

    assert result["is_toc_page"] is True
    assert result["toc_has_title"] is True
    assert result["toc_trailing_page_number_count"] >= 4


def test_normal_technical_paragraph_page_is_not_detected() -> None:
    lines = [
        "The device software validates each uploaded file before processing.",
        "Failures are reported to the operator with a clear status message.",
        "The validation workflow is designed to preserve input data integrity.",
    ]

    result = score_toc_page(lines, page_index=12, total_pages=80)

    assert result["is_toc_page"] is False
    assert result["toc_like_line_count"] == 0


def test_late_page_without_toc_title_receives_late_penalty() -> None:
    lines = [
        "Introduction 1",
        "Applicable regulations and standards 3",
        "4.1 Purpose 6",
        "4.2 Scope 7",
        "Risk management 12",
        "Validation strategy 18",
        "Release criteria 25",
        "Appendix references 42",
    ]

    early_result = score_toc_page(lines, page_index=1, total_pages=120)
    late_result = score_toc_page(lines, page_index=80, total_pages=120)

    assert "late_without_title:-2" in late_result["toc_reasons"]
    assert late_result["toc_score"] < early_result["toc_score"]
