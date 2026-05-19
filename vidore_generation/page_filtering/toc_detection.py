import re
from typing import Any, TypedDict


TOC_TITLE_RE = re.compile(
    r"\b("
    r"table\s+of\s+contents|"
    r"contents|"
    r"sommaire|"
    r"table\s+des\s+mati[eè]res|"
    r"toc"
    r")\b",
    re.IGNORECASE,
)
DOTTED_LEADER_RE = re.compile(r"^.{4,}(?:\.{2,}|\u2026+)\s*\d{1,4}$")
TRAILING_PAGE_NUMBER_RE = re.compile(r"^(?=.*[^\W\d_]).{4,}\s+\d{1,4}$")
SECTION_NUMBER_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[A-Za-z]\.|\([A-Za-z]\))\s+.+$"
)
MOSTLY_NUMERIC_RE = re.compile(r"^[\d\s.\-_/]+$")

MIN_ELIGIBLE_LINE_LENGTH = 5
MAX_SAMPLE_LINES = 5


class TocScore(TypedDict):
    is_toc_page: bool
    toc_needs_manual_review: bool
    toc_score: int
    toc_has_title: bool
    toc_is_early_page: bool
    toc_eligible_line_count: int
    toc_like_line_count: int
    toc_like_ratio: float
    toc_dotted_count: int
    toc_trailing_page_number_count: int
    toc_section_number_count: int
    toc_reasons: list[str]
    toc_sample_lines: list[str]


def normalize_toc_line(line: str) -> str:
    normalized_line = line.replace("\u00a0", " ")
    normalized_line = re.sub(r"\s+", " ", normalized_line)
    return normalized_line.strip()


def _is_eligible_toc_line(line: str) -> bool:
    normalized_line = normalize_toc_line(line)
    if len(normalized_line) < MIN_ELIGIBLE_LINE_LENGTH:
        return False
    return MOSTLY_NUMERIC_RE.fullmatch(normalized_line) is None


def _has_trailing_page_number(line: str) -> bool:
    return TRAILING_PAGE_NUMBER_RE.match(line) is not None


def _has_dotted_leader(line: str) -> bool:
    return DOTTED_LEADER_RE.match(line) is not None


def _has_section_number(line: str) -> bool:
    return SECTION_NUMBER_RE.match(line) is not None


def is_toc_like_line(line: str) -> bool:
    normalized_line = normalize_toc_line(line)
    if not _is_eligible_toc_line(normalized_line):
        return False

    has_trailing_page_number = _has_trailing_page_number(normalized_line)
    return (
        _has_dotted_leader(normalized_line)
        or has_trailing_page_number
        or (_has_section_number(normalized_line) and has_trailing_page_number)
    )


def score_toc_page(lines: list[str], page_index: int, total_pages: int) -> TocScore:
    normalized_lines = [
        normalize_toc_line(line)
        for line in lines
        if normalize_toc_line(line)
    ]
    eligible_lines = [
        line for line in normalized_lines if _is_eligible_toc_line(line)
    ]

    has_toc_title = any(TOC_TITLE_RE.search(line) for line in normalized_lines[:15])
    dotted_count = sum(1 for line in eligible_lines if _has_dotted_leader(line))
    trailing_page_number_count = sum(
        1 for line in eligible_lines if _has_trailing_page_number(line)
    )
    section_number_count = sum(
        1
        for line in eligible_lines
        if _has_section_number(line) and _has_trailing_page_number(line)
    )
    toc_like_lines = [line for line in eligible_lines if is_toc_like_line(line)]
    eligible_count = len(eligible_lines)
    toc_like_count = len(toc_like_lines)
    toc_like_ratio = toc_like_count / eligible_count if eligible_count else 0.0
    early_page_limit = min(25, max(8, int(total_pages * 0.25)))
    is_early_page = page_index < early_page_limit

    score = 0
    reasons: list[str] = []

    if has_toc_title:
        score += 4
        reasons.append("toc_title:+4")

    if dotted_count >= 5:
        score += 3
        reasons.append("dotted_leaders>=5:+3")
    elif dotted_count >= 2:
        score += 2
        reasons.append("dotted_leaders>=2:+2")
    elif dotted_count == 1:
        score += 1
        reasons.append("dotted_leaders==1:+1")

    if trailing_page_number_count >= 8:
        score += 3
        reasons.append("trailing_page_numbers>=8:+3")
    elif trailing_page_number_count >= 4:
        score += 2
        reasons.append("trailing_page_numbers>=4:+2")
    elif trailing_page_number_count >= 2:
        score += 1
        reasons.append("trailing_page_numbers>=2:+1")

    if section_number_count >= 5:
        score += 2
        reasons.append("section_numbered_lines>=5:+2")
    elif section_number_count >= 2:
        score += 1
        reasons.append("section_numbered_lines>=2:+1")

    if eligible_count >= 8 and toc_like_ratio >= 0.50:
        score += 2
        reasons.append("toc_like_ratio>=0.50:+2")
    elif eligible_count >= 8 and toc_like_ratio >= 0.35:
        score += 1
        reasons.append("toc_like_ratio>=0.35:+1")

    if is_early_page:
        score += 1
        reasons.append("early_page:+1")
    elif not has_toc_title:
        score -= 2
        reasons.append("late_without_title:-2")

    is_toc_page = (
        score >= 6
        or (has_toc_title and toc_like_count >= 2)
        or (
            is_early_page
            and toc_like_count >= 8
            and toc_like_ratio >= 0.45
        )
    )
    toc_needs_manual_review = is_toc_page and (
        score <= 6
        or not has_toc_title
        or not is_early_page
        or toc_like_count < 5
    )
    sample_lines = [
        line
        for line in normalized_lines
        if TOC_TITLE_RE.search(line) or is_toc_like_line(line)
    ][:MAX_SAMPLE_LINES]

    return {
        "is_toc_page": is_toc_page,
        "toc_needs_manual_review": toc_needs_manual_review,
        "toc_score": score,
        "toc_has_title": has_toc_title,
        "toc_is_early_page": is_early_page,
        "toc_eligible_line_count": eligible_count,
        "toc_like_line_count": toc_like_count,
        "toc_like_ratio": round(toc_like_ratio, 4),
        "toc_dotted_count": dotted_count,
        "toc_trailing_page_number_count": trailing_page_number_count,
        "toc_section_number_count": section_number_count,
        "toc_reasons": reasons,
        "toc_sample_lines": sample_lines,
    }


def extract_page_lines_with_status(page: Any) -> tuple[list[str], str]:
    text_page = None
    try:
        text_page = page.get_textpage()
        text = text_page.get_text_range()
        lines = [
            normalize_toc_line(line)
            for line in text.splitlines()
            if normalize_toc_line(line)
        ]
        return lines, "ok"
    except Exception as error:
        return [], f"error:{type(error).__name__}:{error}"
    finally:
        if text_page is not None:
            text_page.close()


def extract_page_lines_with_pypdfium(page: Any) -> list[str]:
    lines, _ = extract_page_lines_with_status(page)
    return lines
