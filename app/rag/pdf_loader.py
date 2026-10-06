"""Read a WHO guide PDF into clean pages, each with the page number printed on it.

Printed page numbers come from three places, in this order:
  1. the PDF's own page label, when it is a plain number (pfa, pmplus)
  2. a bare number in the top or bottom margin (mhgap)
  3. nothing readable anywhere in the document: use the position in the PDF and say so (`page_label_source`)
Pages with roman numerals or letters are front matter and are skipped: they can't be cited.
When the rest of the document is numbered consistently, a page with no number (cover, back cover) is
skipped too, because a made-up number could clash with a real page.
"""
import logging
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf
from pydantic import BaseModel, PositiveInt

from .errors import RetrievalError
from .types import NonEmptyText

logger = logging.getLogger(__name__)

MARGIN_FRACTION = 0.10  # top and bottom 10% of the page hold headers, footers and page numbers
MIN_REPEATS = 3  # a margin line seen on this many pages is a running header or footer
MIN_BODY_CHARS = 40  # fewer characters than this left on a page means blank or cover
HEADING_SIZE_RATIO = 1.2  # a heading is at least this much bigger than the usual text
MAX_HEADING_CHARS = 120
DOMINANT_SHARE = 0.8  # this share of numbered pages must agree on (pdf page - printed page)

_BARE_NUMBER = re.compile(r"\d{1,3}")
_ROMAN = re.compile(r"(xl|l?x{0,3})(ix|iv|v?i{0,3})", re.IGNORECASE)


class Page(BaseModel):
    pdf_page: PositiveInt  # position in the PDF file, starting at 1
    page: PositiveInt  # printed page number, what a citation shows
    page_label_source: Literal["printed", "pdf_index"]
    section: str = ""  # nearest heading at or before this page; may be empty
    text: NonEmptyText


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    in_margin: bool


def _read_lines(page: pymupdf.Page) -> list[_Line]:
    height = page.rect.height
    lines = []
    # Keep the PDF's own block order: sorting top to bottom would interleave two-column pages.
    for block in page.get_text("dict")["blocks"]:
        if block["type"] != 0:  # 0 = text; skip images
            continue
        for line in block["lines"]:
            raw = "".join(span["text"] for span in line["spans"])
            text = " ".join(unicodedata.normalize("NFKC", raw).split())  # NFKC turns the "ﬁ" glyph into "fi"
            if not text:
                continue
            top, bottom = line["bbox"][1], line["bbox"][3]
            in_margin = bottom < height * MARGIN_FRACTION or top > height * (1 - MARGIN_FRACTION)
            lines.append(_Line(text, max(span["size"] for span in line["spans"]), in_margin))
    return lines


def _pattern(text: str) -> str:
    """'Chapter 3' and 'Chapter 4' are the same running footer."""
    return re.sub(r"\d+", "#", text)


def _is_roman(text: str) -> bool:
    return bool(text) and _ROMAN.fullmatch(text) is not None


def _repeated_margin_patterns(all_lines: list[list[_Line]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in all_lines:
        counts.update({_pattern(line.text) for line in lines if line.in_margin})
    pages_with_text = sum(1 for lines in all_lines if lines)
    needed = max(2, min(MIN_REPEATS, pages_with_text))
    return {pattern for pattern, count in counts.items() if count >= needed}


def _body_font_size(all_lines: list[list[_Line]]) -> float:
    """The font size that most of the text uses, counted by characters."""
    sizes: Counter[float] = Counter()
    for lines in all_lines:
        for line in lines:
            if not line.in_margin:
                sizes[round(line.size, 1)] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _candidates(label: str, lines: list[_Line]) -> list[int]:
    """Possible printed numbers for a page: its page label, else the bare numbers in its margins."""
    if label.isdigit():
        return [int(label)] if int(label) >= 1 else []
    if label:  # roman numeral or letter label: front matter, no printed number
        return []
    numbers = {int(line.text) for line in lines if line.in_margin and _BARE_NUMBER.fullmatch(line.text)}
    return sorted(number for number in numbers if number >= 1)


def _dominant_offset(candidates: list[list[int]]) -> int | None:
    """The usual (pdf page - printed page), if nearly all unambiguous pages agree on one value."""
    offsets = Counter(index + 1 - found[0] for index, found in enumerate(candidates) if len(found) == 1)
    if not offsets:
        return None
    offset, count = offsets.most_common(1)[0]
    return offset if count >= DOMINANT_SHARE * sum(offsets.values()) else None


def _choose_number(pdf_page: int, candidates: list[int], offset: int | None) -> int | None:
    """One printed number, or None. With several margin numbers (e.g. a flowchart step number
    next to the page number), keep the one that fits the document's usual offset."""
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and offset is not None:
        fitting = [number for number in candidates if pdf_page - number == offset]
        if len(fitting) == 1:
            return fitting[0]
    return None


def _is_page_furniture(line: _Line, repeated: set[str]) -> bool:
    """Margin lines that are not content: repeated headers and footers, page numbers."""
    if not line.in_margin:
        return False
    return (
        _pattern(line.text) in repeated
        or _BARE_NUMBER.fullmatch(line.text) is not None
        or _is_roman(line.text)
    )


def _first_heading(lines: list[_Line], body_size: float) -> str:
    """The first run of consecutive large-font lines on the page, joined into one heading."""
    heading: list[str] = []
    for line in lines:
        is_heading = (
            body_size > 0
            and line.size >= body_size * HEADING_SIZE_RATIO
            and len(line.text) <= MAX_HEADING_CHARS
        )
        if is_heading:
            heading.append(line.text)
        elif heading:
            break
    joined = " ".join(heading)
    return joined if len(joined) <= MAX_HEADING_CHARS else ""  # a long run is a paragraph, not a heading


def _open(path: Path) -> pymupdf.Document:
    if not path.exists():
        raise RetrievalError(f"{path} not found. Run: uv run python scripts/download_sources.py")
    try:
        return pymupdf.open(path)
    except Exception as error:  # pymupdf raises several error types for bad files
        raise RetrievalError(f"Could not read {path.name} as a PDF: {error}") from error


def load_pages(path: Path) -> list[Page]:
    """Return the citable pages of one PDF, in order. Skipped pages are logged with the reason."""
    with _open(path) as document:
        labels = [page.get_label() for page in document]
        all_lines = [_read_lines(page) for page in document]

    repeated = _repeated_margin_patterns(all_lines)
    body_size = _body_font_size(all_lines)
    candidates = [_candidates(label, lines) for label, lines in zip(labels, all_lines)]
    offset = _dominant_offset(candidates)

    pages: list[Page] = []
    section = ""
    for index, (label, lines) in enumerate(zip(labels, all_lines)):
        pdf_page = index + 1
        printed = _choose_number(pdf_page, candidates[index], offset)
        content = [line for line in lines if not _is_page_furniture(line, repeated)]

        skip_reason = None
        if not lines:
            skip_reason = "no text (blank or scanned)"
        elif label and not label.isdigit():
            skip_reason = f"front matter (label {label!r})"
        elif printed is None and any(line.in_margin and _is_roman(line.text) for line in lines):
            skip_reason = "front matter (roman numeral)"
        elif printed is None and offset is not None:
            skip_reason = "no printed page number (the rest of the document has one)"
        elif sum(len(line.text) for line in content) < MIN_BODY_CHARS:
            skip_reason = "almost no text (blank or cover)"
        if skip_reason:
            logger.warning("%s: skipped pdf page %d: %s", path.name, pdf_page, skip_reason)
            continue

        section = _first_heading(content, body_size) or section
        pages.append(
            Page(
                pdf_page=pdf_page,
                page=printed if printed is not None else pdf_page,
                page_label_source="printed" if printed is not None else "pdf_index",
                section=section,
                text="\n".join(line.text for line in content),
            )
        )
    return pages
