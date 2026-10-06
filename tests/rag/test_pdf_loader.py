import logging
from pathlib import Path

import pymupdf
import pytest

from app.rag.errors import RetrievalError
from app.rag.pdf_loader import load_pages

HEADER = (30, "WHO Test Guide header line", 9)  # top margin
FOOTER = (810, "Test guide footer text", 9)  # bottom margin


def body(n: int) -> tuple[float, str, float]:
    return (300, f"This is the body text of test page {n}, long enough to be kept.", 11)


def build_pdf(path: Path, pages: list[list[tuple[float, str, float]]], labels: list[dict] | None = None) -> Path:
    """Each page is a list of (y, text, font size). A4-sized, so margins are y < 84 and y > 758."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        for y, text, size in lines:
            page.insert_text((72, y), text, fontsize=size)
    if labels:
        doc.set_page_labels(labels)
    doc.save(path)
    doc.close()
    return path


def test_digit_page_label_becomes_the_printed_page(tmp_path):
    path = build_pdf(
        tmp_path / "a.pdf",
        [[body(i)] for i in range(3)],
        labels=[{"startpage": 0, "style": "D", "firstpagenum": 5}],
    )
    pages = load_pages(path)
    assert [(p.pdf_page, p.page) for p in pages] == [(1, 5), (2, 6), (3, 7)]
    assert all(p.page_label_source == "printed" for p in pages)


def test_bare_number_in_the_bottom_margin_is_the_printed_page(tmp_path):
    path = build_pdf(tmp_path / "a.pdf", [[body(i), (810, str(10 + i), 9)] for i in range(3)])
    pages = load_pages(path)
    assert [p.page for p in pages] == [10, 11, 12]
    assert all(p.page_label_source == "printed" for p in pages)
    assert all(line.strip() not in {"10", "11", "12"} for p in pages for line in p.text.splitlines())


def test_falls_back_to_the_pdf_index_when_no_number_can_be_read(tmp_path):
    pages = load_pages(build_pdf(tmp_path / "a.pdf", [[body(i)] for i in range(3)]))
    assert [(p.pdf_page, p.page) for p in pages] == [(1, 1), (2, 2), (3, 3)]
    assert all(p.page_label_source == "pdf_index" for p in pages)


def test_several_margin_numbers_are_resolved_by_the_documents_usual_offset(tmp_path):
    # Page 3 also has a flowchart step number "3" in the margin; the real page number is 11.
    path = build_pdf(
        tmp_path / "a.pdf",
        [
            [body(0), (810, "9", 9)],
            [body(1), (810, "10", 9)],
            [body(2), (810, "11", 9), (790, "3", 9)],
            [body(3), (810, "12", 9)],
        ],
    )
    assert [p.page for p in load_pages(path)] == [9, 10, 11, 12]


def test_an_unnumbered_page_is_skipped_when_the_rest_of_the_document_is_numbered(tmp_path, caplog):
    path = build_pdf(
        tmp_path / "a.pdf",
        [[body(0)], [body(1), (810, "1", 9)], [body(2), (810, "2", 9)], [body(3), (810, "3", 9)]],
    )
    with caplog.at_level(logging.WARNING):
        pages = load_pages(path)
    assert [(p.pdf_page, p.page) for p in pages] == [(2, 1), (3, 2), (4, 3)]
    assert "pdf page 1" in caplog.text  # skipped for having no printed number, so no clash with page 1


def test_roman_and_letter_labels_are_front_matter_and_skipped(tmp_path):
    path = build_pdf(
        tmp_path / "a.pdf",
        [[body(i)] for i in range(4)],
        labels=[
            {"startpage": 0, "style": "A", "firstpagenum": 1},  # page 1 has label "A"
            {"startpage": 1, "style": "r", "firstpagenum": 1},  # page 2 has label "i"
            {"startpage": 2, "style": "D", "firstpagenum": 1},  # pages 3, 4 have labels 1, 2
        ],
    )
    assert [(p.pdf_page, p.page) for p in load_pages(path)] == [(3, 1), (4, 2)]


def test_roman_numeral_in_the_margin_is_front_matter_and_skipped(tmp_path):
    path = build_pdf(
        tmp_path / "a.pdf",
        [[body(0), (810, "iv", 9)], [body(1), (810, "1", 9)], [body(2), (810, "2", 9)]],
    )
    assert [(p.pdf_page, p.page) for p in load_pages(path)] == [(2, 1), (3, 2)]


def test_repeated_headers_and_footers_are_removed_but_repeated_body_text_is_kept(tmp_path):
    repeated_body = (400, "A sentence that appears in the body of every page.", 11)
    path = build_pdf(
        tmp_path / "a.pdf",
        [[HEADER, FOOTER, (820, f"Chapter {i}", 9), body(i), repeated_body] for i in range(1, 4)],
    )
    pages = load_pages(path)
    assert len(pages) == 3
    for page in pages:
        assert "header line" not in page.text
        assert "footer text" not in page.text
        assert "Chapter" not in page.text  # same pattern once digits are ignored
        assert "appears in the body of every page" in page.text
        assert "body text of test page" in page.text


def test_blank_and_near_empty_pages_are_skipped_and_logged(tmp_path, caplog):
    path = build_pdf(tmp_path / "a.pdf", [[body(0)], [], [(300, "Cover", 11)], [body(3)]])
    with caplog.at_level(logging.WARNING):
        pages = load_pages(path)
    assert [p.pdf_page for p in pages] == [1, 4]
    assert "pdf page 2" in caplog.text
    assert "pdf page 3" in caplog.text


def test_a_larger_heading_becomes_the_section_and_carries_onto_the_next_page(tmp_path):
    heading_one = (120, "Chapter One: Staying calm", 20)
    heading_two = (120, "Chapter Two: Listening", 20)
    path = build_pdf(
        tmp_path / "a.pdf",
        [[heading_one, body(1)], [body(2)], [heading_two, body(3)]],
    )
    assert [p.section for p in load_pages(path)] == [
        "Chapter One: Staying calm",
        "Chapter One: Staying calm",
        "Chapter Two: Listening",
    ]


def test_several_big_lines_that_add_up_to_a_paragraph_are_not_a_section(tmp_path):
    big_lines = [(120 + 30 * i, "A short line of big text number %d, then more words." % i, 20) for i in range(4)]
    body_lines = [(350 + 20 * i, "Ordinary body text, many words, repeated to make this the usual size. " * 2, 11) for i in range(5)]
    pages = load_pages(build_pdf(tmp_path / "a.pdf", [[*big_lines, *body_lines]]))
    assert pages[0].section == ""


def test_pages_before_any_heading_have_an_empty_section(tmp_path):
    pages = load_pages(build_pdf(tmp_path / "a.pdf", [[body(1)]]))
    assert pages[0].section == ""


def test_a_missing_file_raises_retrieval_error_with_the_fix(tmp_path):
    with pytest.raises(RetrievalError, match="download_sources"):
        load_pages(tmp_path / "missing.pdf")


def test_a_file_that_is_not_a_pdf_raises_retrieval_error(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_text("this is not a pdf")
    with pytest.raises(RetrievalError):
        load_pages(path)
