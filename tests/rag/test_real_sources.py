"""Smoke test against the real WHO PDFs. Skipped when they have not been downloaded."""
from pathlib import Path

import pytest

from app.rag.pdf_loader import load_pages

SOURCES = Path(__file__).resolve().parents[2] / "data" / "sources"

# name -> (expected page count range, a running header that must be gone, or None)
EXPECTED = {
    "pfa": ((48, 56), "Psychological first aid: Guide for field workers"),
    "mhgap": ((150, 158), None),
    "pmplus": ((124, 132), "PM+: INDIVIDUAL PSYCHOLOGICAL HELP FOR ADULTS IMPAIRED BY DISTRESS"),
}


@pytest.mark.parametrize("name", EXPECTED)
def test_real_guide_loads_into_citable_pages(name):
    path = SOURCES / f"{name}.pdf"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded (run scripts/download_sources.py)")
    (low, high), running_header = EXPECTED[name]
    pages = load_pages(path)

    assert low <= len(pages) <= high
    assert all(p.text.strip() for p in pages)
    # Every page was numbered from the page itself, and no two pages share a number.
    assert all(p.page_label_source == "printed" for p in pages)
    assert len({p.page for p in pages}) == len(pages)
    # Each guide has one fixed gap between PDF position and printed number. A mix would mean misread pages.
    assert len({p.pdf_page - p.page for p in pages}) == 1
    # The page number itself must not be left behind as the first line of text.
    assert not any(p.text.splitlines()[0].strip() == str(p.page) for p in pages)
    if running_header:
        assert not any(running_header in p.text for p in pages)
    # Typographic ligatures (one glyph for "fi", "fl") must be split into plain letters for search.
    assert not any(ch in p.text for p in pages for ch in "\ufb00\ufb01\ufb02\ufb03\ufb04")


@pytest.mark.parametrize("name", EXPECTED)
def test_real_guide_chunks_are_valid_and_unique(name, capsys):
    from app.rag.chunking import chunk_pages

    path = SOURCES / f"{name}.pdf"
    if not path.exists():
        pytest.skip(f"{path.name} not downloaded (run scripts/download_sources.py)")
    pages = load_pages(path)
    chunks = chunk_pages(name, pages)

    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)
    assert len(chunks) >= len(pages)  # every page yields at least one chunk
    assert {c.page for c in chunks} == {p.page for p in pages}
    with capsys.disabled():
        print(f"\n{name}: {len(pages)} pages -> {len(chunks)} chunks at 800 tokens")
