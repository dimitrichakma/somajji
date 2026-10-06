import pytest

from app.rag.chunking import MIN_PIECE_CHARS, chunk_pages
from app.rag.pdf_loader import Page

CHARS_PER_TOKEN = 4


def make_page(page: int, text: str, section: str = "", pdf_page: int | None = None) -> Page:
    return Page(
        pdf_page=pdf_page or page + 7,
        page=page,
        page_label_source="printed",
        section=section,
        text=text,
    )


def long_text(marker: str, lines: int = 100) -> str:
    """Numbered 50 character lines, so a chunk boundary and its overlap are easy to see."""
    return "\n".join(f"{marker} line {i:03d} ".ljust(50, ".") for i in range(lines))


def test_a_short_page_is_one_chunk_with_every_field_copied():
    page = make_page(12, "Approach the person calmly and listen.", section="Look, listen, link")
    [chunk] = chunk_pages("pfa", [page])
    assert chunk.chunk_id == "pfa-p12-c0"
    assert (chunk.source, chunk.page, chunk.pdf_page) == ("pfa", 12, 19)
    assert (chunk.page_label_source, chunk.section) == ("printed", "Look, listen, link")
    assert chunk.text == "Approach the person calmly and listen."


def test_chunks_never_mix_text_from_different_pages():
    pages = [
        make_page(1, "PAGE1 is a short page with some words on it."),
        make_page(2, "PAGE2 is another short page with some words."),
        make_page(3, long_text("PAGE3")),
    ]
    chunks = chunk_pages("pfa", pages, chunk_tokens=200, overlap_tokens=25)
    assert len(chunks) > 3
    for chunk in chunks:
        markers = {name for name in ("PAGE1", "PAGE2", "PAGE3") if name in chunk.text}
        assert markers == {f"PAGE{chunk.page}"}


def test_a_long_page_splits_into_ordered_chunks_within_the_size_limit():
    chunks = chunk_pages("mhgap", [make_page(5, long_text("X"))], chunk_tokens=200, overlap_tokens=25)
    assert len(chunks) > 1
    assert [c.chunk_id for c in chunks] == [f"mhgap-p5-c{i}" for i in range(len(chunks))]
    assert all(len(c.text) <= 200 * CHARS_PER_TOKEN for c in chunks)


def test_consecutive_chunks_of_a_long_page_overlap():
    chunks = chunk_pages("pfa", [make_page(5, long_text("X"))], chunk_tokens=200, overlap_tokens=25)
    for earlier, later in zip(chunks, chunks[1:]):
        last_line_of_earlier = earlier.text.splitlines()[-1]
        assert last_line_of_earlier in later.text


def test_a_smaller_chunk_size_gives_more_chunks():
    pages = [make_page(5, long_text("X"))]
    big = chunk_pages("pfa", pages, chunk_tokens=400, overlap_tokens=25)
    small = chunk_pages("pfa", pages, chunk_tokens=200, overlap_tokens=25)
    assert len(small) > len(big)


@pytest.mark.parametrize("chunk_tokens, overlap_tokens", [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_impossible_sizes_are_rejected(chunk_tokens, overlap_tokens):
    with pytest.raises(ValueError):
        chunk_pages("pfa", [make_page(1, "Some text on a page.")], chunk_tokens, overlap_tokens)


def test_ids_are_stable_between_runs_and_unique():
    pages = [make_page(n, long_text(f"P{n}", lines=30)) for n in (3, 4, 5)]
    first = chunk_pages("pfa", pages, chunk_tokens=100, overlap_tokens=10)
    second = chunk_pages("pfa", pages, chunk_tokens=100, overlap_tokens=10)
    ids = [c.chunk_id for c in first]
    assert ids == [c.chunk_id for c in second]
    assert len(set(ids)) == len(ids)


def test_a_repeated_printed_page_is_an_error():
    pages = [make_page(5, "First page five text.", pdf_page=12), make_page(5, "Second page five text.", pdf_page=13)]
    with pytest.raises(ValueError, match="pfa-p5-c0"):
        chunk_pages("pfa", pages)


def test_a_tiny_leftover_is_folded_into_the_previous_chunk():
    # Six 60 character lines fill one 400 character chunk; the 50 character line left over is tiny.
    text = "\n".join(["a" * 60] * 6 + ["LAST LINE " + "b" * 40])
    chunks = chunk_pages("pfa", [make_page(1, text)], chunk_tokens=100, overlap_tokens=0)
    assert len(chunks) == 1
    assert "LAST LINE" in chunks[0].text


def test_no_chunk_of_a_multi_chunk_page_is_tiny():
    chunks = chunk_pages("pfa", [make_page(5, long_text("X", lines=37))], chunk_tokens=200, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(len(c.text) >= MIN_PIECE_CHARS for c in chunks)


def test_stored_text_has_no_prefix_but_embed_text_does():
    [chunk] = chunk_pages("pmplus", [make_page(30, "Slow breathing helps.", section="Managing stress")])
    assert not chunk.text.startswith("[")
    assert chunk.embed_text() == "[pmplus · Managing stress · p.30] Slow breathing helps."


def test_a_page_without_a_section_still_works():
    [chunk] = chunk_pages("pfa", [make_page(1, "Plain text with no heading above it.")])
    assert chunk.section == ""
    assert chunk.embed_text().startswith("[pfa · p.1]")


def test_no_pages_means_no_chunks():
    assert chunk_pages("pfa", []) == []
