"""Split pages into chunks. A chunk never spans two pages, so it always has exactly one page to cite.

Sizes are given in tokens but measured in characters (about 4 characters per token for English).
There is no tokenizer here: the guides are English and the size only needs to be roughly right.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import Page
from .types import Chunk, SourceName

CHARS_PER_TOKEN = 4
MIN_PIECE_CHARS = 100  # a leftover shorter than this is folded into the previous chunk


def _merge_tiny(pieces: list[str]) -> list[str]:
    """Fold fragments under MIN_PIECE_CHARS into the previous piece, so no near-empty vectors are
    indexed. A chunk can therefore be up to MIN_PIECE_CHARS longer than the size limit."""
    merged: list[str] = []
    for piece in pieces:
        if merged and len(piece) < MIN_PIECE_CHARS:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def chunk_pages(
    source: SourceName,
    pages: list[Page],
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
) -> list[Chunk]:
    if chunk_tokens < 1 or overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"Need 0 <= overlap_tokens < chunk_tokens, got chunk_tokens={chunk_tokens}, "
            f"overlap_tokens={overlap_tokens}"
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_tokens * CHARS_PER_TOKEN,
        chunk_overlap=overlap_tokens * CHARS_PER_TOKEN,
    )

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    for page in pages:
        for index, text in enumerate(_merge_tiny(splitter.split_text(page.text))):
            chunk_id = f"{source}-p{page.page}-c{index}"
            if chunk_id in seen_ids:
                raise ValueError(f"Two chunks would get the id {chunk_id}: printed page {page.page} appears twice")
            seen_ids.add(chunk_id)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source=source,
                    page=page.page,
                    pdf_page=page.pdf_page,
                    page_label_source=page.page_label_source,
                    section=page.section,
                    text=text,
                )
            )
    return chunks
