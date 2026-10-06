"""Load the WHO guides into the vector store.

Run: uv run python -m app.rag.ingest            (needs VOYAGE_API_KEY, PINECONE_API_KEY and an existing index)
     uv run python -m app.rag.ingest --dry-run  (no keys, no network: only counts chunks and tokens)

It is safe to run again: each source's old vectors are replaced, so nothing is duplicated or left stale.
"""
import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_args

from app.config import get_settings

from .chunking import CHARS_PER_TOKEN, chunk_pages
from .cli import normalize_dashes
from .embeddings import Embedder, VoyageEmbedder
from .errors import RetrievalError
from .pdf_loader import Page, load_pages
from .store import PineconeStore, VectorRecord, VectorStore
from .types import Chunk, SourceName

SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"
WAIT_ATTEMPTS = 30
WAIT_INTERVAL_SECONDS = 1.0


@dataclass(frozen=True)
class IngestReport:
    source: str
    pages: int
    chunks: int
    estimated_tokens: int  # what Voyage will be asked to embed, at about 4 characters per token


def build_records(chunks: list[Chunk], embedder: Embedder) -> list[VectorRecord]:
    # Embed the prefixed text, so a chunk from the middle of a guide still carries where it came from.
    # Store the clean text, which is what the guide agent will quote.
    vectors = embedder.embed_documents([chunk.embed_text() for chunk in chunks])
    return [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page,
                "pdf_page": chunk.pdf_page,
                "page_label_source": chunk.page_label_source,
                "section": chunk.section,
                "text": chunk.text,
            },
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]


def _estimated_tokens(chunks: list[Chunk]) -> int:
    return sum(len(chunk.embed_text()) for chunk in chunks) // CHARS_PER_TOKEN


def _wait_for_count(
    store: VectorStore,
    source: str,
    expected: int,
    attempts: int,
    interval: float,
    sleep: Callable[[float], None],
) -> None:
    """Pinecone shows writes a moment late, so check until the count matches. A count that never
    matches means a failed delete (stale vectors) or a failed upsert."""
    found = 0
    for _ in range(attempts):
        found = len(store.list_ids(source))
        if found == expected:
            return
        sleep(interval)
    raise RetrievalError(
        f"The store shows {found} vectors for '{source}' but {expected} were stored. "
        "Wait a minute and run the ingest again."
    )


def ingest_pages(
    source: SourceName,
    pages: list[Page],
    store: VectorStore,
    embedder: Embedder,
    chunk_tokens: int = 800,
    overlap_tokens: int = 100,
    wait_attempts: int = WAIT_ATTEMPTS,
    wait_interval: float = WAIT_INTERVAL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> IngestReport:
    chunks = chunk_pages(source, pages, chunk_tokens, overlap_tokens)
    # Embed first: if Voyage fails, the vectors already in the store are left untouched.
    records = build_records(chunks, embedder)
    store.delete_source(source)
    store.upsert(records)
    _wait_for_count(store, source, len(chunks), wait_attempts, wait_interval, sleep)
    return IngestReport(source, len(pages), len(chunks), _estimated_tokens(chunks))


def ingest_source(source: SourceName, pdf_path: Path, store: VectorStore, embedder: Embedder, **options) -> IngestReport:
    return ingest_pages(source, load_pages(pdf_path), store, embedder, **options)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the WHO guides into the vector store.")
    parser.add_argument("--source", choices=get_args(SourceName), help="only this guide (default: all three)")
    parser.add_argument("--chunk-tokens", type=int, default=800)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    parser.add_argument("--sources-dir", type=Path, default=SOURCES_DIR)
    parser.add_argument(
        "--batch-pause", type=float, default=0.0, help="seconds to wait between Voyage batches (free accounts: try 25)"
    )
    parser.add_argument("--dry-run", action="store_true", help="count chunks and tokens only; no keys, no network")
    return parser.parse_args(normalize_dashes(sys.argv[1:] if argv is None else argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sources: list[SourceName] = [args.source] if args.source else list(get_args(SourceName))
    sizes = {"chunk_tokens": args.chunk_tokens, "overlap_tokens": args.overlap_tokens}
    reports: list[IngestReport] = []
    try:
        # Check the keys and the index before reading any PDF or sending anything.
        if not args.dry_run:
            embedder = VoyageEmbedder(batch_pause_seconds=args.batch_pause)
            store = PineconeStore(get_settings().pinecone_index)
        for source in sources:
            pages = load_pages(args.sources_dir / f"{source}.pdf")
            if args.dry_run:
                chunks = chunk_pages(source, pages, **sizes)
                report = IngestReport(source, len(pages), len(chunks), _estimated_tokens(chunks))
            else:
                report = ingest_pages(source, pages, store, embedder, **sizes)
            reports.append(report)
            print(f"{report.source}: {report.pages} pages -> {report.chunks} chunks, ~{report.estimated_tokens:,} tokens")
    except (RetrievalError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    total = sum(report.estimated_tokens for report in reports)
    print(f"Total: ~{total:,} tokens{' (dry run: nothing was sent)' if args.dry_run else ' embedded and stored'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
