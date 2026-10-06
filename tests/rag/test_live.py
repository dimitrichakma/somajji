"""Live tests against the real Voyage and Pinecone services, on the ingested index.

They cost a handful of free Voyage tokens and a few Pinecone reads, so they run only on request:
    uv run pytest -m integration
They skip (not fail) when there are no keys or the index does not exist yet.
"""
import re
from pathlib import Path
from typing import get_args

import pytest

from app.config import get_settings
from app.rag.errors import RetrievalError
from app.rag.chunking import chunk_pages
from app.rag.pdf_loader import load_pages
from app.rag.retrieval import retrieve
from app.rag.store import PineconeStore
from app.rag.types import SourceName

pytestmark = pytest.mark.integration

SOURCES_DIR = Path(__file__).resolve().parents[2] / "data" / "sources"


@pytest.fixture(scope="module")
def store() -> PineconeStore:
    settings = get_settings()
    if not (settings.voyage_api_key and settings.pinecone_api_key):
        pytest.skip("VOYAGE_API_KEY and PINECONE_API_KEY are needed for the live tests")
    try:
        return PineconeStore(settings.pinecone_index)
    except RetrievalError as error:
        pytest.skip(str(error))


@pytest.fixture(autouse=True)
def _skip_unless_live_services_are_ready(store):
    """Every test in this file needs the keys and the index, so all of them skip together without them.
    (Without this, the tests that call retrieve() directly would fail instead of skipping.)"""


@pytest.mark.parametrize("source", get_args(SourceName))
def test_every_page_of_the_guide_is_in_the_index_and_nothing_else(store, source):
    pdf = SOURCES_DIR / f"{source}.pdf"
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not downloaded")
    ids = store.list_ids(source)
    pages_in_index = set()
    for vector_id in ids:
        match = re.fullmatch(rf"{source}-p(\d+)-c\d+", vector_id)
        assert match, f"unexpected id {vector_id}"
        pages_in_index.add(int(match.group(1)))
    assert pages_in_index == {page.page for page in load_pages(pdf)}


@pytest.mark.parametrize("source", get_args(SourceName))
def test_the_index_holds_exactly_the_chunks_ingest_builds_at_the_default_size(store, source):
    """Same ids, no more and no fewer: this is what 'no stale vectors, same count after a re-run' means live.
    It fails if an old chunk size was left behind, or if the index was loaded at a non-default size."""
    pdf = SOURCES_DIR / f"{source}.pdf"
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not downloaded")
    expected = {chunk.chunk_id for chunk in chunk_pages(source, load_pages(pdf))}
    stored = store.list_ids(source)
    assert len(stored) == len(set(stored)), "duplicate ids in the index"
    assert set(stored) == expected


@pytest.mark.parametrize("source", get_args(SourceName))
def test_every_stored_chunk_has_what_a_citation_needs(store, source):
    ids = store.list_ids(source)
    assert ids, f"nothing is stored for {source}"
    for vector_id, metadata in store.fetch(ids).items():
        assert metadata["source"] == source
        assert metadata["chunk_id"] == vector_id
        assert metadata["page"] >= 1 and metadata["pdf_page"] >= 1
        assert metadata["page_label_source"] in {"printed", "pdf_index"}
        assert metadata["text"].strip(), f"{vector_id} has no text"
        # The embedding prefix looks like "[pfa · section · p.12]". A page may start with "[" for other reasons.
        assert not metadata["text"].startswith(f"[{source} "), f"{vector_id} stored the embedding prefix"


def test_spec_test_3_the_three_action_principles_of_pfa_are_found():
    pfa = {p.page: p.text for p in load_pages(SOURCES_DIR / "pfa.pdf")}
    assert "look, listen and link" in pfa[18].lower().replace("\n", " ")  # the page really answers it
    results = retrieve("What are the three action principles of PFA?", k=5)
    assert any(chunk.source == "pfa" and chunk.page in (18, 19) for chunk in results)


def test_spec_test_4_managing_stress_in_pm_plus_is_found():
    # Spec: "a `pmplus` chunk in the top 5". To keep it meaningful, the page must really be about the topic:
    # chosen from the PDF text (it says "managing stress" or "slow breathing"), not from a hand-picked list.
    about_stress = {
        p.page
        for p in load_pages(SOURCES_DIR / "pmplus.pdf")
        if "managing stress" in p.text.lower() or "slow breathing" in p.text.lower()
    }
    assert {42, 43} <= about_stress  # chapter 6 itself is among them
    results = retrieve("How does Problem Management Plus teach people to manage stress?", k=5)
    assert any(chunk.source == "pmplus" and chunk.page in about_stress for chunk in results)


def test_retrieve_returns_at_most_k_sorted_citable_results():
    results = retrieve("A woman is very upset and cannot sleep after losing her home", k=5)
    assert 1 <= len(results) <= 5
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    assert all(r.page >= 1 and r.source and r.text.strip() for r in results)
