from types import SimpleNamespace

import pytest

from app.rag import embeddings, ingest, store
from app.rag.chunking import CHARS_PER_TOKEN, chunk_pages
from app.rag.errors import RetrievalError
from app.rag.ingest import build_records, ingest_pages, ingest_source, main
from app.rag.pdf_loader import Page
from tests.rag.fakes import FakeEmbedder, InMemoryStore
from tests.rag.test_pdf_loader import body, build_pdf

NO_WAIT = {"sleep": lambda seconds: None}


def make_page(page: int, text: str, section: str = "") -> Page:
    return Page(pdf_page=page + 7, page=page, page_label_source="printed", section=section, text=text)


def long_text(marker: str, lines: int = 60) -> str:
    return "\n".join(f"{marker} line {i:03d} ".ljust(50, ".") for i in range(lines))


def two_pages() -> list[Page]:
    return [make_page(1, "Approach the person calmly.", "Look"), make_page(2, long_text("P2"), "Listen")]


def test_records_carry_clean_text_and_full_citation_metadata_and_embed_the_prefixed_text():
    embedder = FakeEmbedder()
    chunks = chunk_pages("pfa", two_pages(), chunk_tokens=200, overlap_tokens=25)
    records = build_records(chunks, embedder)

    assert len(records) == len(chunks)
    assert embedder.documents_seen == [chunk.embed_text() for chunk in chunks]
    for record, chunk in zip(records, chunks):
        metadata = record["metadata"]
        assert record["id"] == chunk.chunk_id
        assert metadata["chunk_id"] == chunk.chunk_id
        assert metadata["text"] == chunk.text and not metadata["text"].startswith("[")
        assert metadata["source"] == "pfa" and metadata["page"] == chunk.page
        assert metadata["pdf_page"] == chunk.pdf_page and metadata["page_label_source"] == "printed"
        assert "section" in metadata
        assert all(metadata[key] not in ("", None) for key in ("source", "page", "pdf_page", "text"))


def test_ingesting_twice_gives_the_same_ids():
    vector_store, embedder = InMemoryStore(), FakeEmbedder()
    ingest_pages("pfa", two_pages(), vector_store, embedder, 200, 25, **NO_WAIT)
    first = vector_store.list_ids("pfa")
    ingest_pages("pfa", two_pages(), vector_store, embedder, 200, 25, **NO_WAIT)
    assert vector_store.list_ids("pfa") == first
    assert len(first) > 2


def test_a_different_chunk_size_leaves_no_stale_vectors():
    vector_store, embedder = InMemoryStore(), FakeEmbedder()
    ingest_pages("pfa", two_pages(), vector_store, embedder, 200, 25, **NO_WAIT)
    old_ids = set(vector_store.list_ids("pfa"))
    ingest_pages("pfa", two_pages(), vector_store, embedder, 100, 10, **NO_WAIT)

    expected = {c.chunk_id for c in chunk_pages("pfa", two_pages(), chunk_tokens=100, overlap_tokens=10)}
    assert set(vector_store.list_ids("pfa")) == expected
    assert expected != old_ids  # the two runs really did make different chunks


def test_re_ingesting_one_source_leaves_the_others_alone():
    vector_store, embedder = InMemoryStore(), FakeEmbedder()
    ingest_pages("pfa", two_pages(), vector_store, embedder, 200, 25, **NO_WAIT)
    ingest_pages("mhgap", two_pages(), vector_store, embedder, 200, 25, **NO_WAIT)
    mhgap_before = vector_store.list_ids("mhgap")
    ingest_pages("pfa", two_pages(), vector_store, embedder, 100, 10, **NO_WAIT)
    assert vector_store.list_ids("mhgap") == mhgap_before
    assert vector_store.delete_calls == ["pfa", "mhgap", "pfa"]


def test_if_embedding_fails_the_old_vectors_stay_and_nothing_is_deleted():
    vector_store = InMemoryStore()
    ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(), 200, 25, **NO_WAIT)
    before = dict(vector_store.records)
    vector_store.delete_calls.clear()

    with pytest.raises(RetrievalError, match="Voyage is down"):
        ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(RetrievalError("Voyage is down")), 100, 10, **NO_WAIT)
    assert vector_store.records == before
    assert vector_store.delete_calls == []


def test_it_waits_for_a_store_that_lags_behind_its_writes():
    vector_store = InMemoryStore(lag=2)
    sleeps: list[float] = []
    report = ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(), 200, 25, sleep=sleeps.append)
    assert len(sleeps) == 2
    assert len(vector_store.list_ids("pfa")) == report.chunks


def test_it_gives_up_with_a_clear_error_if_the_count_never_matches():
    vector_store = InMemoryStore(lag=99)
    with pytest.raises(RetrievalError, match="vectors") as error:
        ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(), 200, 25, wait_attempts=3, **NO_WAIT)
    assert "pfa" in str(error.value)


def test_a_store_that_fails_to_delete_old_vectors_is_caught_by_the_count_check():
    class LeakyStore(InMemoryStore):
        def delete_source(self, source: str) -> None:
            pass  # pretends to delete but does not

    vector_store = LeakyStore()
    ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(), 100, 10, **NO_WAIT)
    # Bigger chunks now make fewer ids, so the old ones that are not replaced would be left behind.
    with pytest.raises(RetrievalError, match="vectors"):
        ingest_pages("pfa", two_pages(), vector_store, FakeEmbedder(), 400, 25, wait_attempts=2, **NO_WAIT)


def test_the_report_counts_pages_chunks_and_estimated_tokens():
    pages = two_pages()
    report = ingest_pages("pfa", pages, InMemoryStore(), FakeEmbedder(), 200, 25, **NO_WAIT)
    chunks = chunk_pages("pfa", pages, chunk_tokens=200, overlap_tokens=25)
    assert (report.source, report.pages, report.chunks) == ("pfa", 2, len(chunks))
    assert report.estimated_tokens == sum(len(c.embed_text()) for c in chunks) // CHARS_PER_TOKEN


def test_a_pdf_goes_through_ingest_source_end_to_end(tmp_path):
    path = build_pdf(tmp_path / "pfa.pdf", [[body(i), (810, str(10 + i), 9)] for i in range(3)])
    vector_store = InMemoryStore()
    report = ingest_source("pfa", path, vector_store, FakeEmbedder(), **NO_WAIT)
    assert (report.pages, report.chunks) == (3, 3)
    assert vector_store.list_ids("pfa") == ["pfa-p10-c0", "pfa-p11-c0", "pfa-p12-c0"]


# --- the command line -------------------------------------------------------------------------------


@pytest.fixture
def no_keys(monkeypatch):
    """Settings with empty keys, so the real .env is never read."""
    fake_settings = lambda: SimpleNamespace(voyage_api_key="", pinecone_api_key="", pinecone_index="somajji-who", voyage_model="voyage-4")
    for module in (embeddings, store, ingest):
        monkeypatch.setattr(module, "get_settings", fake_settings)


def test_missing_keys_stop_the_run_before_any_pdf_is_read(no_keys, monkeypatch, capsys, tmp_path):
    def must_not_run(path):
        raise AssertionError("a PDF was read before the keys were checked")

    monkeypatch.setattr(ingest, "load_pages", must_not_run)
    assert main(["--sources-dir", str(tmp_path)]) == 1
    assert "VOYAGE_API_KEY" in capsys.readouterr().err


def test_a_missing_pinecone_key_is_reported_too(no_keys, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        embeddings, "get_settings", lambda: SimpleNamespace(voyage_api_key="voyage-key", voyage_model="voyage-4")
    )
    monkeypatch.setattr(embeddings.voyageai, "Client", lambda **kwargs: object())
    assert main(["--sources-dir", str(tmp_path)]) == 1
    assert "PINECONE_API_KEY" in capsys.readouterr().err


def test_dry_run_needs_no_keys_and_prints_chunks_and_tokens(no_keys, capsys, tmp_path):
    build_pdf(tmp_path / "pfa.pdf", [[body(i), (810, str(10 + i), 9)] for i in range(3)])
    assert main(["--dry-run", "--source", "pfa", "--sources-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "pfa: 3 pages -> 3 chunks" in output
    assert "tokens" in output and "dry run" in output.lower()


def test_a_full_run_wires_the_embedder_the_store_and_the_batch_pause(monkeypatch, capsys, tmp_path):
    build_pdf(tmp_path / "pfa.pdf", [[body(i), (810, str(10 + i), 9)] for i in range(3)])
    vector_store, built = InMemoryStore(), {}

    def fake_embedder(**kwargs):
        built.update(kwargs)
        return FakeEmbedder()

    monkeypatch.setattr(ingest, "VoyageEmbedder", fake_embedder)
    monkeypatch.setattr(ingest, "PineconeStore", lambda index_name: vector_store)
    monkeypatch.setattr(ingest, "get_settings", lambda: SimpleNamespace(pinecone_index="somajji-who"))
    monkeypatch.setattr(ingest.time, "sleep", lambda seconds: None)  # the wait for the store to catch up

    assert main(["--source", "pfa", "--batch-pause", "25", "--sources-dir", str(tmp_path)]) == 0
    assert built == {"batch_pause_seconds": 25.0}
    assert vector_store.list_ids("pfa") == ["pfa-p10-c0", "pfa-p11-c0", "pfa-p12-c0"]
    assert "embedded and stored" in capsys.readouterr().out


def test_a_pasted_long_dash_works_in_the_ingest_command_too(no_keys, capsys, tmp_path):
    build_pdf(tmp_path / "pfa.pdf", [[body(i), (810, str(10 + i), 9)] for i in range(3)])
    assert main(["\u2014dry-run", "\u2014source", "pfa", "\u2014sources-dir", str(tmp_path)]) == 0
    assert "pfa: 3 pages -> 3 chunks" in capsys.readouterr().out


def test_dry_run_with_a_missing_pdf_says_how_to_download_it(no_keys, capsys, tmp_path):
    assert main(["--dry-run", "--source", "pfa", "--sources-dir", str(tmp_path)]) == 1
    assert "download_sources" in capsys.readouterr().err
