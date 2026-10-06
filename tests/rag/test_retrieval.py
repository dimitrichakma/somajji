import logging

import pytest

from app.rag import retrieval
from app.rag.errors import RetrievalError
from app.rag.retrieval import retrieve
from app.rag.types import RetrievedChunk
from tests.rag.fakes import FakeEmbedder, FakeReranker, InMemoryStore


class QueryEmbedder(FakeEmbedder):
    """Every question becomes the vector [1, 0, 0, 0]."""

    def embed_query(self, text: str) -> list[float]:
        self.queries_seen.append(text)
        return [1.0, 0.0, 0.0, 0.0]


class Untouchable:
    """Fails the test if anything on it is used."""

    def __getattr__(self, name):
        raise AssertionError(f"{name} was used but should not have been")


class FailingStore:
    def query(self, vector, k):
        raise RetrievalError("Pinecone query failed: down")


def make_record(vector_id: str, vector: list[float], **metadata) -> dict:
    fields = {
        "chunk_id": vector_id,
        "source": "pfa",
        "page": 12.0,  # Pinecone hands numbers back as floats
        "pdf_page": 19.0,
        "page_label_source": "printed",
        "section": "Look, listen, link",
        "text": f"Text of {vector_id}.",
    }
    fields.update(metadata)
    return {"id": vector_id, "values": vector, "metadata": fields}


def make_store(*records: dict) -> InMemoryStore:
    store = InMemoryStore()
    store.upsert(list(records))
    return store


GRADED = [
    make_record("pfa-p1-c0", [1, 0, 0, 0], page=1.0),  # cosine 1.0
    make_record("pfa-p2-c0", [1, 1, 0, 0], page=2.0),  # about 0.71
    make_record("pfa-p3-c0", [0, 1, 0, 0], page=3.0),  # 0.0
    make_record("pfa-p4-c0", [-1, 0, 0, 0], page=4.0),  # -1.0
]


def test_a_result_carries_clean_text_the_citation_and_the_score():
    embedder = QueryEmbedder()
    [chunk] = retrieve("How do I help someone in shock?", k=5, embedder=embedder, store=make_store(GRADED[0]))
    assert chunk == RetrievedChunk(
        text="Text of pfa-p1-c0.", source="pfa", page=1, pdf_page=19, section="Look, listen, link", score=1.0
    )
    assert embedder.queries_seen == ["How do I help someone in shock?"]


def test_it_returns_at_most_k_results():
    store = make_store(*GRADED)
    assert len(retrieve("question", k=3, embedder=QueryEmbedder(), store=store)) == 3
    assert len(retrieve("question", k=10, embedder=QueryEmbedder(), store=store)) == 4


def test_results_are_sorted_by_score_highest_first():
    chunks = retrieve("question", k=4, embedder=QueryEmbedder(), store=make_store(*reversed(GRADED)))
    assert [c.page for c in chunks] == [1, 2, 3, 4]
    assert [c.score for c in chunks] == sorted((c.score for c in chunks), reverse=True)


def test_it_sorts_and_cuts_even_if_the_store_answers_out_of_order_and_too_long():
    class SloppyStore:
        def query(self, vector, k):
            return [
                {"id": "b", "score": 0.2, "metadata": make_record("b", [1], page=2.0)["metadata"]},
                {"id": "a", "score": 0.9, "metadata": make_record("a", [1], page=1.0)["metadata"]},
                {"id": "c", "score": 0.5, "metadata": make_record("c", [1], page=3.0)["metadata"]},
            ]

    chunks = retrieve("question", k=2, embedder=QueryEmbedder(), store=SloppyStore())
    assert [c.page for c in chunks] == [1, 3]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_an_empty_question_is_rejected_before_anything_is_touched(question, monkeypatch):
    def must_not_build():
        raise AssertionError("a default client was built")

    monkeypatch.setattr(retrieval, "_default_embedder", must_not_build)
    monkeypatch.setattr(retrieval, "_default_store", must_not_build)
    with pytest.raises(RetrievalError, match="empty"):
        retrieve(question)
    with pytest.raises(RetrievalError, match="empty"):
        retrieve(question, embedder=Untouchable(), store=Untouchable())


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_at_least_one(k):
    with pytest.raises(ValueError):
        retrieve("question", k=k, embedder=QueryEmbedder(), store=make_store(*GRADED))


def test_an_embedding_error_stops_the_call_before_the_store_is_queried():
    embedder = FakeEmbedder(raises=RetrievalError("Voyage embedding failed: down"))
    with pytest.raises(RetrievalError, match="Voyage embedding failed"):
        retrieve("question", embedder=embedder, store=Untouchable())


def test_a_store_error_reaches_the_caller_as_the_same_retrieval_error():
    with pytest.raises(RetrievalError, match="Pinecone query failed: down"):
        retrieve("question", embedder=QueryEmbedder(), store=FailingStore())


@pytest.mark.parametrize(
    "broken",
    [
        {"page": None},
        {"page": 0},
        {"text": "  "},
        {"source": ""},
        {"source": "webmd"},
    ],
)
def test_a_stored_vector_without_a_proper_citation_is_an_error_naming_it(broken):
    store = make_store(make_record("pfa-p9-c0", [1, 0, 0, 0], **broken))
    with pytest.raises(RetrievalError, match="pfa-p9-c0"):
        retrieve("question", embedder=QueryEmbedder(), store=store)


def test_a_vector_with_no_page_key_at_all_is_an_error_naming_it():
    record = make_record("pfa-p9-c0", [1, 0, 0, 0])
    del record["metadata"]["page"]
    with pytest.raises(RetrievalError, match="pfa-p9-c0"):
        retrieve("question", embedder=QueryEmbedder(), store=make_store(record))


def test_without_an_embedder_or_store_the_default_ones_are_used(monkeypatch):
    embedder = QueryEmbedder()
    monkeypatch.setattr(retrieval, "_default_embedder", lambda: embedder)
    monkeypatch.setattr(retrieval, "_default_store", lambda: make_store(GRADED[0]))
    assert [c.page for c in retrieve("question")] == [1]
    assert embedder.queries_seen == ["question"]


def test_the_question_text_is_not_logged(caplog):
    import logging

    with caplog.at_level(logging.DEBUG):
        retrieve("a very private question", embedder=QueryEmbedder(), store=make_store(GRADED[0]))
    assert "private" not in caplog.text


# --- reranking ---------------------------------------------------------------------------------------------------


def many(n: int = 8) -> InMemoryStore:
    """n stored chunks that all point the same way, so the search order is simply 1, 2, 3, ... by page."""
    return make_store(*(make_record(f"pfa-p{i}-c0", [1, 0.01 * i, 0, 0], page=float(i)) for i in range(1, n + 1)))


def test_with_a_reranker_the_store_is_asked_for_candidates_and_the_best_k_come_back_in_rerank_order():
    store, reranker = many(), FakeReranker(order=[3, 1, 0])
    results = retrieve("a question", k=3, embedder=QueryEmbedder(), store=store, reranker=reranker)
    assert store.query_ks == [20]
    assert [r.page for r in results] == [4, 2, 1]  # index 3 is page 4, and so on
    assert [r.score for r in results] == [0.9, 0.8, 0.7]  # the reranker's scores replace the search scores
    assert reranker.calls[0]["top_k"] == 3 and len(reranker.calls[0]["documents"]) == 8


def test_the_number_of_candidates_can_be_set_and_never_goes_below_k():
    store = many()
    retrieve("q", k=3, embedder=QueryEmbedder(), store=store, reranker=FakeReranker(), candidates=8)
    retrieve("q", k=30, embedder=QueryEmbedder(), store=store, reranker=FakeReranker(), candidates=8)
    assert store.query_ks == [8, 30]


def test_without_a_reranker_the_store_is_asked_for_exactly_k_and_behaviour_is_unchanged(monkeypatch):
    monkeypatch.setattr(retrieval, "_default_reranker", lambda: None)
    store = many()
    results = retrieve("q", k=3, embedder=QueryEmbedder(), store=store)
    assert store.query_ks == [3] and len(results) == 3


def test_a_default_reranker_is_used_when_none_is_passed(monkeypatch):
    reranker = FakeReranker(order=[1, 0])
    monkeypatch.setattr(retrieval, "_default_reranker", lambda: reranker)
    results = retrieve("q", k=2, embedder=QueryEmbedder(), store=many())
    assert [r.page for r in results] == [2, 1] and len(reranker.calls) == 1


def test_a_failing_reranker_falls_back_to_the_search_order_with_a_warning_that_hides_the_question(caplog):
    reranker = FakeReranker(raises=RetrievalError("Voyage rerank failed: down"))
    with caplog.at_level(logging.WARNING):
        results = retrieve("a very private question", k=3, embedder=QueryEmbedder(), store=many(), reranker=reranker)
    assert [r.page for r in results] == [1, 2, 3]
    assert "Reranking failed" in caplog.text and "private" not in caplog.text


def test_a_single_candidate_is_not_sent_to_the_reranker():
    reranker = FakeReranker()
    results = retrieve("q", k=5, embedder=QueryEmbedder(), store=many(1), reranker=reranker)
    assert len(results) == 1 and reranker.calls == []


def test_a_corrupt_stored_vector_still_stops_the_search_even_with_a_reranker():
    store = make_store(make_record("pfa-p9-c0", [1, 0, 0, 0], page=None), make_record("pfa-p1-c0", [1, 0, 0, 0]))
    with pytest.raises(RetrievalError, match="pfa-p9-c0"):
        retrieve("q", embedder=QueryEmbedder(), store=store, reranker=FakeReranker())
