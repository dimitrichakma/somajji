from types import SimpleNamespace

import pytest
import voyageai.error as voyage_errors

from app.rag import rerank
from app.rag.errors import RetrievalError
from app.rag.rerank import VoyageReranker

DOCS = ["first", "second", "third", "fourth"]


class FakeRerankClient:
    def __init__(self, order=None, raises=None):
        self.order, self.raises, self.calls = order, raises, []

    def rerank(self, query, documents, model, top_k=None, truncation=True):
        self.calls.append({"query": query, "documents": list(documents), "model": model, "top_k": top_k})
        if self.raises:
            raise self.raises
        order = self.order if self.order is not None else list(range(len(documents)))
        results = [SimpleNamespace(index=i, relevance_score=0.9 - 0.1 * n) for n, i in enumerate(order[:top_k])]
        return SimpleNamespace(results=results)


def make(client=None, **kwargs):
    return VoyageReranker(client=client or FakeRerankClient(), model="rerank-3-lite", **kwargs)


def test_it_sends_the_query_documents_model_and_top_k():
    client = FakeRerankClient()
    make(client).rerank("my question", DOCS, top_k=2)
    assert client.calls == [{"query": "my question", "documents": DOCS, "model": "rerank-3-lite", "top_k": 2}]


def test_it_returns_index_and_score_pairs_best_first_limited_to_top_k():
    ranked = make(FakeRerankClient(order=[2, 0, 3, 1])).rerank("q", DOCS, top_k=3)
    assert [index for index, _ in ranked] == [2, 0, 3]
    assert [score for _, score in ranked] == sorted((score for _, score in ranked), reverse=True)
    assert all(isinstance(score, float) for _, score in ranked)


def test_no_documents_means_no_call():
    client = FakeRerankClient()
    assert make(client).rerank("q", [], top_k=5) == []
    assert client.calls == []


def test_an_empty_question_is_rejected_without_a_call():
    client = FakeRerankClient()
    with pytest.raises(RetrievalError, match="empty"):
        make(client).rerank("  ", DOCS, top_k=2)
    assert client.calls == []


def test_voyage_errors_become_retrieval_errors_and_keep_the_cause():
    original = voyage_errors.ServerError("boom")
    with pytest.raises(RetrievalError, match="Voyage rerank failed") as error:
        make(FakeRerankClient(raises=original)).rerank("q", DOCS, top_k=2)
    assert error.value.__cause__ is original and "boom" in str(error.value)


def test_a_rate_limit_error_says_to_wait_or_add_a_payment_method():
    with pytest.raises(RetrievalError, match="rate limit"):
        make(FakeRerankClient(raises=voyage_errors.RateLimitError("slow"))).rerank("q", DOCS, top_k=2)


@pytest.mark.parametrize("order", [[0, 9], [-1, 0], [1, 1]])
def test_an_index_out_of_range_or_repeated_is_an_error(order):
    with pytest.raises(RetrievalError, match="invalid ranking"):
        make(FakeRerankClient(order=order)).rerank("q", DOCS, top_k=2)


def test_a_missing_key_fails_before_any_client_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(rerank.voyageai, "Client", lambda **kwargs: built.append(kwargs))
    with pytest.raises(RetrievalError, match="VOYAGE_API_KEY"):
        VoyageReranker(api_key="", model="rerank-3-lite")
    assert built == []


def test_the_real_client_gets_a_timeout_and_retries(monkeypatch):
    built = []
    monkeypatch.setattr(rerank.voyageai, "Client", lambda **kwargs: built.append(kwargs))
    VoyageReranker(api_key="key", model="rerank-3-lite")
    [kwargs] = built
    assert kwargs["api_key"] == "key" and kwargs["timeout"] > 0 and kwargs["max_retries"] >= 1


def test_the_key_and_model_come_from_the_settings_unless_given(monkeypatch):
    built = []
    monkeypatch.setattr(rerank.voyageai, "Client", lambda **kwargs: built.append(kwargs) or FakeRerankClient())
    monkeypatch.setattr(
        rerank, "get_settings", lambda: SimpleNamespace(voyage_api_key="key-from-settings", voyage_rerank_model="rerank-x")
    )
    client_used = VoyageReranker()
    assert built[0]["api_key"] == "key-from-settings"
    assert client_used._model == "rerank-x"
    assert VoyageReranker(model="rerank-y")._model == "rerank-y"


def test_an_empty_model_name_means_reranking_is_off_and_is_refused():
    with pytest.raises(RetrievalError, match="VOYAGE_RERANK_MODEL"):
        VoyageReranker(client=FakeRerankClient(), model="")
