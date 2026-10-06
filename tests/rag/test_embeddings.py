from types import SimpleNamespace

import pytest
import voyageai.error as voyage_errors

from app.rag import embeddings
from app.rag.embeddings import BATCH_SIZE, DIMENSION, VoyageEmbedder
from app.rag.errors import RetrievalError


class FakeVoyageClient:
    """Stands in for voyageai.Client. The first number of each vector is the number in the text ("t7" -> 7)."""

    def __init__(self, dimension: int = DIMENSION, raises: Exception | None = None, drop: int = 0):
        self.calls: list[dict] = []
        self.dimension = dimension
        self.raises = raises
        self.drop = drop  # return this many fewer vectors than texts

    def embed(self, texts, model=None, input_type=None, truncation=True, output_dtype=None, output_dimension=None):
        self.calls.append({"texts": texts, "model": model, "input_type": input_type, "dimension": output_dimension})
        if self.raises:
            raise self.raises
        vectors = [[float(text[1:] or 0)] + [0.0] * (self.dimension - 1) for text in texts]
        return SimpleNamespace(embeddings=vectors[: len(vectors) - self.drop])


def test_documents_are_sent_with_the_model_input_type_and_dimension():
    client = FakeVoyageClient()
    VoyageEmbedder(client=client).embed_documents(["t1", "t2"])
    [call] = client.calls
    assert (call["model"], call["input_type"], call["dimension"]) == ("voyage-4", "document", 1024)
    assert DIMENSION == 1024


def test_a_query_is_sent_as_a_query_and_returns_one_vector():
    client = FakeVoyageClient()
    vector = VoyageEmbedder(client=client).embed_query("t5")
    assert client.calls[0]["input_type"] == "query"
    assert vector[0] == 5.0 and len(vector) == DIMENSION


def test_many_texts_go_out_in_batches_and_come_back_in_order():
    client = FakeVoyageClient()
    texts = [f"t{i}" for i in range(70)]
    vectors = VoyageEmbedder(client=client).embed_documents(texts)
    assert [len(call["texts"]) for call in client.calls] == [BATCH_SIZE, BATCH_SIZE, 70 - 2 * BATCH_SIZE]
    assert [vector[0] for vector in vectors] == [float(i) for i in range(70)]


def test_no_texts_means_no_call():
    client = FakeVoyageClient()
    assert VoyageEmbedder(client=client).embed_documents([]) == []
    assert client.calls == []


@pytest.mark.parametrize(
    "error_class",
    [
        voyage_errors.RateLimitError,
        voyage_errors.AuthenticationError,
        voyage_errors.APIConnectionError,
        voyage_errors.ServerError,
    ],
)
def test_voyage_errors_become_retrieval_errors_and_keep_the_cause(error_class):
    original = error_class("boom")
    embedder = VoyageEmbedder(client=FakeVoyageClient(raises=original))
    for call in (lambda: embedder.embed_documents(["t1"]), lambda: embedder.embed_query("t1")):
        with pytest.raises(RetrievalError, match="Voyage") as error:
            call()
        assert error.value.__cause__ is original
        assert "boom" in str(error.value)


def test_a_missing_key_fails_before_any_client_is_built(monkeypatch):
    built = []
    monkeypatch.setattr(embeddings.voyageai, "Client", lambda **kwargs: built.append(kwargs))
    with pytest.raises(RetrievalError, match="VOYAGE_API_KEY"):
        VoyageEmbedder(api_key="")
    assert built == []


def test_the_real_client_gets_a_timeout_and_retries(monkeypatch):
    built = []
    monkeypatch.setattr(embeddings.voyageai, "Client", lambda **kwargs: built.append(kwargs))
    VoyageEmbedder(api_key="key-from-argument")
    [kwargs] = built
    assert kwargs["api_key"] == "key-from-argument"
    assert kwargs["timeout"] and kwargs["timeout"] > 0
    assert kwargs["max_retries"] >= 1


def test_the_key_comes_from_settings_when_none_is_passed(monkeypatch):
    built = []
    monkeypatch.setattr(embeddings.voyageai, "Client", lambda **kwargs: built.append(kwargs))
    monkeypatch.setattr(embeddings, "get_settings", lambda: SimpleNamespace(voyage_api_key="key-from-settings", voyage_model="voyage-4"))
    VoyageEmbedder()
    assert built[0]["api_key"] == "key-from-settings"


def test_vectors_of_the_wrong_size_are_rejected():
    embedder = VoyageEmbedder(client=FakeVoyageClient(dimension=512))
    with pytest.raises(RetrievalError, match="1024"):
        embedder.embed_documents(["t1"])


def test_a_wrong_number_of_vectors_is_rejected():
    embedder = VoyageEmbedder(client=FakeVoyageClient(drop=1))
    with pytest.raises(RetrievalError, match="vectors"):
        embedder.embed_documents(["t1", "t2"])


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_an_empty_query_is_rejected_without_a_call(query):
    client = FakeVoyageClient()
    with pytest.raises(RetrievalError, match="empty"):
        VoyageEmbedder(client=client).embed_query(query)
    assert client.calls == []


def test_a_pause_is_taken_between_batches_but_not_after_the_last_one():
    sleeps: list[float] = []
    embedder = VoyageEmbedder(client=FakeVoyageClient(), batch_pause_seconds=5, sleep=sleeps.append)
    embedder.embed_documents([f"t{i}" for i in range(70)])  # three batches
    assert sleeps == [5, 5]


def test_there_is_no_pause_by_default_or_for_a_single_batch():
    sleeps: list[float] = []
    VoyageEmbedder(client=FakeVoyageClient(), sleep=sleeps.append).embed_documents([f"t{i}" for i in range(70)])
    VoyageEmbedder(client=FakeVoyageClient(), batch_pause_seconds=5, sleep=sleeps.append).embed_documents(["t1"])
    assert sleeps == []


def test_a_rate_limit_error_tells_the_user_how_to_slow_down():
    embedder = VoyageEmbedder(client=FakeVoyageClient(raises=voyage_errors.RateLimitError("slow down")))
    with pytest.raises(RetrievalError, match="batch-pause") as error:
        embedder.embed_documents(["t1"])
    assert "slow down" in str(error.value)


def test_other_errors_do_not_get_the_rate_limit_hint():
    embedder = VoyageEmbedder(client=FakeVoyageClient(raises=voyage_errors.ServerError("boom")))
    with pytest.raises(RetrievalError) as error:
        embedder.embed_documents(["t1"])
    assert "batch-pause" not in str(error.value)


def test_the_model_comes_from_the_settings_unless_one_is_given():
    default = FakeVoyageClient()
    VoyageEmbedder(client=default).embed_documents(["t1"])
    assert default.calls[0]["model"] == "voyage-4"  # tests/conftest.py sets VOYAGE_MODEL=voyage-4

    chosen = FakeVoyageClient()
    VoyageEmbedder(client=chosen, model="voyage-4-large").embed_query("t1")
    assert chosen.calls[0]["model"] == "voyage-4-large"
