from types import SimpleNamespace

import pytest
from pinecone import ApiError, NotFoundError, PineconeError

from app.rag import store as store_module
from app.rag.errors import RetrievalError
from app.rag.store import PineconeStore
from tests.rag.fakes import FakePineconeIndex


def make_record(n: int, source: str = "pfa") -> dict:
    return {"id": f"{source}-p{n}-c0", "values": [0.1, 0.2], "metadata": {"source": source, "page": n}}


def test_records_are_upserted_in_batches_of_100_into_the_namespace():
    index = FakePineconeIndex()
    PineconeStore("somajji-who", index=index, namespace="ns").upsert([make_record(n) for n in range(1, 251)])
    assert [len(call["vectors"]) for call in index.upserts] == [100, 100, 50]
    assert {call["namespace"] for call in index.upserts} == {"ns"}
    assert index.upserts[0]["vectors"][0] == make_record(1)


def test_no_records_means_no_call():
    index = FakePineconeIndex()
    PineconeStore("x", index=index).upsert([])
    assert index.upserts == []


def test_delete_source_filters_on_the_source_in_the_namespace():
    index = FakePineconeIndex()
    PineconeStore("x", index=index, namespace="ns").delete_source("pfa")
    assert index.deletes == [{"filter": {"source": {"$eq": "pfa"}}, "namespace": "ns"}]


def test_delete_on_an_empty_namespace_is_not_an_error():
    index = FakePineconeIndex()
    index.raises["delete"] = NotFoundError("Namespace not found")
    PineconeStore("x", index=index).delete_source("pfa")  # first ever ingest: nothing to delete


def test_list_ids_uses_the_source_prefix_and_joins_the_pages():
    index = FakePineconeIndex(pages=[["pfa-p1-c0", "pfa-p2-c0"], ["pfa-p3-c0"]])
    ids = PineconeStore("x", index=index, namespace="ns").list_ids("pfa")
    assert ids == ["pfa-p1-c0", "pfa-p2-c0", "pfa-p3-c0"]
    assert index.list_calls == [{"prefix": "pfa-", "namespace": "ns"}]


@pytest.mark.parametrize(
    "method, call",
    [
        ("upsert", lambda s: s.upsert([make_record(1)])),
        ("delete", lambda s: s.delete_source("pfa")),
        ("list", lambda s: s.list_ids("pfa")),
    ],
)
def test_pinecone_errors_become_retrieval_errors_and_keep_the_cause(method, call):
    original = ApiError("server said no", status_code=500)
    index = FakePineconeIndex()
    index.raises[method] = original
    with pytest.raises(RetrievalError, match="Pinecone") as error:
        call(PineconeStore("x", index=index))
    assert error.value.__cause__ is original
    assert "server said no" in str(error.value)


def test_a_missing_key_fails_before_any_connection(monkeypatch):
    connected = []
    monkeypatch.setattr(store_module, "Pinecone", lambda **kwargs: connected.append(kwargs))
    with pytest.raises(RetrievalError, match="PINECONE_API_KEY"):
        PineconeStore("somajji-who", api_key="")
    assert connected == []


def test_the_key_comes_from_settings_when_none_is_passed(monkeypatch):
    seen = {}

    class FakePinecone:
        def __init__(self, api_key):
            seen["api_key"] = api_key

        def Index(self, name):
            seen["index"] = name
            return FakePineconeIndex()

    monkeypatch.setattr(store_module, "Pinecone", FakePinecone)
    monkeypatch.setattr(store_module, "get_settings", lambda: SimpleNamespace(pinecone_api_key="key-from-settings"))
    PineconeStore("somajji-who")
    assert seen == {"api_key": "key-from-settings", "index": "somajji-who"}


def test_a_missing_index_says_so_and_does_not_create_one(monkeypatch):
    class FakePinecone:
        created = []

        def __init__(self, api_key):
            pass

        def Index(self, name):
            raise NotFoundError("index not found")

        def create_index(self, **kwargs):
            self.created.append(kwargs)

    monkeypatch.setattr(store_module, "Pinecone", FakePinecone)
    with pytest.raises(RetrievalError, match="does not exist") as error:
        PineconeStore("somajji-who", api_key="k")
    assert "somajji-who" in str(error.value)
    assert FakePinecone.created == []


def test_connection_problems_while_opening_the_index_become_retrieval_errors(monkeypatch):
    class FakePinecone:
        def __init__(self, api_key):
            pass

        def Index(self, name):
            raise PineconeError("cannot connect")

    monkeypatch.setattr(store_module, "Pinecone", FakePinecone)
    with pytest.raises(RetrievalError, match="cannot connect"):
        PineconeStore("somajji-who", api_key="k")


def make_match(vector_id: str, score: float, metadata) -> SimpleNamespace:
    return SimpleNamespace(id=vector_id, score=score, metadata=metadata)


def test_query_sends_the_vector_top_k_namespace_and_asks_for_metadata():
    index = FakePineconeIndex()
    PineconeStore("x", index=index, namespace="ns").query([0.1, 0.2], 5)
    assert index.query_calls == [{"vector": [0.1, 0.2], "top_k": 5, "namespace": "ns", "include_metadata": True}]


def test_query_maps_matches_to_plain_dicts_and_treats_missing_metadata_as_empty():
    index = FakePineconeIndex()
    index.matches = [make_match("pfa-p1-c0", 0.9, {"page": 1.0}), make_match("pfa-p2-c0", 0.5, None)]
    assert PineconeStore("x", index=index).query([0.1], 2) == [
        {"id": "pfa-p1-c0", "score": 0.9, "metadata": {"page": 1.0}},
        {"id": "pfa-p2-c0", "score": 0.5, "metadata": {}},
    ]


def test_query_with_no_hits_returns_an_empty_list():
    assert PineconeStore("x", index=FakePineconeIndex()).query([0.1], 5) == []


def test_query_errors_become_retrieval_errors_and_keep_the_cause():
    original = ApiError("too many requests", status_code=429)
    index = FakePineconeIndex()
    index.raises["query"] = original
    with pytest.raises(RetrievalError, match="Pinecone query failed") as error:
        PineconeStore("x", index=index).query([0.1], 5)
    assert error.value.__cause__ is original


def test_fetch_returns_the_stored_metadata_by_id_in_batches_of_100():
    index = FakePineconeIndex()
    index.stored = {f"pfa-p{n}-c0": {"page": float(n), "text": f"t{n}"} for n in range(1, 251)}
    found = PineconeStore("x", index=index, namespace="ns").fetch(list(index.stored))
    assert [len(call["ids"]) for call in index.fetch_calls] == [100, 100, 50]
    assert {call["namespace"] for call in index.fetch_calls} == {"ns"}
    assert found["pfa-p7-c0"] == {"page": 7.0, "text": "t7"} and len(found) == 250


def test_fetch_leaves_out_ids_that_are_not_stored_and_treats_missing_metadata_as_empty():
    index = FakePineconeIndex()
    index.stored = {"pfa-p1-c0": None, "pfa-p2-c0": {"page": 2.0}}
    found = PineconeStore("x", index=index).fetch(["pfa-p1-c0", "pfa-p2-c0", "pfa-p3-c0"])
    assert found == {"pfa-p1-c0": {}, "pfa-p2-c0": {"page": 2.0}}


def test_fetch_errors_become_retrieval_errors_and_keep_the_cause():
    original = ApiError("unavailable", status_code=503)
    index = FakePineconeIndex()
    index.raises["fetch"] = original
    with pytest.raises(RetrievalError, match="Pinecone fetch failed") as error:
        PineconeStore("x", index=index).fetch(["pfa-p1-c0"])
    assert error.value.__cause__ is original
