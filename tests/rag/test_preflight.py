from types import SimpleNamespace

import pytest
from pinecone import ApiError

from app.rag.errors import RetrievalError
from scripts import preflight
from scripts.preflight import (
    IndexInfo,
    check_keys,
    format_report,
    list_indexes,
    main,
    probe_rerank,
    probe_voyage,
    run_preflight,
)

VOYAGE_SECRET = "voyage-SECRET-value-123"
PINECONE_SECRET = "pinecone-SECRET-value-456"


def make_settings(voyage: str = VOYAGE_SECRET, pinecone: str = PINECONE_SECRET, index: str = "somajji-who"):
    return SimpleNamespace(voyage_api_key=voyage, pinecone_api_key=pinecone, pinecone_index=index, voyage_model="voyage-4", voyage_rerank_model="rerank-3-lite")


def make_index(name="cbt-mental-health", dimension=1024, metric="cosine", ready=True, spec=None):
    spec = {"serverless": {"cloud": "aws", "region": "us-east-1"}} if spec is None else spec
    return SimpleNamespace(name=name, dimension=dimension, metric=metric, status=SimpleNamespace(ready=ready), spec=spec)


class FakeEmbedder:
    def __init__(self, size: int = 1024, raises: Exception | None = None):
        self.size, self.raises, self.calls = size, raises, []

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.raises:
            raise self.raises
        return [0.1] * self.size


class FakeReranker:
    def __init__(self, raises: Exception | None = None, bad: bool = False):
        self.raises, self.bad, self.calls = raises, bad, []

    def rerank(self, query, documents, top_k):
        self.calls.append((query, list(documents), top_k))
        if self.raises:
            raise self.raises
        return [] if self.bad else [(0, 0.9)]


class FakePinecone:
    def __init__(self, indexes=(), raises: Exception | None = None):
        self.indexes, self.raises, self.calls = list(indexes), raises, []

    def list_indexes(self):
        self.calls.append("list_indexes")
        if self.raises:
            raise self.raises
        return self.indexes


def test_keys_are_reported_by_name_and_never_by_value():
    assert check_keys(make_settings()) == {"VOYAGE_API_KEY": True, "PINECONE_API_KEY": True}
    assert check_keys(make_settings(voyage="", pinecone="  ")) == {"VOYAGE_API_KEY": False, "PINECONE_API_KEY": False}


def test_the_voyage_probe_sends_one_short_sentence_and_checks_the_size():
    embedder = FakeEmbedder()
    assert probe_voyage(embedder) == (1024, None)
    assert len(embedder.calls) == 1 and len(embedder.calls[0]) < 40


def test_the_voyage_probe_reports_a_wrong_size_or_an_error_instead_of_raising():
    size, error = probe_voyage(FakeEmbedder(size=512))
    assert size == 512 and "1024" in error
    size, error = probe_voyage(FakeEmbedder(raises=RetrievalError("Voyage embedding failed: bad model")))
    assert size is None and "bad model" in error


def test_pinecone_indexes_are_listed_with_size_metric_place_and_readiness():
    indexes = list_indexes(FakePinecone([make_index(), make_index("other", 768, "dotproduct", ready=False)]))
    assert indexes[0] == IndexInfo("cbt-mental-health", 1024, "cosine", True, "aws/us-east-1")
    assert (indexes[1].dimension, indexes[1].metric, indexes[1].ready) == (768, "dotproduct", False)


def test_a_place_that_cannot_be_read_is_reported_as_unknown_not_a_crash():
    assert list_indexes(FakePinecone([make_index(spec="weird")]))[0].where == "unknown"


def test_the_place_can_be_read_from_an_object_as_well_as_a_dict():
    spec = SimpleNamespace(serverless=SimpleNamespace(cloud="gcp", region="europe-west4"))
    assert list_indexes(FakePinecone([make_index(spec=spec)]))[0].where == "gcp/europe-west4"


def test_a_pinecone_error_becomes_a_retrieval_error():
    with pytest.raises(RetrievalError, match="Pinecone"):
        list_indexes(FakePinecone(raises=ApiError("nope", status_code=401)))


def test_the_report_says_how_many_free_indexes_are_used_and_whether_the_target_exists():
    report = run_preflight(make_settings(), FakeEmbedder(), FakePinecone([make_index(), make_index("somajji-who")]))
    text = format_report(report)
    assert "2 of 5" in text
    assert "'somajji-who' already exists" in text
    report = run_preflight(make_settings(), FakeEmbedder(), FakePinecone([make_index()]))
    assert "'somajji-who' does not exist yet" in format_report(report)


def test_the_report_never_contains_a_key_value():
    text = format_report(run_preflight(make_settings(), FakeEmbedder(), FakePinecone([make_index()])))
    assert VOYAGE_SECRET not in text and PINECONE_SECRET not in text
    assert "VOYAGE_API_KEY" in text and "set" in text


def test_a_missing_key_skips_that_service_and_makes_no_call():
    embedder, pinecone = FakeEmbedder(), FakePinecone()
    report = run_preflight(make_settings(voyage="", pinecone=""), embedder, pinecone)
    assert embedder.calls == [] and pinecone.calls == []
    assert "not set" in format_report(report)


@pytest.fixture
def wired(monkeypatch):
    built = {"voyage": 0, "pinecone": 0}
    embedder, pinecone = FakeEmbedder(), FakePinecone([make_index()])

    def make_embedder(**kwargs):
        built["voyage"] += 1
        return embedder

    def make_pinecone(api_key):
        built["pinecone"] += 1
        return pinecone

    monkeypatch.setattr(preflight, "VoyageEmbedder", make_embedder)
    monkeypatch.setattr(preflight, "VoyageReranker", lambda **kwargs: FakeReranker())
    monkeypatch.setattr(preflight, "Pinecone", make_pinecone)
    return SimpleNamespace(built=built, embedder=embedder, pinecone=pinecone, monkeypatch=monkeypatch)


def test_main_returns_zero_when_everything_works_and_never_prints_a_key(wired, capsys):
    wired.monkeypatch.setattr(preflight, "get_settings", lambda: make_settings())
    assert main([]) == 0
    output = capsys.readouterr()
    assert "OK" in output.out
    assert VOYAGE_SECRET not in output.out + output.err and PINECONE_SECRET not in output.out + output.err


def test_main_returns_one_and_builds_nothing_when_keys_are_missing(wired, capsys):
    wired.monkeypatch.setattr(preflight, "get_settings", lambda: make_settings(voyage="", pinecone=""))
    assert main([]) == 1
    assert wired.built == {"voyage": 0, "pinecone": 0}
    assert "VOYAGE_API_KEY" in capsys.readouterr().out


def test_main_returns_one_when_a_service_fails(wired, capsys):
    wired.pinecone.raises = ApiError("forbidden", status_code=403)
    wired.monkeypatch.setattr(preflight, "get_settings", lambda: make_settings())
    assert main([]) == 1
    assert "forbidden" in capsys.readouterr().out


def test_the_rerank_probe_sends_a_tiny_request_and_checks_one_result_comes_back():
    reranker = FakeReranker()
    assert probe_rerank(reranker) is None
    assert len(reranker.calls) == 1 and len(reranker.calls[0][1]) == 2
    assert "no result" in probe_rerank(FakeReranker(bad=True))
    assert "rerank down" in probe_rerank(FakeReranker(raises=RetrievalError("rerank down")))


def test_the_report_shows_the_rerank_model_and_whether_it_works():
    ok = run_preflight(make_settings(), FakeEmbedder(), FakePinecone([make_index()]), FakeReranker())
    assert "Rerank: model rerank-3-lite: OK" in format_report(ok)
    bad = run_preflight(make_settings(), FakeEmbedder(), FakePinecone([make_index()]), FakeReranker(raises=RetrievalError("nope")))
    assert "Rerank: model rerank-3-lite: FAILED: nope" in format_report(bad) and not bad.ok


def test_an_empty_rerank_model_is_reported_as_off_and_is_not_a_problem():
    settings = make_settings()
    settings.voyage_rerank_model = ""
    report = run_preflight(settings, FakeEmbedder(), FakePinecone([make_index()]), None)
    assert "Rerank: off" in format_report(report) and report.ok
