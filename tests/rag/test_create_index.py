from types import SimpleNamespace

import pytest
from pinecone import ApiError

from app.rag.errors import RetrievalError
from scripts import create_index
from scripts.create_index import IndexPlan, describe_plan, ensure_index, main

PLAN = IndexPlan(name="somajji-who")


class FakeSpec:
    """Stands in for pinecone.ServerlessSpec."""

    def __init__(self, cloud: str, region: str):
        self.cloud, self.region = cloud, region


def make_index(dimension=1024, metric="cosine", ready=True):
    return SimpleNamespace(dimension=dimension, metric=metric, status=SimpleNamespace(ready=ready))


class FakePinecone:
    def __init__(self, existing=None, ready_after: int = 0, raises: Exception | None = None):
        self.existing = existing  # an index object if it already exists
        self.ready_after = ready_after  # describe_index says "not ready" this many times after creation
        self.raises = raises
        self.created: list[dict] = []
        self.describe_calls = 0

    def has_index(self, name):
        return self.existing is not None

    def create_index(self, **kwargs):
        if self.raises:
            raise self.raises
        self.created.append(kwargs)
        self.existing = make_index(ready=False)

    def describe_index(self, name):
        self.describe_calls += 1
        if self.created and self.describe_calls > self.ready_after:
            self.existing.status.ready = True
        return self.existing


def run(pc, create=True, **kwargs):
    sleeps: list[float] = []
    result = ensure_index(pc, PLAN, create=create, sleep=sleeps.append, spec_class=FakeSpec, **kwargs)
    return result, sleeps


def test_the_default_run_only_describes_the_plan_and_creates_nothing():
    pc = FakePinecone()
    result, _ = run(pc, create=False)
    assert result == "planned" and pc.created == []


def test_the_plan_text_states_every_choice_that_matters():
    text = describe_plan(PLAN)
    for expected in ("somajji-who", "aws", "us-east-1", "cosine", "1024", "serverless", "deletion protection"):
        assert expected in text
    assert text.rstrip().endswith("off")


def test_create_makes_exactly_one_serverless_index_with_the_planned_settings():
    pc = FakePinecone()
    result, _ = run(pc)
    assert result == "created"
    [call] = pc.created
    assert (call["name"], call["dimension"], call["metric"]) == ("somajji-who", 1024, "cosine")
    assert (call["spec"].cloud, call["spec"].region) == ("aws", "us-east-1")
    assert call["deletion_protection"] == "disabled"
    assert call["timeout"] == -1  # return at once; we wait ourselves, so we can show progress and give up cleanly


def test_it_waits_until_the_new_index_is_ready():
    pc = FakePinecone(ready_after=3)
    result, sleeps = run(pc)
    assert result == "created" and len(sleeps) >= 2


def test_it_gives_up_with_a_clear_error_if_the_index_never_becomes_ready():
    pc = FakePinecone(ready_after=10_000)
    with pytest.raises(RetrievalError, match="not ready"):
        run(pc, wait_attempts=3)


def test_an_existing_index_with_the_same_settings_is_left_alone_even_with_create():
    pc = FakePinecone(existing=make_index())
    result, _ = run(pc)
    assert result == "exists" and pc.created == []


@pytest.mark.parametrize("existing", [make_index(dimension=768), make_index(metric="dotproduct")])
def test_an_existing_index_with_other_settings_is_refused_and_not_touched(existing):
    pc = FakePinecone(existing=existing)
    with pytest.raises(RetrievalError, match="already exists") as error:
        run(pc)
    assert "somajji-who" in str(error.value) and pc.created == []


def test_a_quota_error_explains_the_free_plan_limit():
    pc = FakePinecone(raises=ApiError("QUOTA_EXCEEDED", status_code=403))
    with pytest.raises(RetrievalError, match="5 indexes"):
        run(pc)


@pytest.fixture
def wired(monkeypatch):
    holder = SimpleNamespace(pc=FakePinecone(), built=[])

    def make_pinecone(api_key):
        holder.built.append(api_key)
        return holder.pc

    monkeypatch.setattr(create_index, "Pinecone", make_pinecone)
    monkeypatch.setattr(create_index, "ServerlessSpec", FakeSpec)
    monkeypatch.setattr(create_index.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        create_index, "get_settings", lambda: SimpleNamespace(pinecone_api_key="KEY", pinecone_index="from-settings")
    )
    return holder


def test_main_without_create_prints_the_plan_and_creates_nothing(wired, capsys):
    assert main([]) == 0
    assert wired.pc.created == []
    output = capsys.readouterr().out
    assert "from-settings" in output and "nothing was created" in output.lower()


def test_main_with_create_creates_the_index_named_in_the_settings(wired, capsys):
    assert main(["--create"]) == 0
    assert wired.pc.created[0]["name"] == "from-settings"
    assert "created" in capsys.readouterr().out.lower()


def test_main_accepts_a_pasted_long_dash(wired):
    assert main(["—create"]) == 0
    assert len(wired.pc.created) == 1


def test_main_without_a_key_stops_before_connecting(wired, monkeypatch, capsys):
    monkeypatch.setattr(
        create_index, "get_settings", lambda: SimpleNamespace(pinecone_api_key="", pinecone_index="x")
    )
    assert main(["--create"]) == 1
    assert wired.built == []
    assert "PINECONE_API_KEY" in capsys.readouterr().err
