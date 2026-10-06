import pytest
from pydantic import ValidationError

from app.config import Settings


def make_settings(**overrides) -> Settings:
    """Build Settings from explicit values only, ignoring any real .env file."""
    values = {"database_url": "postgresql+psycopg://u:p@localhost:5432/db"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_missing_database_url_names_the_setting(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError) as error:
        Settings(_env_file=None)
    assert "DATABASE_URL" in str(error.value).upper()


def test_defaults_are_safe(monkeypatch):
    for name in ("ENV", "DEMO_MODE", "DAILY_BUDGET_USD", "EVAL_MAX_USD", "ADMIN_EMAILS"):
        monkeypatch.delenv(name, raising=False)
    settings = make_settings()
    assert settings.env == "development"
    assert settings.demo_mode is False
    assert settings.daily_budget_usd == 1.0
    assert settings.eval_max_usd == 2.0
    assert settings.admin_emails == []


def test_admin_emails_are_split_and_trimmed():
    settings = make_settings(admin_emails=" a@x.com, b@x.com ,,")
    assert settings.admin_emails == ["a@x.com", "b@x.com"]


@pytest.mark.parametrize(
    "given",
    [
        "postgresql+psycopg://u:p@host:5432/db",
        "postgresql://u:p@host:5432/db",  # what Railway provides
        "postgres://u:p@host:5432/db",
    ],
)
def test_database_url_works_in_any_form(given):
    settings = make_settings(database_url=given)
    assert settings.sqlalchemy_database_url == "postgresql+psycopg://u:p@host:5432/db"
    assert settings.psycopg_database_url == "postgresql://u:p@host:5432/db"


def test_pinecone_index_has_a_default(monkeypatch):
    monkeypatch.delenv("PINECONE_INDEX", raising=False)
    assert make_settings().pinecone_index == "somajji-who"


def test_pinecone_index_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("PINECONE_INDEX", "my-test-index")
    assert make_settings().pinecone_index == "my-test-index"


def test_the_embedding_model_defaults_to_voyage_4(monkeypatch):
    monkeypatch.delenv("VOYAGE_MODEL", raising=False)
    assert make_settings().voyage_model == "voyage-4"


def test_the_embedding_model_can_be_changed_from_the_environment(monkeypatch):
    monkeypatch.setenv("VOYAGE_MODEL", "voyage-4-large")
    assert make_settings().voyage_model == "voyage-4-large"


def test_the_reranker_defaults_to_rerank_3_lite_and_can_be_changed_or_switched_off(monkeypatch):
    monkeypatch.delenv("VOYAGE_RERANK_MODEL", raising=False)
    assert make_settings().voyage_rerank_model == "rerank-3-lite"
    monkeypatch.setenv("VOYAGE_RERANK_MODEL", "rerank-3")
    assert make_settings().voyage_rerank_model == "rerank-3"
    monkeypatch.setenv("VOYAGE_RERANK_MODEL", "")
    assert make_settings().voyage_rerank_model == ""  # empty means: reranking is off
