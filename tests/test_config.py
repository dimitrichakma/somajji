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
