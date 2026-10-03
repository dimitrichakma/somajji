"""Every setting the app reads, in one place. Values come from the environment or `.env`."""
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


# Hosts hand out the database URL in different forms (Railway uses `postgresql://`,
# some use `postgres://`), so accept any of them and convert to what each library needs.
_DB_SCHEMES = ("postgresql+psycopg://", "postgresql://", "postgres://")


def _with_scheme(url: str, scheme: str) -> str:
    for prefix in _DB_SCHEMES:
        if url.startswith(prefix):
            return scheme + url[len(prefix):]
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["development", "production"] = "development"
    database_url: str  # required: the app cannot start without it

    # Keys are optional for now; the phase that first needs one will check it.
    anthropic_api_key: str = ""
    voyage_api_key: str = ""
    pinecone_api_key: str = ""
    langsmith_api_key: str = ""

    demo_mode: bool = False
    admin_emails: Annotated[list[str], NoDecode] = []

    daily_budget_usd: float = 1.0
    eval_max_usd: float = 2.0

    @field_validator("admin_emails", mode="before")
    @classmethod
    def split_admin_emails(cls, value: object) -> object:
        if isinstance(value, str):
            return [email.strip() for email in value.split(",") if email.strip()]
        return value

    @property
    def sqlalchemy_database_url(self) -> str:
        """SQLAlchemy form. Without `+psycopg` it would look for the psycopg2 driver, which we don't install."""
        return _with_scheme(self.database_url, "postgresql+psycopg://")

    @property
    def psycopg_database_url(self) -> str:
        """Plain `postgresql://` form. `psycopg.connect` rejects the SQLAlchemy `+psycopg` prefix."""
        return _with_scheme(self.database_url, "postgresql://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
