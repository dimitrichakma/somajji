"""Every setting the app reads, in one place. Values come from the environment or `.env`."""
from functools import lru_cache
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


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
    def psycopg_database_url(self) -> str:
        """Plain `postgresql://` form. `psycopg.connect` rejects the SQLAlchemy `+psycopg` prefix."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
