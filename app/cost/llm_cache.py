"""LLM response cache, Postgres-backed.

An exact-match cache on (prompt, model+params): identical classifier calls -
the same safety check on a repeated message, the same routing decision for a
repeated question, an eval re-scoring the same (answer, context) - are served
from the DB instead of the API. Big win for the example-question buttons, for
testers asking similar things, and for eval iteration.

It does NOT help answer synthesis (the retrieved context is unique per request)
and it is exact-match only (no semantic similarity).

  LLM_CACHE=0   disable entirely
  DATABASE_URL  cache in Postgres (persists, shared across workers); if Postgres
                cannot be reached, fall back to an in-process cache
"""
import hashlib
import logging
import os

import psycopg
from langchain_core.caches import BaseCache, InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads

from app.config import get_settings

logger = logging.getLogger("llm_cache")

_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash TEXT NOT NULL,
    llm_hash    TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (prompt_hash, llm_hash)
)
"""


def _h(s):
    return hashlib.sha256(s.encode()).hexdigest()


class PostgresLLMCache(BaseCache):
    def __init__(self, database_url):
        self._url = database_url
        with psycopg.connect(self._url, autocommit=True, connect_timeout=3) as conn:
            conn.execute(_TABLE_SQL)

    def lookup(self, prompt, llm_string):
        try:
            with psycopg.connect(self._url, autocommit=True, connect_timeout=3) as conn:
                row = conn.execute(
                    "SELECT response FROM llm_cache WHERE prompt_hash = %s AND llm_hash = %s",
                    (_h(prompt), _h(llm_string)),
                ).fetchone()
            return loads(row[0]) if row else None
        except Exception:
            logger.warning("llm_cache lookup failed; treating as miss", exc_info=True)
            return None

    def update(self, prompt, llm_string, return_val):
        try:
            with psycopg.connect(self._url, autocommit=True, connect_timeout=3) as conn:
                conn.execute(
                    """INSERT INTO llm_cache (prompt_hash, llm_hash, response)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (prompt_hash, llm_hash) DO NOTHING""",
                    (_h(prompt), _h(llm_string), dumps(return_val)),
                )
        except Exception:
            logger.warning("llm_cache write failed; skipping", exc_info=True)

    def clear(self, **kwargs):
        with psycopg.connect(self._url, autocommit=True, connect_timeout=3) as conn:
            conn.execute("TRUNCATE llm_cache")


def configure_llm_cache():
    """Install the global LangChain LLM cache. Called once at import time."""
    if os.getenv("LLM_CACHE", "1") == "0":
        return None
    try:
        cache = PostgresLLMCache(get_settings().psycopg_database_url)
    except Exception:
        logger.warning("could not init Postgres LLM cache; using in-process", exc_info=True)
        cache = InMemoryCache()
    set_llm_cache(cache)
    return cache
