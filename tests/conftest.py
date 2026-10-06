"""Test-wide environment. Set before any app module is imported."""
import os

# Local Postgres from compose.yaml. A real DATABASE_URL in the shell wins.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+psycopg://somajji:localdev@localhost:5432/somajji"
)
# Tests never call Claude, so a dummy key is enough for the client objects to be built.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
# Keep the LLM cache in memory so importing app.llm never opens a database connection.
os.environ["LLM_CACHE"] = "0"
# The unit tests expect the default embedding model, whatever a developer put in .env for an experiment.
os.environ["VOYAGE_MODEL"] = "voyage-4"
# Reranking is off in unit tests (an empty model name switches it off), so no test can call Voyage by accident.
os.environ["VOYAGE_RERANK_MODEL"] = ""
