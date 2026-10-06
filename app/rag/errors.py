"""The one error the knowledge base raises. The guide agent (Phase 3) catches this."""


class RetrievalError(Exception):
    """Search could not run: empty question, missing key, or Voyage / Pinecone failed."""
