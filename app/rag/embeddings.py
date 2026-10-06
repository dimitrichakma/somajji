"""Turn text into vectors with Voyage. Every failure becomes a RetrievalError."""
import time
from collections.abc import Callable
from typing import Protocol

import voyageai
from voyageai.error import RateLimitError, VoyageError

from app.config import get_settings

from .errors import RetrievalError

DIMENSION = 1024  # must match the Pinecone index
# Chunks are at most about 900 tokens, so 32 per request stays well under Voyage's per-request limit.
BATCH_SIZE = 32
TIMEOUT_SECONDS = 30  # voyageai's own default is no timeout at all
MAX_RETRIES = 2  # voyageai's own default is no retries


class Embedder(Protocol):
    """What ingest and retrieval need. Tests pass a fake instead of the real one."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VoyageEmbedder:
    def __init__(
        self,
        api_key: str | None = None,
        client: voyageai.Client | None = None,
        batch_pause_seconds: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        model: str | None = None,
    ) -> None:
        # The model comes from the settings (VOYAGE_MODEL, default voyage-4) unless one is given.
        self._model = model or get_settings().voyage_model
        # A free Voyage account may allow only a few requests a minute; a pause between batches avoids that.
        self._batch_pause_seconds = batch_pause_seconds
        self._sleep = sleep
        if client is None:
            key = get_settings().voyage_api_key if api_key is None else api_key
            if not key:
                raise RetrievalError("VOYAGE_API_KEY is not set. Add it to .env.")
            client = voyageai.Client(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        self._client = client

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            response = self._client.embed(
                texts, model=self._model, input_type=input_type, output_dimension=DIMENSION
            )
        except RateLimitError as error:
            raise RetrievalError(
                f"Voyage embedding failed: {error}. The rate limit was reached: rerun with a pause between "
                "batches (for example --batch-pause 25), or add a payment method in the Voyage dashboard."
            ) from error
        except VoyageError as error:
            raise RetrievalError(f"Voyage embedding failed: {error}") from error

        vectors = response.embeddings
        if len(vectors) != len(texts):
            raise RetrievalError(f"Voyage returned {len(vectors)} vectors for {len(texts)} texts")
        if any(len(vector) != DIMENSION for vector in vectors):
            raise RetrievalError(f"Voyage returned vectors that are not {DIMENSION} numbers long")
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Chunks to be stored (input_type "document"), sent in batches and returned in order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            if start and self._batch_pause_seconds:
                self._sleep(self._batch_pause_seconds)
            vectors.extend(self._embed(texts[start : start + BATCH_SIZE], "document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """A question to search with (input_type "query")."""
        if not text.strip():
            raise RetrievalError("Cannot search with an empty question")
        return self._embed([text], "query")[0]
