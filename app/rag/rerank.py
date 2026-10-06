"""Re-score search results with a more careful model. Every failure becomes a RetrievalError.

Search first finds about 20 candidate chunks by similarity; a reranker reads the question and each chunk
together and puts the best ones on top. Measured on our questions, the right page was in the top 20 for 19 of 20
but in the top 5 for only 15; reranking lifts that to 19.
"""
from typing import Protocol

import voyageai
from voyageai.error import RateLimitError, VoyageError

from app.config import get_settings

from .embeddings import MAX_RETRIES, TIMEOUT_SECONDS
from .errors import RetrievalError


class Reranker(Protocol):
    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        """(index into `documents`, relevance score) for the best `top_k`, best first."""
        ...


class VoyageReranker:
    def __init__(self, api_key: str | None = None, client: voyageai.Client | None = None, model: str | None = None) -> None:
        self._model = get_settings().voyage_rerank_model if model is None else model
        if not self._model:
            raise RetrievalError("VOYAGE_RERANK_MODEL is empty: reranking is switched off.")
        if client is None:
            key = get_settings().voyage_api_key if api_key is None else api_key
            if not key:
                raise RetrievalError("VOYAGE_API_KEY is not set. Add it to .env.")
            client = voyageai.Client(api_key=key, timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
        self._client = client

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        if not query.strip():
            raise RetrievalError("Cannot rerank with an empty question")
        if not documents:
            return []
        try:
            response = self._client.rerank(query, documents, model=self._model, top_k=top_k)
        except RateLimitError as error:
            raise RetrievalError(
                f"Voyage rerank failed: {error}. The rate limit was reached: wait a minute, "
                "or add a payment method in the Voyage dashboard."
            ) from error
        except VoyageError as error:
            raise RetrievalError(f"Voyage rerank failed: {error}") from error

        ranked = [(result.index, float(result.relevance_score)) for result in response.results]
        indexes = [index for index, _ in ranked]
        if any(not 0 <= index < len(documents) for index in indexes) or len(set(indexes)) != len(indexes):
            raise RetrievalError("Voyage returned an invalid ranking")
        return ranked
