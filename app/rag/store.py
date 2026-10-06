"""Where the chunks live. `VectorStore` is what ingest needs; `PineconeStore` is the real one.

PineconeStore never creates an index: that costs money and needs Dimitri's yes first.
"""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol, TypedDict

from pinecone import NotFoundError, Pinecone, PineconeError

from app.config import get_settings

from .errors import RetrievalError

NAMESPACE = "default"
UPSERT_BATCH_SIZE = 100


class VectorRecord(TypedDict):
    id: str
    values: list[float]
    metadata: dict[str, Any]


class VectorMatch(TypedDict):
    id: str
    score: float  # cosine similarity: higher means closer
    metadata: dict[str, Any]


class VectorStore(Protocol):
    def upsert(self, records: list[VectorRecord]) -> None: ...

    def delete_source(self, source: str) -> None:
        """Remove every vector that came from this source."""
        ...

    def list_ids(self, source: str) -> list[str]:
        """Ids of the vectors stored for this source."""
        ...

    def query(self, vector: list[float], k: int) -> list[VectorMatch]:
        """The k vectors closest to `vector`, with their metadata."""
        ...


@contextmanager
def _translate_errors(action: str) -> Iterator[None]:
    try:
        yield
    except PineconeError as error:
        raise RetrievalError(f"Pinecone {action} failed: {error}") from error


class PineconeStore:
    def __init__(
        self,
        index_name: str,
        api_key: str | None = None,
        index: Any | None = None,
        namespace: str = NAMESPACE,
    ) -> None:
        self._namespace = namespace
        if index is None:
            key = get_settings().pinecone_api_key if api_key is None else api_key
            if not key:
                raise RetrievalError("PINECONE_API_KEY is not set. Add it to .env.")
            try:
                index = Pinecone(api_key=key).Index(index_name)
            except NotFoundError as error:
                raise RetrievalError(
                    f"Pinecone index '{index_name}' does not exist. It has to be created first "
                    "(creating one costs money, so ask Dimitri)."
                ) from error
            except PineconeError as error:
                raise RetrievalError(f"Pinecone could not open index '{index_name}': {error}") from error
        self._index = index

    def upsert(self, records: list[VectorRecord]) -> None:
        with _translate_errors("upsert"):
            for start in range(0, len(records), UPSERT_BATCH_SIZE):
                batch = records[start : start + UPSERT_BATCH_SIZE]
                self._index.upsert(vectors=batch, namespace=self._namespace)

    def delete_source(self, source: str) -> None:
        with _translate_errors("delete"):
            try:
                self._index.delete(filter={"source": {"$eq": source}}, namespace=self._namespace)
            except NotFoundError:
                pass  # the namespace does not exist yet (first ingest), so there is nothing to delete

    def list_ids(self, source: str) -> list[str]:
        # Chunk ids start with "<source>-", so a prefix listing finds every vector of one source.
        ids: list[str] = []
        with _translate_errors("list"):
            for page in self._index.list(prefix=f"{source}-", namespace=self._namespace):
                ids.extend(item.id for item in page.vectors)
        return ids

    def query(self, vector: list[float], k: int) -> list[VectorMatch]:
        with _translate_errors("query"):
            response = self._index.query(
                vector=vector, top_k=k, namespace=self._namespace, include_metadata=True
            )
        return [
            {"id": match.id, "score": float(match.score), "metadata": dict(match.metadata or {})}
            for match in response.matches
        ]

    def fetch(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """The stored metadata of these ids, to check what is really in the index. Missing ids are left out."""
        found: dict[str, dict[str, Any]] = {}
        with _translate_errors("fetch"):
            for start in range(0, len(ids), UPSERT_BATCH_SIZE):
                response = self._index.fetch(ids=ids[start : start + UPSERT_BATCH_SIZE], namespace=self._namespace)
                for vector_id, vector in response.vectors.items():
                    found[vector_id] = dict(vector.metadata or {})
        return found
