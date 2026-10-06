"""Stand-ins for Voyage and Pinecone, so ingest and retrieval can be tested with no keys and no cost."""
import math
from collections.abc import Iterator
from types import SimpleNamespace

from app.rag.store import VectorMatch, VectorRecord


class FakeEmbedder:
    """Gives every text a small fixed-size vector made from the text, and remembers what it was given."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.documents_seen: list[str] = []
        self.queries_seen: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = sum(ord(char) for char in text)
        return [float((seed + i) % 17) for i in range(4)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.raises:
            raise self.raises
        self.documents_seen.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        if self.raises:
            raise self.raises
        self.queries_seen.append(text)
        return self._vector(text)


def _cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


class InMemoryStore:
    """A VectorStore that lives in a dict. `lag` makes the first N list_ids calls show nothing,
    like Pinecone does for a moment after a write."""

    def __init__(self, lag: int = 0) -> None:
        self.records: dict[str, VectorRecord] = {}
        self.lag = lag
        self.delete_calls: list[str] = []
        self.query_ks: list[int] = []  # how many results each query asked for

    def upsert(self, records: list[VectorRecord]) -> None:
        for record in records:
            self.records[record["id"]] = record

    def delete_source(self, source: str) -> None:
        self.delete_calls.append(source)
        self.records = {k: v for k, v in self.records.items() if v["metadata"]["source"] != source}

    def list_ids(self, source: str) -> list[str]:
        if self.lag > 0:
            self.lag -= 1
            return []
        return sorted(k for k, v in self.records.items() if v["metadata"]["source"] == source)

    def query(self, vector: list[float], k: int) -> list[VectorMatch]:
        self.query_ks.append(k)
        scored = [
            {"id": record["id"], "score": _cosine(vector, record["values"]), "metadata": record["metadata"]}
            for record in self.records.values()
        ]
        return sorted(scored, key=lambda match: match["score"], reverse=True)[:k]


class FakePineconeIndex:
    """Looks like pinecone.Index for the few calls PineconeStore makes, and records them."""

    def __init__(self, pages: list[list[str]] | None = None) -> None:
        self.upserts: list[dict] = []
        self.deletes: list[dict] = []
        self.list_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.fetch_calls: list[dict] = []
        self.stored: dict[str, dict | None] = {}  # id -> metadata, what fetch() can find
        self.matches: list[SimpleNamespace] = []  # what query() answers with
        self.pages = pages or []
        self.raises: dict[str, Exception] = {}  # method name -> error to raise

    def _maybe_raise(self, method: str) -> None:
        if method in self.raises:
            raise self.raises[method]

    def upsert(self, *, vectors, namespace: str = "") -> None:
        self._maybe_raise("upsert")
        self.upserts.append({"vectors": list(vectors), "namespace": namespace})

    def delete(self, *, filter=None, namespace: str = "", ids=None) -> None:
        self._maybe_raise("delete")
        self.deletes.append({"filter": filter, "namespace": namespace})

    def list(self, *, prefix=None, namespace: str = "") -> Iterator:
        self._maybe_raise("list")
        self.list_calls.append({"prefix": prefix, "namespace": namespace})
        for page in self.pages:
            yield SimpleNamespace(vectors=[SimpleNamespace(id=vector_id) for vector_id in page])

    def query(self, *, vector, top_k: int, namespace: str = "", include_metadata: bool = False):
        self._maybe_raise("query")
        self.query_calls.append(
            {"vector": vector, "top_k": top_k, "namespace": namespace, "include_metadata": include_metadata}
        )
        return SimpleNamespace(matches=self.matches)

    def fetch(self, *, ids, namespace: str = ""):
        self._maybe_raise("fetch")
        self.fetch_calls.append({"ids": list(ids), "namespace": namespace})
        return SimpleNamespace(
            vectors={i: SimpleNamespace(metadata=self.stored[i]) for i in ids if i in self.stored}
        )


class FakeReranker:
    """Puts the documents in a chosen order. `order` lists document indexes, best first; scores fall from 0.9."""

    def __init__(self, order: list[int] | None = None, raises: Exception | None = None) -> None:
        self.order, self.raises = order, raises
        self.calls: list[dict] = []

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
        self.calls.append({"query": query, "documents": list(documents), "top_k": top_k})
        if self.raises:
            raise self.raises
        order = self.order if self.order is not None else list(range(len(documents)))
        return [(index, round(0.9 - 0.1 * rank, 2)) for rank, index in enumerate(order[:top_k])]
