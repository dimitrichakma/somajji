"""Search the WHO guides. This is what the guide agent (Phase 3) calls.

Search finds about 20 candidate chunks by similarity, then (when a reranker is configured) a reranker re-scores
them and the best `k` are returned. `score` is then the reranker's relevance score, not the similarity.
If the reranker fails, the search order is used instead and a warning is logged.

`retrieve` always returns the nearest chunks, even when none of them is a good answer. Deciding
"nothing relevant, so say so" is the relevance grader's job in Phase 3, not this function's.

The question is sent to Voyage, so it must already be cleaned of names, phone numbers and places
by the Phase 2 guardrail. It is never logged here.
"""
import logging
from functools import lru_cache

from pydantic import ValidationError

from app.config import get_settings

from .embeddings import Embedder, VoyageEmbedder
from .errors import RetrievalError
from .rerank import Reranker, VoyageReranker
from .store import PineconeStore, VectorMatch, VectorStore
from .types import RetrievedChunk


logger = logging.getLogger(__name__)

CANDIDATES = 20  # how many search results the reranker reads


@lru_cache
def _default_embedder() -> Embedder:
    return VoyageEmbedder()


@lru_cache
def _default_store() -> VectorStore:
    return PineconeStore(get_settings().pinecone_index)


@lru_cache
def _default_reranker() -> Reranker | None:
    """None when reranking is switched off (VOYAGE_RERANK_MODEL is empty)."""
    return VoyageReranker() if get_settings().voyage_rerank_model.strip() else None


def _to_chunk(match: VectorMatch) -> RetrievedChunk:
    metadata = match["metadata"]
    try:
        return RetrievedChunk(
            text=metadata.get("text", ""),
            source=metadata.get("source", ""),
            page=metadata.get("page"),  # type: ignore[arg-type]  # validated below
            pdf_page=metadata.get("pdf_page"),  # type: ignore[arg-type]
            section=metadata.get("section", ""),
            score=match["score"],
        )
    except ValidationError as error:
        # A result without a citation is useless ("no citation, no answer"), so fail loudly.
        raise RetrievalError(
            f"Stored vector {match['id']} is missing its source, page or text. Re-run the ingest."
        ) from error


def retrieve(
    question: str,
    k: int = 5,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    reranker: Reranker | None = None,
    candidates: int = CANDIDATES,
) -> list[RetrievedChunk]:
    """The k best chunks for the question, best first. Raises RetrievalError if search cannot run."""
    if not question.strip():
        raise RetrievalError("Cannot search with an empty question")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    embedder = embedder if embedder is not None else _default_embedder()
    store = store if store is not None else _default_store()
    reranker = reranker if reranker is not None else _default_reranker()

    vector = embedder.embed_query(question)  # if this fails, the store is never queried
    wanted = max(k, candidates) if reranker is not None else k
    found = [_to_chunk(match) for match in store.query(vector, wanted)]
    # Do not rely on the store's order or count: sort and cut here.
    found = sorted(found, key=lambda chunk: chunk.score, reverse=True)[:wanted]
    if reranker is None or len(found) < 2:
        return found[:k]

    try:
        ranked = reranker.rerank(question, [chunk.text for chunk in found], top_k=k)
    except RetrievalError as error:
        logger.warning("Reranking failed, using the search order instead: %s", error)  # never log the question
        return found[:k]
    return [found[index].model_copy(update={"score": score}) for index, score in ranked][:k]
