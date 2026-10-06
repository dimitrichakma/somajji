"""Free checks before anything is created online. Creates, changes and deletes nothing.

Run: uv run python -m scripts.preflight

It shows which keys are set (by name only, never the value), sends ONE short test sentence to Voyage to
confirm the model works and answers with 1024 numbers, and lists your Pinecone indexes (read only).
"""
import sys
from dataclasses import dataclass, field
from typing import Any

from pinecone import Pinecone, PineconeError

from app.config import get_settings
from app.rag.cli import normalize_dashes
from app.rag.embeddings import DIMENSION, VoyageEmbedder
from app.rag.errors import RetrievalError
from app.rag.rerank import VoyageReranker

FREE_PLAN_INDEX_LIMIT = 5
PROBE_TEXT = "Hello."  # one harmless word: a few tokens, well inside the free allowance


@dataclass(frozen=True)
class IndexInfo:
    name: str
    dimension: int | None
    metric: str | None
    ready: bool | None
    where: str  # "cloud/region", or "unknown"


@dataclass
class PreflightReport:
    target_index: str
    voyage_model: str
    keys: dict[str, bool]
    voyage_size: int | None = None
    voyage_error: str | None = None
    voyage_skipped: bool = False
    rerank_model: str = ""
    rerank_error: str | None = None
    rerank_skipped: bool = False
    indexes: list[IndexInfo] | None = None
    pinecone_error: str | None = None
    pinecone_skipped: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_keys(settings: Any) -> dict[str, bool]:
    """Which keys are set. Only yes or no: the values are never read into the report."""
    return {
        "VOYAGE_API_KEY": bool((settings.voyage_api_key or "").strip()),
        "PINECONE_API_KEY": bool((settings.pinecone_api_key or "").strip()),
    }


def probe_voyage(embedder: Any) -> tuple[int | None, str | None]:
    """Embed one short sentence. Returns (size of the answer, error text)."""
    try:
        size = len(embedder.embed_query(PROBE_TEXT))
    except RetrievalError as error:
        return None, str(error)
    if size != DIMENSION:
        return size, f"the answer has {size} numbers but the index needs {DIMENSION}"
    return size, None


def probe_rerank(reranker: Any) -> str | None:
    """Rerank two tiny documents. Returns None if it works, else the problem."""
    try:
        ranked = reranker.rerank("Hello.", ["Hello there.", "Goodbye."], top_k=1)
    except RetrievalError as error:
        return str(error)
    return None if len(ranked) == 1 else "the reranker returned no result"


def _where(index: Any) -> str:
    try:
        spec = index.spec
        serverless = spec.get("serverless") if isinstance(spec, dict) else getattr(spec, "serverless", None)
        if serverless is None:
            return "unknown"
        read = serverless.get if isinstance(serverless, dict) else lambda key: getattr(serverless, key, None)
        cloud, region = read("cloud"), read("region")
        return f"{cloud}/{region}" if cloud and region else "unknown"
    except Exception:  # the shape differs between Pinecone versions; this is only for display
        return "unknown"


def _ready(index: Any) -> bool | None:
    status = getattr(index, "status", None)
    return status.get("ready") if isinstance(status, dict) else getattr(status, "ready", None)


def list_indexes(pinecone: Any) -> list[IndexInfo]:
    try:
        return [
            IndexInfo(
                name=index.name,
                dimension=getattr(index, "dimension", None),
                metric=getattr(index, "metric", None),
                ready=_ready(index),
                where=_where(index),
            )
            for index in pinecone.list_indexes()
        ]
    except PineconeError as error:
        raise RetrievalError(f"Pinecone could not list your indexes: {error}") from error


def run_preflight(settings: Any, embedder: Any, pinecone: Any, reranker: Any = None) -> PreflightReport:
    report = PreflightReport(
        target_index=settings.pinecone_index, voyage_model=settings.voyage_model, keys=check_keys(settings)
    )
    for name, is_set in report.keys.items():
        if not is_set:
            report.problems.append(f"{name} is not set in .env")

    if report.keys["VOYAGE_API_KEY"]:
        report.voyage_size, report.voyage_error = probe_voyage(embedder)
        if report.voyage_error:
            report.problems.append(f"Voyage: {report.voyage_error}")
    else:
        report.voyage_skipped = True

    report.rerank_model = settings.voyage_rerank_model
    if not report.rerank_model:
        report.rerank_skipped = True  # switched off on purpose: not a problem
    elif report.keys["VOYAGE_API_KEY"] and reranker is not None:
        report.rerank_error = probe_rerank(reranker)
        if report.rerank_error:
            report.problems.append(f"Rerank: {report.rerank_error}")

    if report.keys["PINECONE_API_KEY"]:
        try:
            report.indexes = list_indexes(pinecone)
        except RetrievalError as error:
            report.pinecone_error = str(error)
            report.problems.append(str(error))
    else:
        report.pinecone_skipped = True
    return report


def format_report(report: PreflightReport) -> str:
    lines = ["Keys (set or not set; the values are never shown):"]
    lines += [f"  {name:18} {'set' if is_set else 'NOT set'}" for name, is_set in report.keys.items()]

    if report.voyage_skipped:
        lines.append("Voyage: skipped (VOYAGE_API_KEY not set)")
    elif report.voyage_error:
        lines.append(f"Voyage: model {report.voyage_model}: FAILED: {report.voyage_error}")
    else:
        lines.append(f"Voyage: model {report.voyage_model}: OK, the answer has {report.voyage_size} numbers")

    if not report.rerank_model:
        lines.append("Rerank: off (VOYAGE_RERANK_MODEL is empty)")
    elif report.rerank_error:
        lines.append(f"Rerank: model {report.rerank_model}: FAILED: {report.rerank_error}")
    elif report.keys.get("VOYAGE_API_KEY"):
        lines.append(f"Rerank: model {report.rerank_model}: OK")
    else:
        lines.append("Rerank: skipped (VOYAGE_API_KEY not set)")

    if report.pinecone_skipped:
        lines.append("Pinecone: skipped (PINECONE_API_KEY not set)")
    elif report.pinecone_error:
        lines.append(f"Pinecone: FAILED: {report.pinecone_error}")
    else:
        indexes = report.indexes or []
        lines.append(f"Pinecone: {len(indexes)} of {FREE_PLAN_INDEX_LIMIT} indexes used (the free plan allows {FREE_PLAN_INDEX_LIMIT})")
        for index in indexes:
            state = "ready" if index.ready else "not ready"
            lines.append(f"  {index.name}: {index.dimension} numbers, {index.metric}, {index.where}, {state}")
        exists = any(index.name == report.target_index for index in indexes)
        lines.append(
            f"  The index '{report.target_index}' already exists"
            if exists
            else f"  The index '{report.target_index}' does not exist yet"
        )

    lines.append("OK: nothing was created" if report.ok else "NOT OK: " + "; ".join(report.problems))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    normalize_dashes(sys.argv[1:] if argv is None else argv)  # no options yet; keeps the habit of one parser
    settings = get_settings()
    keys = check_keys(settings)
    embedder = VoyageEmbedder() if keys["VOYAGE_API_KEY"] else None
    pinecone = Pinecone(api_key=settings.pinecone_api_key) if keys["PINECONE_API_KEY"] else None
    reranker = VoyageReranker() if keys["VOYAGE_API_KEY"] and settings.voyage_rerank_model else None
    report = run_preflight(settings, embedder, pinecone, reranker)
    print(format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
