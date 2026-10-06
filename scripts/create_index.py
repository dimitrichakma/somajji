"""Create the Pinecone index for the WHO guides. By default it only SHOWS what it would create.

Run: uv run python -m scripts.create_index            (prints the plan, creates nothing)
     uv run python -m scripts.create_index --create   (creates it, after Dimitri has said yes)

An index is an online resource: add it to docs/TEARDOWN.md first. On the Pinecone free plan (Starter) an index
costs nothing, but only AWS us-east-1 is allowed and there is room for 5 indexes.
"""
import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pinecone import ApiError, Pinecone, PineconeError, ServerlessSpec

from app.config import get_settings
from app.rag.cli import normalize_dashes
from app.rag.embeddings import DIMENSION
from app.rag.errors import RetrievalError

FREE_PLAN_INDEX_LIMIT = 5
WAIT_ATTEMPTS = 60
WAIT_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class IndexPlan:
    name: str
    dimension: int = DIMENSION  # must match what Voyage returns
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"  # the only region the free plan allows
    deletion_protection: str = "disabled"  # so the index can be deleted easily when the demo ends


def describe_plan(plan: IndexPlan) -> str:
    protection = "off" if plan.deletion_protection == "disabled" else "on"
    return "\n".join(
        [
            "Index to create:",
            f"  name:                 {plan.name}",
            f"  type:                 serverless on {plan.cloud}, region {plan.region}",
            f"  vector size:          {plan.dimension} numbers",
            f"  similarity metric:    {plan.metric}",
            f"  deletion protection:  {protection}",
        ]
    )


def _is_ready(index: Any) -> bool:
    status = getattr(index, "status", None)
    return bool(status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False))


def ensure_index(
    pinecone: Any,
    plan: IndexPlan,
    create: bool,
    sleep: Callable[[float], None] = time.sleep,
    wait_attempts: int = WAIT_ATTEMPTS,
    wait_interval: float = WAIT_INTERVAL_SECONDS,
    spec_class: Any = ServerlessSpec,
) -> str:
    """Returns "planned", "exists" or "created". Creates only when `create` is true and the index is missing."""
    try:
        if pinecone.has_index(plan.name):
            existing = pinecone.describe_index(plan.name)
            if existing.dimension == plan.dimension and existing.metric == plan.metric:
                return "exists"
            raise RetrievalError(
                f"An index named '{plan.name}' already exists but with {existing.dimension} numbers and metric "
                f"'{existing.metric}' (this project needs {plan.dimension} and '{plan.metric}'). Nothing was changed. "
                "Choose another PINECONE_INDEX in .env, or delete that index yourself."
            )
        if not create:
            return "planned"

        pinecone.create_index(
            name=plan.name,
            spec=spec_class(cloud=plan.cloud, region=plan.region),
            dimension=plan.dimension,
            metric=plan.metric,
            deletion_protection=plan.deletion_protection,
            timeout=-1,  # return at once; we wait ourselves so we can give up cleanly
        )
        for _ in range(wait_attempts):
            if _is_ready(pinecone.describe_index(plan.name)):
                return "created"
            sleep(wait_interval)
    except ApiError as error:
        hint = (
            f" The free plan allows {FREE_PLAN_INDEX_LIMIT} indexes: delete one you no longer need, or upgrade."
            if getattr(error, "status_code", None) == 403
            else ""
        )
        raise RetrievalError(f"Pinecone refused: {error}.{hint}") from error
    except PineconeError as error:
        raise RetrievalError(f"Pinecone failed: {error}") from error
    raise RetrievalError(f"The index '{plan.name}' was created but is not ready yet. Run this command again in a minute.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the Pinecone index (prints the plan unless --create).")
    parser.add_argument("--create", action="store_true", help="really create the index (asks Pinecone to do it)")
    parser.add_argument("--cloud", default="aws")
    parser.add_argument("--region", default="us-east-1", help="the free plan allows only us-east-1")
    return parser.parse_args(normalize_dashes(sys.argv[1:] if argv is None else argv))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    if not (settings.pinecone_api_key or "").strip():
        print("ERROR: PINECONE_API_KEY is not set. Add it to .env.", file=sys.stderr)
        return 1
    plan = IndexPlan(name=settings.pinecone_index, cloud=args.cloud, region=args.region)
    print(describe_plan(plan))
    try:
        result = ensure_index(Pinecone(api_key=settings.pinecone_api_key), plan, create=args.create)
    except RetrievalError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        {
            "planned": "Nothing was created. Run again with --create to create it.",
            "exists": "The index already exists with the right settings. Nothing was changed.",
            "created": "The index was created and is ready.",
        }[result]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
