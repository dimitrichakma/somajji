"""Shared Claude models and the cross-cutting controls on every call.

Tiers
  fast_llm   - Haiku: classifier-style calls (routing, grading, planning,
               condensing, safety). Cheap and quick.
  smart_llm  - Sonnet: answer synthesis - the production quality-critical call.
  judge_llm  - Opus: offline LLM-as-judge, evaluation only, never on the
               request path. Different family than smart_llm so it isn't
               grading its own output.
  judge_fast_llm - Sonnet: the mechanical Ragas metrics (eval only).

Controls applied here so there is one place to reason about cost/limits:
  * Rate limiting - one shared client-side limiter paces all outbound calls
    (ANTHROPIC_RPS, default 2/s) to smooth bursts and avoid 429 storms.
  * Response caching - configure_llm_cache() installs a global exact-match
    cache (Postgres in prod, in-process otherwise; LLM_CACHE=0 disables).
  * Usage metering - the judge models carry the eval spend guard (EVAL_USAGE,
    hard ceiling). Production metering is per-request: the API passes a
    fresh UsageTracker as a callback and rolls the result into the daily budget.
  * max_tokens - every model is capped so a single runaway generation can't
    blow up a bill.

Model ids: FAST_MODEL / SMART_MODEL / JUDGE_MODEL / JUDGE_FAST_MODEL env vars.
"""
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.rate_limiters import InMemoryRateLimiter

from .cost.eval_budget import COST_TRACKER  # back-compat alias -> app.cost.usage.EVAL_USAGE
from .cost.llm_cache import configure_llm_cache

load_dotenv()

configure_llm_cache()

FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-4-5")
SMART_MODEL = os.getenv("SMART_MODEL", "claude-sonnet-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5")
JUDGE_FAST_MODEL = os.getenv("JUDGE_FAST_MODEL", "claude-sonnet-5")

# one shared limiter for every call this process makes. request-rate only
# (token-rate isn't in langchain-core); mainly there to keep parallel
# sub-question routing from hammering the API and tripping 429s.
_rps = float(os.getenv("ANTHROPIC_RPS", "2"))
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=_rps, check_every_n_seconds=0.1, max_bucket_size=max(int(_rps * 2), 1)
)

fast_llm = ChatAnthropic(
    model=FAST_MODEL, timeout=60, max_tokens=1024, rate_limiter=_rate_limiter
)

# Sonnet 5 thinks by default; synthesis is "answer from the given context", not
# a reasoning task, so turn thinking off - it only adds latency and cost here.
smart_llm = ChatAnthropic(
    model=SMART_MODEL, timeout=90, max_tokens=2048,
    thinking={"type": "disabled"}, rate_limiter=_rate_limiter,
)

# Opus 5 is $5/$25 per M tok (5x Sonnet). Bounded grading calls, not open
# reasoning, so thinking is off by default (JUDGE_THINKING=1 re-enables).
# EVAL_USAGE aborts the whole run at EVAL_MAX_USD.
_judge_thinking = (
    {"type": "adaptive"} if os.getenv("JUDGE_THINKING", "0") == "1" else {"type": "disabled"}
)
judge_llm = ChatAnthropic(
    model=JUDGE_MODEL, timeout=120, max_tokens=8192,
    thinking=_judge_thinking, rate_limiter=_rate_limiter, callbacks=[COST_TRACKER],
)

judge_fast_llm = ChatAnthropic(
    model=JUDGE_FAST_MODEL, timeout=90, max_tokens=8192,
    thinking={"type": "disabled"}, rate_limiter=_rate_limiter, callbacks=[COST_TRACKER],
)
