"""Token + cost accounting for every Claude call, prod and eval.

`UsageTracker` is a LangChain callback that sums token usage per model as calls
happen and prices it in USD. Two ways to use it:

* **Process-wide guard (eval).** `EVAL_USAGE` is a module singleton with a hard
  ceiling (`EVAL_MAX_USD`, default $2). It raises `BudgetExceeded` mid-run the
  moment accumulated spend crosses the line, so a runaway metric or retry storm
  can't quietly run up a bill. Attached to the judge models in app/llm.py.

* **Per-request meter (prod).** Make a fresh `UsageTracker()` (no ceiling) per
  request, pass it as a callback, then read `.by_model` afterward to log
  the request's cost and roll it into the persistent daily budget
  (app/cost/day_budget.py).

Pricing includes the prompt-cache multipliers, so cache hits are counted at
0.1x input - the numbers track real spend when caching is on.
"""
import threading

from langchain_core.callbacks import BaseCallbackHandler

from app.config import get_settings

# USD per 1M tokens: (input, output). https://platform.claude.com/docs pricing.
# Matched by substring in order, so a more specific id must come before its prefix
# ("claude-opus-5-5" before "claude-opus-5").
MODEL_PRICING = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
CACHE_READ_MULT = 0.1     # cache hit: 0.1x base input
CACHE_WRITE_MULT = 1.25   # 5-minute cache write: 1.25x base input


class BudgetExceeded(BaseException):
    """Raised mid-run once accumulated spend passes a tracker's ceiling.

    Subclasses BaseException (not Exception) on purpose: Ragas wraps metric jobs
    in tenacity retry-on-Exception and a catch-all, so a plain Exception here
    would be retried 10x then swallowed. BaseException slips past both.
    """


def _price(model_id):
    for known, pricing in MODEL_PRICING.items():
        if known in (model_id or ""):
            return pricing
    return None


class UsageTracker(BaseCallbackHandler):
    raise_error = True  # let BudgetExceeded propagate out of the callback

    def __init__(self, max_usd=None):
        self.max_usd = max_usd
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            # model_id -> {input, output, cache_read, cache_write, calls}
            self.by_model = {}
            self.calls = 0
            self._tripped = False

    # ------------------------------------------------------------------ usage

    def _record(self, model_id, in_tok, out_tok, cache_read, cache_write):
        with self._lock:
            row = self.by_model.setdefault(
                model_id, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "calls": 0}
            )
            row["input"] += in_tok
            row["output"] += out_tok
            row["cache_read"] += cache_read
            row["cache_write"] += cache_write
            row["calls"] += 1
            self.calls += 1
            if self.max_usd is not None and not self._tripped:
                cost = self._cost_usd_locked()
                if cost > self.max_usd:
                    self._tripped = True
                    raise BudgetExceeded(
                        f"spend ${cost:.2f} exceeded ceiling ${self.max_usd:.2f} "
                        f"after {self.calls} calls"
                    )

    def on_llm_end(self, response, **kwargs):
        model_id = "unknown"
        in_tok = out_tok = cache_read = cache_write = 0
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    msg = getattr(gen, "message", None)
                    meta = getattr(msg, "usage_metadata", None) if msg else None
                    if meta:
                        in_tok += meta.get("input_tokens", 0)
                        out_tok += meta.get("output_tokens", 0)
                        details = meta.get("input_token_details") or {}
                        cache_read += details.get("cache_read", 0) or 0
                        cache_write += details.get("cache_creation", 0) or 0
                    rmeta = getattr(msg, "response_metadata", {}) if msg else {}
                    model_id = rmeta.get("model_name") or rmeta.get("model") or model_id
            if in_tok == 0 and out_tok == 0:
                usage = (response.llm_output or {}).get("usage", {})
                in_tok = usage.get("input_tokens", 0)
                out_tok = usage.get("output_tokens", 0)
                model_id = (response.llm_output or {}).get("model", model_id)
        except Exception:
            pass
        # usage_metadata reports cache_read / cache_creation as part of input_tokens
        # on some versions and separately on others; treat `input` as the
        # full-price remainder so we never double count.
        in_tok = max(in_tok - cache_read - cache_write, 0)
        self._record(model_id, in_tok, out_tok, cache_read, cache_write)

    # ------------------------------------------------------------------- cost

    def _cost_usd_locked(self):
        total = 0.0
        for model_id, r in self.by_model.items():
            pricing = _price(model_id)
            if pricing is None:
                continue
            in_rate, out_rate = pricing
            total += (
                r["input"] / 1e6 * in_rate
                + r["cache_read"] / 1e6 * in_rate * CACHE_READ_MULT
                + r["cache_write"] / 1e6 * in_rate * CACHE_WRITE_MULT
                + r["output"] / 1e6 * out_rate
            )
        return total

    @property
    def cost_usd(self):
        with self._lock:
            return self._cost_usd_locked()

    def snapshot(self):
        """Plain dict, safe to serialize / hand to the daily budget."""
        with self._lock:
            return {
                "cost_usd": round(self._cost_usd_locked(), 6),
                "calls": self.calls,
                "by_model": {m: dict(r) for m, r in self.by_model.items()},
            }

    def report(self):
        with self._lock:
            ceiling = f", ceiling ${self.max_usd:.2f}" if self.max_usd is not None else ""
            lines = [f"spend: ${self._cost_usd_locked():.4f}  ({self.calls} calls{ceiling})"]
            for model_id, r in sorted(self.by_model.items()):
                priced = "" if _price(model_id) else "  [unpriced]"
                extra = ""
                if r["cache_read"] or r["cache_write"]:
                    extra = f", cache {r['cache_read']:,} read / {r['cache_write']:,} write"
                lines.append(
                    f"  {model_id}: {r['calls']} calls, "
                    f"{r['input']:,} in / {r['output']:,} out{extra}{priced}"
                )
            return "\n".join(lines)


def friendly_api_error(exc):
    """One clean line for an Anthropic API failure, or None if `exc` is something
    else. Lets the eval scripts exit with a readable message instead of dumping
    a langchain/anthropic traceback (the usual cause is a spent credit balance).
    """
    try:
        import anthropic
    except ImportError:
        return None
    if not isinstance(exc, anthropic.APIError):
        return None
    msg = str(getattr(exc, "message", "") or exc)
    if "credit balance is too low" in msg:
        return ("Anthropic API: out of credits. Top up at "
                "console.anthropic.com -> Plans & Billing, then re-run. "
                "Nothing was scored, so nothing was charged for this run.")
    if "rate_limit" in msg.lower() or " 429" in msg:
        return f"Anthropic API rate limited - wait a bit and re-run. ({msg})"
    if "overloaded" in msg.lower() or " 529" in msg:
        return f"Anthropic API overloaded - re-run shortly. ({msg})"
    return f"Anthropic API error: {msg}"


# process-wide guard for eval runs (hard ceiling); attached to the judge models
EVAL_USAGE = UsageTracker(max_usd=get_settings().eval_max_usd)
