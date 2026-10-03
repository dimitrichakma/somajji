"""The files copied from the chatbot import cleanly and follow Somajji's settings."""
import importlib

import pytest

from app.config import get_settings


@pytest.mark.parametrize(
    "module",
    ["app.llm", "app.cost.usage", "app.cost.eval_budget", "app.cost.day_budget", "app.cost.llm_cache"],
)
def test_module_imports(module):
    importlib.import_module(module)


def test_no_neo4j_or_tavily_left_behind():
    import pathlib

    for path in pathlib.Path("app").rglob("*.py"):
        text = path.read_text().lower()
        assert "neo4j" not in text and "tavily" not in text, path


def test_eval_guard_uses_the_configured_ceiling():
    from app.cost.usage import EVAL_USAGE

    assert EVAL_USAGE.max_usd == get_settings().eval_max_usd


def test_model_ids_are_the_latest_and_priced():
    from app import llm
    from app.cost.usage import _price

    assert llm.SMART_MODEL == "claude-sonnet-5-5"
    assert llm.JUDGE_MODEL == "claude-opus-5-5"
    assert _price("claude-opus-5-5") == (4.0, 20.0)  # not the older Opus 5 price
    assert _price("claude-sonnet-5-5") == (2.0, 10.0)
    assert _price("claude-haiku-4-5-20251001") == (1.0, 5.0)


def test_no_model_sends_the_rejected_disabled_thinking_setting():
    from app import llm

    for model in (llm.fast_llm, llm.smart_llm, llm.judge_llm, llm.judge_fast_llm):
        assert (model.thinking or {}).get("type") != "disabled"


def test_day_budget_status_reads_postgres():
    from app.cost import day_budget

    day_budget.setup()
    spent, limit, is_over = day_budget.status()
    assert limit == get_settings().daily_budget_usd
    assert spent >= 0.0
    assert isinstance(is_over, bool)
