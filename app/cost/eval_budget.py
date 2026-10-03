"""Back-compat shim - the spend guard moved to app.cost.usage and became the
general per-call usage meter. Import from app.cost.usage in new code.
"""
from .usage import EVAL_USAGE as COST_TRACKER  # noqa: F401
from .usage import BudgetExceeded, UsageTracker  # noqa: F401

CostTracker = UsageTracker  # old name
