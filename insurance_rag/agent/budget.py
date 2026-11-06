"""Token budget for one run: fail before Groq's daily cap, not after."""

from __future__ import annotations

from insurance_rag.config import settings

__all__ = ["BudgetExceeded", "spend"]


class BudgetExceeded(RuntimeError):
    """Raised when a run would pass `settings.agent_token_budget`."""


def spend(used: int, more: int) -> int:
    """The new running total, or `BudgetExceeded` if it passes the budget."""
    total = used + more
    if total > settings.agent_token_budget:
        raise BudgetExceeded(f"{total} tokens passes the {settings.agent_token_budget} budget")
    return total
