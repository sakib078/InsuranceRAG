"""Token budget for the whole process: fail before Groq's daily cap, not after."""

from __future__ import annotations

from insurance_rag.config import settings

__all__ = ["BudgetExceeded", "spend"]

_spent = 0


class BudgetExceeded(RuntimeError):
    """Raised once the process has spent past `settings.agent_token_budget`."""


def spend(more: int) -> None:
    """Add to the process total, or raise `BudgetExceeded` if it passes the budget."""
    global _spent
    _spent += more
    if _spent > settings.agent_token_budget:
        raise BudgetExceeded(f"{_spent} tokens passes the {settings.agent_token_budget} budget")
