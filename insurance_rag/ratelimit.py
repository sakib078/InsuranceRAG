"""Retry through provider rate limits, shared by the generator and the eval judges."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 6
MAX_BACKOFF = 90.0

#: Parses each provider's retry-after hint, including the minutes group.
_RETRY_AFTER_RE = re.compile(
    r"(?:try again|retry) in (?:(\d+)m)?([\d.]+)s|retryDelay['\"]?:\s*['\"]?(\d+)s"
)


#: A per-day budget does not refill inside a run. Retrying one only fails slower.
_DAILY_RE = re.compile(r"per day|TPD|RPD", re.IGNORECASE)


def is_daily_cap(exc: Exception) -> bool:
    return bool(_DAILY_RE.search(str(exc)))


#: A transient provider 5xx, slept through rather than raised.
_TRANSIENT_RE = re.compile(r"\b(500|502|503|504)\b")


def is_transient(exc: Exception) -> bool:
    return bool(_TRANSIENT_RE.search(str(exc)))


def is_rate_limit(exc: Exception) -> bool:
    """Worth waiting out: a per-minute 429 or a transient 5xx. Never a 413 or a per-day budget."""
    if is_daily_cap(exc):
        return False
    return "429" in str(exc) or "ratelimit" in type(exc).__name__.lower() or is_transient(exc)


def is_exhausted(exc: Exception) -> bool:
    """Out of budget on this provider: a daily cap, an unpaid quota, or a 429 that outlived retry."""
    text = str(exc)
    return is_daily_cap(exc) or "402" in text or "429" in text


def wait_for(exc: Exception, attempt: int) -> float:
    """Honour the provider's own retry hint when it gives one; otherwise back off and jitter."""
    match = _RETRY_AFTER_RE.search(str(exc))
    if match:
        minutes, seconds, delay = match.groups()
        hinted = float(delay) if delay else float(minutes or 0) * 60 + float(seconds)
        return min(hinted + 1.0, MAX_BACKOFF)
    return min(2.0**attempt + random.uniform(0, 1), MAX_BACKOFF)


def with_retry(call: Callable[[], T]) -> T:
    """Run `call`, sleeping through rate limits. Anything else raises on the first attempt."""
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call()
        except Exception as exc:
            if not is_rate_limit(exc) or attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(wait_for(exc, attempt))
    raise RuntimeError("unreachable")
