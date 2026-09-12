"""Retry through provider rate limits. Shared by the generator and the eval judges.

Free tiers refuse long before they break, so a 429 is the expected path rather than an error.
A 413 is not retryable - the request is too large for the tier no matter how long you wait - so
it is deliberately left to raise.
"""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

MAX_ATTEMPTS = 6
MAX_BACKOFF = 90.0

#: Providers phrase the hint differently: Groq "Please try again in 10m45.408s", Google
#: "Please retry in 44.1s" and "retryDelay: 44s". Missing the minutes group is how a 10-minute
#: wait silently becomes a 2-second one.
_RETRY_AFTER_RE = re.compile(
    r"(?:try again|retry) in (?:(\d+)m)?([\d.]+)s|retryDelay['\"]?:\s*['\"]?(\d+)s"
)


#: A per-day budget does not refill inside a run. Retrying one only fails slower.
_DAILY_RE = re.compile(r"per day|TPD|RPD", re.IGNORECASE)


def is_daily_cap(exc: Exception) -> bool:
    return bool(_DAILY_RE.search(str(exc)))


def is_rate_limit(exc: Exception) -> bool:
    """Retryable limits only: 429 per-minute. Not 413 (too large), not a per-day budget."""
    if is_daily_cap(exc):
        return False
    return "429" in str(exc) or "ratelimit" in type(exc).__name__.lower()


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
