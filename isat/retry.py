"""Retry logic with exponential backoff for flaky operations.

Used for:
  - Session creation (may fail on first attempt due to GPU init)
  - Benchmark runs (may fail due to thermal throttle or OOM)
  - Network operations (Prometheus push, API calls)
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type

log = logging.getLogger("isat.retry")


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable:
    """Decorator for retrying functions with exponential backoff.

    Usage:
        @retry(max_attempts=3, base_delay=2.0)
        def create_session():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_attempts:
                        log.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, max_attempts, e,
                        )
                        raise

                    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random())

                    log.warning(
                        "%s attempt %d/%d failed: %s -- retrying in %.1fs",
                        func.__name__, attempt, max_attempts, e, delay,
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper
    return decorator


class RetryContext:
    """Context manager for retry loops (non-decorator usage).

    Usage:
        with RetryContext(max_attempts=3) as ctx:
            for attempt in ctx:
                try:
                    result = do_something()
                    ctx.success(result)
                except Exception as e:
                    ctx.fail(e)
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff_factor = backoff_factor
        self._result: Any = None
        self._succeeded = False
        self._attempt = 0
        self._last_error: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def __iter__(self):
        for self._attempt in range(1, self.max_attempts + 1):
            if self._succeeded:
                break
            yield self._attempt
            if not self._succeeded and self._attempt < self.max_attempts:
                delay = self.base_delay * (self.backoff_factor ** (self._attempt - 1))
                time.sleep(delay * (0.5 + random.random()))

    def success(self, result: Any = None) -> None:
        self._result = result
        self._succeeded = True

    def fail(self, error: Exception) -> None:
        self._last_error = error
        if self._attempt >= self.max_attempts:
            raise error

    @property
    def result(self) -> Any:
        return self._result
