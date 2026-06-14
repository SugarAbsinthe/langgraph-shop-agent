"""Structured request tracing for the ShopAgent backend.

Provides a contextvar-based request_id that flows through the entire
request lifecycle, and a RequestLogger for consistent structured output.

Log format uses key=value pairs for easy grep/awk parsing:
  [request_id] event_name key1=val1 key2=val2 ...
"""

import contextvars
import logging
import time
import uuid

_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="")
_node_timers: contextvars.ContextVar[dict] = contextvars.ContextVar("node_timers", default={})

_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("shopagent")
logger.setLevel(logging.INFO)
logger.handlers = [_stream_handler]
logger.propagate = False


def set_request_id(rid: str = "") -> str:
    """Set the request_id for the current context. Returns the id."""
    rid = rid or uuid.uuid4().hex[:12]
    _request_id_ctx.set(rid)
    return rid


def get_request_id() -> str:
    """Get the current request_id, or empty string if not set."""
    return _request_id_ctx.get("")


def _format_kv(**kwargs) -> str:
    """Format key=value pairs, skipping None values."""
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}s")
        else:
            parts.append(f"{k}={v}")
    return " ".join(parts)


def log(event: str, **kwargs):
    """Log a structured event with the current request_id."""
    rid = get_request_id()
    prefix = f"[{rid}]" if rid else ""
    msg = f"{prefix} {event} " + _format_kv(**kwargs)
    logger.info(msg.rstrip())


class Timer:
    """Context manager for timing a code block. Logs event on exit."""

    def __init__(self, event: str, **kwargs):
        self.event = event
        self.kwargs = kwargs
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        log(self.event, duration=elapsed, **self.kwargs)
