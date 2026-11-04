"""Replay clock: run a day of real traffic in minutes.

Every request really executes: real LangGraph runs, real SQL, real rules, real span durations. Only the
*start time* of each trace is placed on a virtual timeline, so six hours of traffic at production-like
rates can be replayed in a few minutes and Argus's 10-minute detection windows fill up as they would live.

A per-request offset (virtual start - wall-clock start) is held in a context variable and added to every
span of that request at export time. Durations are never touched: a slow reranker is slow in wall-clock
time too.
"""

from __future__ import annotations

import contextvars
import dataclasses
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from argus.sdk.tracer import SpanRecord

MINUTE_NS = 60 * 10**9
_offset_ns: contextvars.ContextVar[int] = contextvars.ContextVar("mission_offset_ns", default=0)


@dataclass
class Timeline:
    start_ns: int
    now_min: float = 0.0

    def at(self, minute: float) -> int:
        return self.start_ns + int(minute * MINUTE_NS)

    @contextmanager
    def request_at(self, minute: float) -> Iterator[None]:
        """Run one request as if it started at `minute` on the virtual timeline."""
        self.now_min = minute
        token = _offset_ns.set(self.at(minute) - time.time_ns())
        try:
            yield
        finally:
            _offset_ns.reset(token)


class ShiftingExporter:
    """Wraps an exporter and moves each span onto the virtual timeline before it is exported."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def export(self, record: SpanRecord, resource: dict[str, Any]) -> None:
        offset = _offset_ns.get()
        shifted = dataclasses.replace(
            record,
            start_ns=record.start_ns + offset,
            end_ns=(record.end_ns or record.start_ns) + offset,
            attributes=dict(record.attributes),
        )
        self.inner.export(shifted, resource)

    def flush(self) -> None:
        self.inner.flush()
