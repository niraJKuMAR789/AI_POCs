"""Span exporters. The HTTP exporter batches in a background thread and never blocks the app:
on overflow it drops (and counts) spans rather than applying backpressure to agent code."""

from __future__ import annotations

import atexit
import queue
import threading
import time
from typing import Any

import httpx

from argus.sdk.otlp import encode
from argus.sdk.tracer import SpanRecord


class InMemoryExporter:
    def __init__(self) -> None:
        self.spans: list[tuple[SpanRecord, dict[str, Any]]] = []

    def export(self, record: SpanRecord, resource: dict[str, Any]) -> None:
        self.spans.append((record, resource))

    def payload(self) -> dict[str, Any]:
        return encode(self.spans)

    def flush(self) -> None:
        pass


class HTTPExporter:
    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str | None = None,
        max_batch: int = 256,
        interval_s: float = 2.0,
        max_queue: int = 10_000,
        timeout_s: float = 5.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.url = endpoint.rstrip("/") + "/v1/traces"
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.max_batch, self.interval_s = max_batch, interval_s
        self._queue: queue.Queue[tuple[SpanRecord, dict[str, Any]]] = queue.Queue(max_queue)
        self._client = client or httpx.Client(timeout=timeout_s)
        self.dropped = 0
        self.sent = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="argus-exporter", daemon=True)
        self._thread.start()
        atexit.register(self.shutdown)

    def export(self, record: SpanRecord, resource: dict[str, Any]) -> None:
        try:
            self._queue.put_nowait((record, resource))
        except queue.Full:
            self.dropped += 1

    def _drain(self) -> list[tuple[SpanRecord, dict[str, Any]]]:
        batch: list[tuple[SpanRecord, dict[str, Any]]] = []
        while len(batch) < self.max_batch:
            try:
                batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _send(self, batch: list[tuple[SpanRecord, dict[str, Any]]]) -> None:
        for attempt in range(3):
            try:
                response = self._client.post(self.url, json=encode(batch), headers=self.headers)
                if response.status_code < 500:
                    self.sent += len(batch) if response.is_success else 0
                    self.dropped += 0 if response.is_success else len(batch)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2 * 2**attempt)
        self.dropped += len(batch)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self.interval_s)
            self.flush()

    def flush(self) -> None:
        while batch := self._drain():
            self._send(batch)

    def shutdown(self) -> None:
        self._stop.set()
        self.flush()
