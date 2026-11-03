"""OTLP/JSON encoding (the OpenTelemetry protocol's JSON mapping) for Argus spans."""

from __future__ import annotations

from typing import Any

from argus import __version__
from argus.sdk.tracer import SpanRecord


def any_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}  # OTLP JSON encodes int64 as a string
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, list | tuple):
        return {"arrayValue": {"values": [any_value(v) for v in value]}}
    return {"stringValue": str(value)}


def attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": any_value(v)} for k, v in values.items() if v is not None]


def encode(batch: list[tuple[SpanRecord, dict[str, Any]]]) -> dict[str, Any]:
    by_resource: dict[tuple, tuple[dict[str, Any], list[SpanRecord]]] = {}
    for record, resource in batch:
        key = tuple(sorted((k, str(v)) for k, v in resource.items()))
        by_resource.setdefault(key, (resource, []))[1].append(record)
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": attributes(resource)},
                "scopeSpans": [
                    {
                        "scope": {"name": "argus-sdk", "version": __version__},
                        "spans": [
                            {
                                "traceId": r.trace_id,
                                "spanId": r.span_id,
                                "parentSpanId": r.parent_span_id or "",
                                "name": r.name,
                                "kind": 1,
                                "startTimeUnixNano": str(r.start_ns),
                                "endTimeUnixNano": str(r.end_ns or r.start_ns),
                                "attributes": attributes(r.attributes),
                                "status": {"code": 2 if r.status == "error" else 1, "message": r.status_message},
                            }
                            for r in records
                        ],
                    }
                ],
            }
            for resource, records in by_resource.values()
        ]
    }
