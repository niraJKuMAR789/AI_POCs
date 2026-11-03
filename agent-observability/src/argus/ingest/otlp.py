"""OTLP/JSON decoding → flat span dicts. Accepts output of any OpenTelemetry SDK, not just Argus."""

from __future__ import annotations

import base64
from typing import Any


def _value(v: dict[str, Any]) -> Any:
    if "stringValue" in v:
        return v["stringValue"]
    if "intValue" in v:
        return int(v["intValue"])
    if "doubleValue" in v:
        return float(v["doubleValue"])
    if "boolValue" in v:
        return bool(v["boolValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {kv["key"]: _value(kv["value"]) for kv in v["kvlistValue"].get("values", [])}
    if "bytesValue" in v:
        return v["bytesValue"]
    return None


def _attrs(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {item["key"]: _value(item.get("value", {})) for item in items or []}


def _id(raw: str | None, nbytes: int) -> str | None:
    """OTLP JSON ids are hex; some exporters send base64 (protobuf-JSON mapping). Normalize to hex."""
    if not raw:
        return None
    if len(raw) == nbytes * 2 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return raw.lower()
    try:
        decoded = base64.b64decode(raw)
    except ValueError:
        return raw
    return decoded.hex() if len(decoded) == nbytes else raw


def decode(payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans = []
    for resource_spans in payload.get("resourceSpans", []):
        resource = _attrs(resource_spans.get("resource", {}).get("attributes"))
        for scope_spans in resource_spans.get("scopeSpans", []) or resource_spans.get(
            "instrumentationLibrarySpans", []
        ):
            for s in scope_spans.get("spans", []):
                status = s.get("status") or {}
                code = status.get("code", 0)
                spans.append(
                    {
                        "trace_id": _id(s.get("traceId"), 16),
                        "span_id": _id(s.get("spanId"), 8),
                        "parent_span_id": _id(s.get("parentSpanId"), 8),
                        "name": s.get("name", ""),
                        "start_ns": int(s.get("startTimeUnixNano", 0)),
                        "end_ns": int(s.get("endTimeUnixNano", 0) or s.get("startTimeUnixNano", 0)),
                        "status": "error" if code in (2, "STATUS_CODE_ERROR") else "ok",
                        "status_message": status.get("message", ""),
                        "attributes": _attrs(s.get("attributes")),
                        "resource": resource,
                    }
                )
    return spans
