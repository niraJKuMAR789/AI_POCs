"""PII and secret detection with redaction, applied before anything is persisted."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def _luhn_ok(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    return total % 10 == 0


def _ssn_ok(match: str) -> bool:
    area, group, serial = match.split("-")
    return area not in ("000", "666") and not area.startswith("9") and group != "00" and serial != "0000"


@dataclass(frozen=True)
class Detector:
    kind: str
    pattern: re.Pattern[str]
    validate: object = None  # optional callable(str) -> bool


DETECTORS = [
    # Secrets first: they are the most damaging leak and must not be partially masked by other patterns.
    Detector(
        "PRIVATE_KEY",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
    Detector(
        "API_KEY",
        re.compile(
            r"\b(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}|nvapi-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|"
            r"xox[bpas]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16})\b"
        ),
    ),
    Detector("BEARER_TOKEN", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/-]{20,}=*")),
    Detector("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    Detector("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), _ssn_ok),
    Detector("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    Detector("PHONE", re.compile(r"(?<!\d)(?:\+1[ .-]?)?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)")),
    # Not part of a longer dotted run (e.g. version "1.2.3.4.5").
    Detector(
        "IP_ADDRESS",
        re.compile(
            r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?![\d]|\.\d)"
        ),
    ),
]


def scan(text: str) -> list[tuple[str, int, int]]:
    """Non-overlapping (kind, start, end) matches; earlier detectors win overlaps."""
    taken: list[tuple[int, int]] = []
    found = []
    for det in DETECTORS:
        for m in det.pattern.finditer(text):
            if det.validate is not None and not det.validate(m.group(0)):  # type: ignore[operator]
                continue
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((det.kind, m.start(), m.end()))
    return sorted(found, key=lambda x: x[1])


def redact(text: str) -> tuple[str, Counter[str]]:
    counts: Counter[str] = Counter()
    out, cursor = [], 0
    for kind, start, end in scan(text):
        out += [text[cursor:start], f"[REDACTED:{kind}]"]
        counts[kind] += 1
        cursor = end
    out.append(text[cursor:])
    return "".join(out), counts
