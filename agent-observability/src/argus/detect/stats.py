"""Small, dependency-free robust statistics used by the detectors."""

from __future__ import annotations

import math
from collections.abc import Sequence


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def median(values: Sequence[float]) -> float:
    return quantile(values, 0.5)


def robust_z(x: float, baseline: Sequence[float], *, rel_floor: float = 0.05) -> tuple[float, float]:
    """(z, center) using median and MAD; the scale is floored so flat baselines don't explode."""
    center = median(baseline)
    mad = median([abs(b - center) for b in baseline]) * 1.4826
    scale = max(mad, rel_floor * abs(center), 1e-9)
    return (x - center) / scale, center


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed exactly in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = min(max(p, 1e-9), 1 - 1e-9)
    logs = [_log_binom_pmf(i, n, p) for i in range(k, n + 1)]
    m = max(logs)
    return min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs))


def poisson_sf(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    lam = max(lam, 1e-9)
    cdf, term = 0.0, math.exp(-lam)
    for i in range(k):
        cdf += term
        term *= lam / (i + 1)
    return max(0.0, 1.0 - cdf)
