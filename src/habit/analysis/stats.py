from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def sign_flip_test(deltas: Sequence[float]) -> float:
    """Exact sign-flip (two-sided) p-value for mean(deltas) when n <= 20."""
    x = np.asarray(list(deltas), dtype=float)
    n = len(x)
    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError("Exact sign-flip only for n <= 20")
    obs = abs(x.mean())
    count = 0
    total = 0
    for signs in product([-1.0, 1.0], repeat=n):
        m = abs((x * np.asarray(signs)).mean())
        total += 1
        if m >= obs - 1e-15:
            count += 1
    return count / total


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI using bootstrap-SE normal interval.

    Resamples paired ΔDS values with replacement (10k default). The interval is
    mean ± z_{1-α/2} * SD(bootstrap means), which recovers nominal coverage near
    95% for the preregistered n=16, noise sd=0.25 setting (percentile intervals
    undercover slightly at this n).
    """
    from scipy import stats as sps

    x = np.asarray(list(deltas), dtype=float)
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = x[idx].mean(axis=1)
    se = float(means.std(ddof=1))
    # t critical value matches nominal 95% coverage for normal paired means at small n.
    tcrit = float(sps.t.ppf(1 - alpha / 2, df=max(n - 1, 1)))
    m = float(x.mean())
    lo = m - tcrit * se
    hi = m + tcrit * se
    return m, lo, hi


def tost_equivalence(
    deltas: Sequence[float],
    *,
    low: float = -0.10,
    high: float = 0.10,
) -> dict[str, float | bool]:
    """TOST using normal approximation on the mean (simple preregistered form)."""
    from scipy import stats

    x = np.asarray(list(deltas), dtype=float)
    n = len(x)
    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    if se == 0:
        # Degenerate: pass iff mean inside bounds
        return {"mean": mean, "p_lower": 0.0 if mean > low else 1.0, "p_upper": 0.0 if mean < high else 1.0, "equivalent": low < mean < high}
    t_l = (mean - low) / se
    t_u = (mean - high) / se
    p_l = 1 - stats.t.cdf(t_l, df=n - 1)
    p_u = stats.t.cdf(t_u, df=n - 1)
    p = max(p_l, p_u)
    return {"mean": mean, "p_lower": float(p_l), "p_upper": float(p_u), "p": float(p), "equivalent": bool(p < 0.05)}


def holm_correction(p_values: Sequence[float]) -> list[float]:
    p = list(p_values)
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj
