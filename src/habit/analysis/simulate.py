from __future__ import annotations

from typing import Any

import numpy as np

from habit.analysis.stats import paired_bootstrap_ci, sign_flip_test


def simulate_delta_ds(
    n: int = 16,
    true_delta: float = 0.20,
    noise_sd: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return true_delta + rng.normal(0.0, noise_sd, size=n)


def coverage_simulation(
    *,
    n: int = 16,
    true_delta: float = 0.20,
    noise_sd: float = 0.25,
    n_sims: int = 1000,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    covered = 0
    for i in range(n_sims):
        deltas = true_delta + rng.normal(0.0, noise_sd, size=n)
        _, lo, hi = paired_bootstrap_ci(deltas, n_resamples=n_resamples, seed=int(rng.integers(0, 1e9)))
        if lo <= true_delta <= hi:
            covered += 1
    rate = covered / n_sims
    return {"coverage": rate, "n_sims": n_sims, "true_delta": true_delta, "ok_93_97": 0.93 <= rate <= 0.97}


def null_false_positive_simulation(
    *,
    n: int = 16,
    noise_sd: float = 0.25,
    n_sims: int = 2000,
    alpha: float = 0.05,
    seed: int = 1,
) -> dict[str, Any]:
    """False positive rate of two-sided sign-flip test under null (true ΔDS=0)."""
    rng = np.random.default_rng(seed)
    fp = 0
    for i in range(n_sims):
        deltas = rng.normal(0.0, noise_sd, size=n)
        p = sign_flip_test(deltas)
        if p < alpha:
            fp += 1
    rate = fp / n_sims
    return {"fpr": rate, "n_sims": n_sims, "ok_3_7": 0.03 <= rate <= 0.07}
