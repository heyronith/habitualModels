from __future__ import annotations

from habit.analysis.simulate import coverage_simulation, null_false_positive_simulation
from habit.analysis.stats import holm_correction, paired_bootstrap_ci, sign_flip_test, tost_equivalence


def test_sign_flip_and_bootstrap_smoke():
    deltas = [0.2] * 8 + [-0.05] * 8
    p = sign_flip_test(deltas)
    assert 0.0 <= p <= 1.0
    mean, lo, hi = paired_bootstrap_ci(deltas, n_resamples=1000, seed=0)
    assert lo <= mean <= hi
    tost = tost_equivalence(deltas)
    assert "equivalent" in tost
    adj = holm_correction([0.01, 0.04, 0.20])
    assert adj[0] <= adj[1] <= 1.0


def test_planted_delta_coverage():
    # Full acceptance numbers are heavy; use full counts as specified.
    cov = coverage_simulation(
        n=16, true_delta=0.20, noise_sd=0.25, n_sims=1000, n_resamples=10_000, seed=0
    )
    assert cov["ok_93_97"], cov


def test_null_fpr():
    res = null_false_positive_simulation(n=16, noise_sd=0.25, n_sims=2000, seed=1)
    assert res["ok_3_7"], res
