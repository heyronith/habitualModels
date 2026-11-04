from __future__ import annotations

import numpy as np

from habit.train.rl import SET_A_TEMPLATES, sample_set_a_template


def test_rl_samples_only_set_a_templates():
    """F6: uniform over A1–A3; never Set B."""
    rng = np.random.default_rng(0)
    seen = {sample_set_a_template(rng) for _ in range(300)}
    assert seen == set(SET_A_TEMPLATES)
    assert "B1" not in seen and "B2" not in seen
