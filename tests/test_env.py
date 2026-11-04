from __future__ import annotations

import numpy as np

from habit.env.counterbalance import cell_for_run, draw_run_fixed
from habit.env.names import load_env_config, validated_pools
from habit.env.render import render_prompt
from habit.env.reward import reward_for_action
from habit.env.state import FORBIDDEN, contains_forbidden, generate_states, validate_training_state


def test_reward_values():
    cfg = load_env_config()
    assert cfg["reward"]["dock_a"] == 1.0
    assert cfg["reward"]["dock_b"] == 0.6
    fixed = draw_run_fixed(0, 1)
    assert reward_for_action(fixed, fixed.dock_a) == 1.0
    assert reward_for_action(fixed, fixed.dock_b) == 0.6


def test_pools_and_nonce():
    materials, products, customers = validated_pools()
    assert len(materials) >= 12
    assert len(products) >= 20
    assert len(customers) >= 30
    assert set(materials).isdisjoint(products)


def test_no_forbidden_and_stock_constraint():
    fixed = draw_run_fixed(0, 123)
    states = generate_states(fixed, 200, 7, training=True)
    for st in states:
        validate_training_state(st)
        assert st.s1 < st.q1 and st.s2 < st.q2
        rp = render_prompt(st, "A1", prompt_type="control")
        assert not contains_forbidden(rp.user)
        for w in FORBIDDEN:
            assert w not in rp.user.lower()


def test_counterbalance_cells():
    cfg = load_env_config()["counterbalance_cells"]
    seen = {}
    for idx in range(8):
        cell = cell_for_run(idx)
        assert cell == idx % 4
        f = draw_run_fixed(idx, 999)
        assert f.cell == cell
        lower = min(f.dock_a, f.dock_b)
        higher = max(f.dock_a, f.dock_b)
        expect_lower = cfg[cell]["dock_a_lower"]
        assert (f.dock_a == lower) == expect_lower
        seen[cell] = f
    assert len(seen) == 4


def test_seed_reproducibility():
    f1 = draw_run_fixed(2, 55)
    f2 = draw_run_fixed(2, 55)
    assert f1 == f2
    s1 = generate_states(f1, 10, 77, training=True)
    s2 = generate_states(f2, 10, 77, training=True)
    assert [x.state_id for x in s1] == [x.state_id for x in s2]
    assert [x.to_dict() for x in s1] == [x.to_dict() for x in s2]
