from __future__ import annotations

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import load_test_lines, render_prompt
from habit.env.state import FORBIDDEN, State, contains_forbidden
from habit.test.battery import expected_prompt_count, generate_battery_items, generate_competence_items


def test_battery_size_and_determinism():
    fixed = draw_run_fixed(0, 1)
    a = generate_battery_items(fixed)
    b = generate_battery_items(fixed)
    assert len(a) == expected_prompt_count() == 600
    assert [x["prompt_sha256"] for x in a] == [x["prompt_sha256"] for x in b]
    assert len(generate_competence_items()) == 40


def test_test_only_lines_and_control_clean():
    fixed = draw_run_fixed(1, 2)
    items = generate_battery_items(fixed)
    lines = load_test_lines()
    for it in items:
        user = it["messages"][1]["content"].lower()
        if it["type"] == "control":
            for w in FORBIDDEN:
                assert w not in user
            assert "thrown away" not in user
            assert "repainted" not in user
            assert "cracked" not in user
            assert "automatically" not in user
        elif it["type"] == "deval_M1":
            assert "thrown away" in user
            assert fixed.M1.lower() in user
        elif it["type"] == "deval_M2":
            assert "thrown away" in user
        elif it["type"] == "sham":
            assert "repainted" in user
        elif it["type"] == "spoil_M1":
            assert "cracked" in user
        elif it["type"] == "degrade_M1":
            assert "automatically" in user
