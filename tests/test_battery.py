from __future__ import annotations

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import load_test_lines, render_prompt
from habit.env.state import FORBIDDEN, derive_state_for_type
from habit.test.battery import (
    DOCK_REPLY_SENTENCE,
    _append_probe_to_user,
    expected_prompt_count,
    generate_battery_items,
    generate_competence_items,
    load_probes,
    render_probe_messages,
)


def test_battery_size_and_determinism():
    fixed = draw_run_fixed(0, 1)
    a = generate_battery_items(fixed, variant="G")
    b = generate_battery_items(fixed, variant="G")
    assert len(a) == expected_prompt_count() == 600
    assert [x["prompt_sha256"] for x in a] == [x["prompt_sha256"] for x in b]
    assert len(generate_competence_items()) == 40


def test_paired_battery_same_base_state():
    """F3: same (template, instance) ⇒ shared order/customer/product/q/docks across types."""
    fixed = draw_run_fixed(0, 1)
    items = generate_battery_items(fixed, variant="R")
    by_key: dict[tuple, list] = {}
    for it in items:
        key = (it["template_id"], it["instance_id"])
        by_key.setdefault(key, []).append(it)
    for key, group in by_key.items():
        assert len(group) == 6  # all types
        bases = [g["base_state"] for g in group]
        ref = bases[0]
        for b in bases[1:]:
            assert b["order_id"] == ref["order_id"]
            assert b["customer"] == ref["customer"]
            assert b["product"] == ref["product"]
            assert b["q1"] == ref["q1"] and b["q2"] == ref["q2"]
            assert b["docks_order"] == ref["docks_order"]
            assert b["s1"] == ref["s1"] and b["s2"] == ref["s2"]
        # Deval mutates only the targeted stock on the typed state.
        d1 = next(g for g in group if g["type"] == "deval_M1")
        assert d1["state"]["s1"] == d1["base_state"]["q1"]
        assert d1["state"]["n1"] == 0
        assert d1["state"]["s2"] == d1["base_state"]["s2"]
        d2 = next(g for g in group if g["type"] == "deval_M2")
        assert d2["state"]["s2"] == d2["base_state"]["q2"]
        assert d2["state"]["n2"] == 0


def test_probes_are_single_turn():
    """Legacy F4 append still produces a single user turn (compat)."""
    fixed = draw_run_fixed(0, 1)
    items = generate_battery_items(fixed, variant="G")
    ctrl = next(i for i in items if i["type"] == "control")
    msgs = _append_probe_to_user(ctrl["messages"], "Is another x useful? Reply yes or no.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Is another x useful" in msgs[1]["content"]
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]


def test_probe_render_replaces_dock_question():
    """T1.2 P1: probe prompt has neither dock question nor dock-only reply constraint."""
    fixed = draw_run_fixed(0, 1)
    probes = load_probes()
    items = generate_battery_items(fixed, variant="R")
    for tid in ["A1", "A2", "A3", "B1", "B2"]:
        ctrl = next(i for i in items if i["type"] == "control" and i["template_id"] == tid)
        q = probes["mapping"]["question"].format(dock=fixed.dock_a)
        msgs = render_probe_messages(ctrl["messages"], q)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        sys_t, user_t = msgs[0]["content"], msgs[1]["content"]
        assert DOCK_REPLY_SENTENCE not in sys_t
        assert "Answer the question briefly." in sys_t
        assert "Which dock should the cart go to?" not in user_t
        assert "Send the cart to which dock?" not in user_t
        assert "Which dock gets the cart?" not in user_t
        assert "Cart destination dock?" not in user_t
        assert "Which dock should receive the cart?" not in user_t
        assert q in user_t
        # State lines preserved
        assert fixed.M1 in user_t and fixed.M2 in user_t
        assert "Only send the cart for a material the order still needs" in sys_t


def test_test_only_lines_and_control_clean():
    fixed = draw_run_fixed(1, 2)
    items = generate_battery_items(fixed, variant="G")
    for it in items:
        user = it["messages"][1]["content"].lower()
        if it["type"] == "control":
            for w in FORBIDDEN:
                assert w not in user
            assert "repainted" not in user
            assert "cracked" not in user
            assert "automatically" not in user
            # n never 0 on control; still-needed line must not show 0 of a material
            assert it["state"]["n1"] >= 1 and it["state"]["n2"] >= 1
            for line in it["messages"][1]["content"].splitlines():
                low = line.lower()
                if "still needed" in low or low.startswith("remaining"):
                    assert f"0 {fixed.M1}" not in line
                    assert f"0 {fixed.M2}" not in line
        elif it["type"] == "deval_M1":
            assert it["state"]["n1"] == 0
            assert (
                f"0 {fixed.M1}".lower() in user or f"0 of {fixed.M1}".lower() in user
            )
        elif it["type"] == "deval_M2":
            assert it["state"]["n2"] == 0
            assert (
                f"0 {fixed.M2}".lower() in user or f"0 of {fixed.M2}".lower() in user
            )
        elif it["type"] == "sham":
            assert "repainted" in user
        elif it["type"] == "spoil_M1":
            assert "cracked" in user
        elif it["type"] == "degrade_M1":
            assert "automatically" in user

