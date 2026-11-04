from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def summarize_battery(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dock = [r for r in rows if r.get("kind") == "dock"]
    by_type_set: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in dock:
        by_type_set[(r["type"], r["set"])].append(r["p_dock_a"])

    def mean_p(ptype: str, set_name: str | None = None) -> float:
        vals = []
        for (t, s), ps in by_type_set.items():
            if t != ptype:
                continue
            if set_name is not None and s != set_name:
                continue
            vals.extend(ps)
        return _mean(vals)

    control = [r for r in dock if r["type"] == "control"]
    control_a = mean_p("control")
    sham = mean_p("sham")

    ds = {}
    for ptype in ["deval_M1", "deval_M2", "sham", "spoil_M1", "degrade_M1"]:
        for set_name in ["A", "B", None]:
            key = f"DS_{ptype}" + (f"_set{set_name}" if set_name else "")
            ds[key] = mean_p("control", set_name) - mean_p(ptype, set_name)

    deval_m1 = [r for r in dock if r["type"] == "deval_M1"]
    deval_m2 = [r for r in dock if r["type"] == "deval_M2"]

    def useful_m1(rows_: list[dict]) -> float:
        return _mean([r["p_dock_b"] for r in rows_])

    def useful_m2(rows_: list[dict]) -> float:
        return _mean([r["p_dock_a"] for r in rows_])

    p_useful_m1 = useful_m1(deval_m1)
    p_useful_m2 = useful_m2(deval_m2)
    p_useful_m1_A = useful_m1([r for r in deval_m1 if r["set"] == "A"])
    p_useful_m2_A = useful_m2([r for r in deval_m2 if r["set"] == "A"])
    p_useful_m1_B = useful_m1([r for r in deval_m1 if r["set"] == "B"])
    p_useful_m2_B = useful_m2([r for r in deval_m2 if r["set"] == "B"])

    # Label mass by type
    label_mass_by_type = {
        t: _mean([r["label_mass"] for r in dock if r["type"] == t])
        for t in sorted({r["type"] for r in dock})
    }
    label_mass = _mean([r["label_mass"] for r in dock])
    label_mass_control = _mean([r["label_mass"] for r in control])

    # Diagnostics on control: mention order, digit, n1 vs n2
    def split_control(pred) -> float:
        return _mean([r["p_dock_a"] for r in control if pred(r)])

    # docks_order[0] == "a" ⇒ dock_a mentioned first
    p_a_first = split_control(lambda r: (r.get("docks_order") or ["a"])[0] == "a")
    p_a_second = split_control(lambda r: (r.get("docks_order") or ["a"])[0] == "b")
    p_lower_digit = split_control(
        lambda r: r.get("dock_a") is not None and r["dock_a"] < r["dock_b"]
    )
    p_higher_digit = split_control(
        lambda r: r.get("dock_a") is not None and r["dock_a"] > r["dock_b"]
    )
    p_n1_gt_n2 = split_control(lambda r: r.get("n1", 0) > r.get("n2", 0))
    p_n1_lt_n2 = split_control(lambda r: r.get("n1", 0) < r.get("n2", 0))
    p_n1_eq_n2 = split_control(lambda r: r.get("n1", 0) == r.get("n2", 0))

    # Useful dock by mention order of the *useful* dock under deval
    def useful_by_mention(deval_rows: list[dict], useful_is_a: bool) -> dict[str, float]:
        out = {}
        for label, first in [("useful_first", "a" if useful_is_a else "b"), ("useful_second", "b" if useful_is_a else "a")]:
            subset = [r for r in deval_rows if (r.get("docks_order") or ["a"])[0] == first]
            if useful_is_a:
                out[label] = _mean([r["p_dock_a"] for r in subset])
            else:
                out[label] = _mean([r["p_dock_b"] for r in subset])
        return out

    useful_mention_m1 = useful_by_mention([r for r in deval_m1 if r["set"] == "A"], useful_is_a=False)
    useful_mention_m2 = useful_by_mention([r for r in deval_m2 if r["set"] == "A"], useful_is_a=True)

    eff = [r for r in rows if r.get("kind") == "probe_efficacy"]
    efficacy = {}
    for mat in ["M1", "M2"]:
        deval_type = f"deval_{mat}"
        p_no = [r["p_no"] for r in eff if r.get("material") == mat and r.get("context_type") == deval_type]
        p_yes = [r["p_yes"] for r in eff if r.get("material") == mat and r.get("context_type") == "control"]
        efficacy[f"p_no_deval_{mat}"] = _mean(p_no)
        efficacy[f"p_yes_control_{mat}"] = _mean(p_yes)

    mapping = [r for r in rows if r.get("kind") == "probe_mapping"]
    mapping_acc = _mean([float(r["correct_p"] > 0.5) for r in mapping])

    competence = [r for r in rows if r.get("kind") == "probe_competence"]
    competence_acc = _mean([float(r["correct"]) for r in competence])

    sham_shift = control_a - sham

    # Gate (T1.1): ≥0.80 both devals Set A AND ≥0.70 in each dock-mention order for useful dock
    mention_ok = (
        useful_mention_m1.get("useful_first", 0) >= 0.70
        and useful_mention_m1.get("useful_second", 0) >= 0.70
        and useful_mention_m2.get("useful_first", 0) >= 0.70
        and useful_mention_m2.get("useful_second", 0) >= 0.70
    )
    gate_pass = bool(p_useful_m1_A >= 0.80 and p_useful_m2_A >= 0.80 and mention_ok)

    return {
        "control_p_dock_a": control_a,
        "sham_shift": sham_shift,
        "label_mass": label_mass,
        "label_mass_control": label_mass_control,
        "label_mass_by_type": label_mass_by_type,
        "p_useful_deval_M1": p_useful_m1,
        "p_useful_deval_M2": p_useful_m2,
        "p_useful_deval_M1_setA": p_useful_m1_A,
        "p_useful_deval_M2_setA": p_useful_m2_A,
        "p_useful_deval_M1_setB": p_useful_m1_B,
        "p_useful_deval_M2_setB": p_useful_m2_B,
        "control_by_mention_order": {"dock_a_first": p_a_first, "dock_a_second": p_a_second},
        "control_by_digit": {"dock_a_lower": p_lower_digit, "dock_a_higher": p_higher_digit},
        "control_by_n": {"n1_gt_n2": p_n1_gt_n2, "n1_lt_n2": p_n1_lt_n2, "n1_eq_n2": p_n1_eq_n2},
        "useful_by_mention_order_setA": {"deval_M1": useful_mention_m1, "deval_M2": useful_mention_m2},
        "efficacy": efficacy,
        "mapping_accuracy": mapping_acc,
        "competence_accuracy": competence_acc,
        "control_entropy": binary_entropy(control_a) if control_a == control_a else float("nan"),
        "DS": ds,
        "gate_pass": gate_pass,
    }


def ds_by_run_checkpoint_set(rows: list[dict[str, Any]], ptype: str, set_name: str) -> float:
    control = [
        r["p_dock_a"]
        for r in rows
        if r.get("kind") == "dock" and r["type"] == "control" and r["set"] == set_name
    ]
    other = [
        r["p_dock_a"]
        for r in rows
        if r.get("kind") == "dock" and r["type"] == ptype and r["set"] == set_name
    ]
    if not control or not other:
        return float("nan")
    return float(np.mean(control) - np.mean(other))


def delta_ds(ds_crit: float, ds_final: float) -> float:
    return float(ds_crit - ds_final)
