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
        return float(np.mean(vals)) if vals else float("nan")

    control_a = mean_p("control")
    sham = mean_p("sham")
    ds = {}
    for ptype in ["deval_M1", "deval_M2", "sham", "spoil_M1", "degrade_M1"]:
        for set_name in ["A", "B", None]:
            key = f"DS_{ptype}" + (f"_set{set_name}" if set_name else "")
            c = mean_p("control", set_name)
            t = mean_p(ptype, set_name)
            ds[key] = c - t

    # Useful dock under devaluation
    deval_m1 = [r for r in dock if r["type"] == "deval_M1"]
    deval_m2 = [r for r in dock if r["type"] == "deval_M2"]
    # useful = dock_b for deval_M1, dock_a for deval_M2
    p_useful_m1 = float(np.mean([r["p_dock_b"] for r in deval_m1])) if deval_m1 else float("nan")
    p_useful_m2 = float(np.mean([r["p_dock_a"] for r in deval_m2])) if deval_m2 else float("nan")
    p_useful_m1_A = float(
        np.mean([r["p_dock_b"] for r in deval_m1 if r["set"] == "A"])
    ) if deval_m1 else float("nan")
    p_useful_m2_A = float(
        np.mean([r["p_dock_a"] for r in deval_m2 if r["set"] == "A"])
    ) if deval_m2 else float("nan")

    label_mass = float(np.mean([r["label_mass"] for r in dock])) if dock else float("nan")

    # Efficacy: p(no) on deval, p(yes) on control
    eff = [r for r in rows if r.get("kind") == "probe_efficacy"]
    efficacy = {}
    for mat in ["M1", "M2"]:
        deval_type = f"deval_{mat}"
        p_no = [r["p_no"] for r in eff if r.get("material") == mat and r.get("context_type") == deval_type]
        p_yes = [r["p_yes"] for r in eff if r.get("material") == mat and r.get("context_type") == "control"]
        efficacy[f"p_no_deval_{mat}"] = float(np.mean(p_no)) if p_no else float("nan")
        efficacy[f"p_yes_control_{mat}"] = float(np.mean(p_yes)) if p_yes else float("nan")

    mapping = [r for r in rows if r.get("kind") == "probe_mapping"]
    mapping_acc = float(np.mean([r["correct_p"] > 0.5 for r in mapping])) if mapping else float("nan")

    competence = [r for r in rows if r.get("kind") == "probe_competence"]
    competence_acc = float(np.mean([r["correct"] for r in competence])) if competence else float("nan")

    sham_shift = control_a - sham

    return {
        "control_p_dock_a": control_a,
        "sham_shift": sham_shift,
        "label_mass": label_mass,
        "p_useful_deval_M1": p_useful_m1,
        "p_useful_deval_M2": p_useful_m2,
        "p_useful_deval_M1_setA": p_useful_m1_A,
        "p_useful_deval_M2_setA": p_useful_m2_A,
        "efficacy": efficacy,
        "mapping_accuracy": mapping_acc,
        "competence_accuracy": competence_acc,
        "control_entropy": binary_entropy(control_a) if control_a == control_a else float("nan"),
        "DS": ds,
        "gate_pass": bool(
            p_useful_m1_A >= 0.80 and p_useful_m2_A >= 0.80
        ),
    }


def ds_by_run_checkpoint_set(rows: list[dict[str, Any]], ptype: str, set_name: str) -> float:
    control = [r["p_dock_a"] for r in rows if r.get("kind") == "dock" and r["type"] == "control" and r["set"] == set_name]
    other = [r["p_dock_a"] for r in rows if r.get("kind") == "dock" and r["type"] == ptype and r["set"] == set_name]
    if not control or not other:
        return float("nan")
    return float(np.mean(control) - np.mean(other))


def delta_ds(ds_crit: float, ds_final: float) -> float:
    return float(ds_crit - ds_final)
