#!/usr/bin/env python3
"""T1.2: P1 probe sanity → P2/P3 8B+14B baselines → P5 cost smokes → P6 thinking.

Budget: ≤ $4 total (P1–P5 ≤ $3, P6 ≤ $1). Inference except 10-update cost smokes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from habit.analysis.metrics import load_jsonl, paired_ds_rows, summarize_battery
from habit.infra.ledger import assert_within_budget, load_budget, project_cost, record_spend, spent_usd
from habit.infra.logging import git_commit, git_is_dirty, utc_now_iso, write_json
from habit.infra.paths import RUNS
from habit.infra.run_ids import make_run_id

TASK_LEDGER = ROOT / "data" / "ledger_t1_2.jsonl"
CAP_TOTAL = 4.0
CAP_P1_P5 = 3.0
CAP_P6 = 1.0


def _task_spent() -> float:
    if not TASK_LEDGER.exists():
        return 0.0
    return sum(float(json.loads(l)["cost_usd"]) for l in TASK_LEDGER.read_text().splitlines() if l.strip())


def _record_task(run_id: str, cost: float, *, phase: str) -> None:
    TASK_LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with TASK_LEDGER.open("a") as f:
        f.write(json.dumps({"run_id": run_id, "cost_usd": cost, "phase": phase}) + "\n")


def _write_artifacts(local_dir: Path, payload: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for k, v in payload.items():
        if k.startswith("artifact_") and isinstance(v, str):
            (local_dir / k[len("artifact_") :]).write_text(v, encoding="utf-8")


def _guard(projected: float, *, phase: str, p6: bool = False) -> None:
    prior = _task_spent()
    phase_cap = CAP_P6 if p6 else CAP_P1_P5
    # Soft split: P6 uses its own remaining room under $1; P1–P5 under $3; hard ≤ $4.
    phase_spent = 0.0
    if TASK_LEDGER.exists():
        for line in TASK_LEDGER.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            is_p6 = str(row.get("phase", "")).startswith("p6")
            if p6 and is_p6:
                phase_spent += float(row["cost_usd"])
            if (not p6) and (not is_p6):
                phase_spent += float(row["cost_usd"])
    print(
        f"[{phase}] projected ${projected:.3f}; phase_spent ${phase_spent:.3f}/"
        f"${phase_cap:.1f}; task ${prior:.3f}/${CAP_TOTAL:.1f}; global ${spent_usd():.3f}",
        flush=True,
    )
    if phase_spent + projected > phase_cap + 1e-9:
        raise SystemExit(f"T1.2 {phase} budget guard: {phase_spent + projected:.3f} > ${phase_cap}")
    if prior + projected > CAP_TOTAL + 1e-9:
        raise SystemExit(f"T1.2 total budget guard: {prior + projected:.3f} > ${CAP_TOTAL}")
    assert_within_budget(projected, task_only=False)


def _mean_key(cell_metrics: list[dict], k: str):
    vals = []
    for c in cell_metrics:
        m = c.get("metrics") or {}
        if k in m and m[k] == m[k]:
            vals.append(float(m[k]))
    return sum(vals) / len(vals) if vals else float("nan")


def _aggregate_cells(model_key: str, variant: str, cell_metrics: list[dict]) -> dict:
    agg = {
        "variant": variant,
        "model_key": model_key,
        "p_useful_deval_M1_setA": _mean_key(cell_metrics, "p_useful_deval_M1_setA"),
        "p_useful_deval_M2_setA": _mean_key(cell_metrics, "p_useful_deval_M2_setA"),
        "p_useful_deval_M1_setB": _mean_key(cell_metrics, "p_useful_deval_M1_setB"),
        "p_useful_deval_M2_setB": _mean_key(cell_metrics, "p_useful_deval_M2_setB"),
        "control_p_dock_a": _mean_key(cell_metrics, "control_p_dock_a"),
        "sham_shift": _mean_key(cell_metrics, "sham_shift"),
        "label_mass_control": _mean_key(cell_metrics, "label_mass_control"),
        "mapping_accuracy": _mean_key(cell_metrics, "mapping_accuracy"),
        "competence_accuracy": _mean_key(cell_metrics, "competence_accuracy"),
        "gate_pass_cells": [c.get("metrics", {}).get("gate_pass") for c in cell_metrics],
        "paired_DS_setA_cells": [c.get("metrics", {}).get("paired_DS_setA") for c in cell_metrics],
        "diagnostics": {
            "control_by_mention_order": [c.get("metrics", {}).get("control_by_mention_order") for c in cell_metrics],
            "control_by_digit": [c.get("metrics", {}).get("control_by_digit") for c in cell_metrics],
            "useful_by_mention_order_setA": [
                c.get("metrics", {}).get("useful_by_mention_order_setA") for c in cell_metrics
            ],
            "efficacy": [c.get("metrics", {}).get("efficacy") for c in cell_metrics],
            "label_mass_by_type": [c.get("metrics", {}).get("label_mass_by_type") for c in cell_metrics],
        },
        "label_draw_tries": [c.get("label_draw_tries") for c in cell_metrics],
        "prefill": cell_metrics[0].get("prefill") if cell_metrics else None,
    }
    # Mean paired DS across cells
    ds_m1 = []
    ds_m2 = []
    for p in agg["paired_DS_setA_cells"]:
        if not p:
            continue
        if p.get("DS_M1", {}).get("mean") == p.get("DS_M1", {}).get("mean"):
            ds_m1.append(p["DS_M1"]["mean"])
        if p.get("DS_M2", {}).get("mean") == p.get("DS_M2", {}).get("mean"):
            ds_m2.append(p["DS_M2"]["mean"])
    agg["paired_DS_M1_mean"] = sum(ds_m1) / len(ds_m1) if ds_m1 else float("nan")
    agg["paired_DS_M2_mean"] = sum(ds_m2) / len(ds_m2) if ds_m2 else float("nan")
    mention_vals = []
    for d in agg["diagnostics"]["useful_by_mention_order_setA"]:
        if not d:
            continue
        for side in d.values():
            mention_vals.extend(side.values())
    agg["gate_pass"] = bool(
        agg["p_useful_deval_M1_setA"] >= 0.80
        and agg["p_useful_deval_M2_setA"] >= 0.80
        and all(v >= 0.70 for v in mention_vals if v == v)
    )
    return agg


def run_p1(app, probes_only_cell, template_check_qwen) -> dict:
    """Probe fix sanity on Qwen3-4B × R × 4 cells. Stop if mapping < 0.95."""
    projected = project_cost("A10G", 4 * 90 + 60)
    _guard(projected, phase="p1")
    results = {"phase": "p1", "cells": []}
    with app.run():
        print("Template check qwen3_4b …", flush=True)
        tc = template_check_qwen.remote("qwen3_4b")
        write_json(ROOT / "reports" / "T1.2_template_check_qwen3_4b.json", tc)
        spend_tc = record_spend(
            run_id="t1.2-template-qwen3_4b",
            gpu=tc.get("gpu", "A10G"),
            seconds=float(tc.get("gpu_seconds", 0)),
            note="T1.2 template check 4B",
        )
        _record_task("t1.2-template-qwen3_4b", spend_tc["cost_usd"], phase="p1")

        run_id = make_run_id("t12-probe-qwen3_4b-R")
        local_dir = RUNS / run_id
        args = [("qwen3_4b", cell, 3000 + cell, run_id, "R") for cell in range(4)]
        print("P1 probes-only 4B R cells 0-3 …", flush=True)
        raw = list(probes_only_cell.starmap(args))
        raw.sort(key=lambda r: int(r["run_index"]))
        maps = []
        total_s = 0.0
        merged = []
        for r in raw:
            cell = int(r["run_index"])
            _write_artifacts(local_dir / f"cell{cell}", r)
            slim = {k: v for k, v in r.items() if not str(k).startswith("artifact_")}
            write_json(local_dir / f"cell{cell}_summary.json", slim)
            maps.append(float(r["metrics"]["mapping_accuracy"]))
            total_s += float(r["gpu_seconds"])
            if "artifact_probe_results.jsonl" in r:
                merged.append(r["artifact_probe_results.jsonl"])
            results["cells"].append(slim)
        (local_dir / "probe_results.jsonl").write_text("".join(merged), encoding="utf-8")
        mapping_mean = sum(maps) / len(maps)
        results["mapping_accuracy_mean"] = mapping_mean
        results["mapping_by_cell"] = maps
        results["example_probe"] = (raw[0].get("probe_summary") or {}).get("example_probe_messages")
        gpu_used = max((c.get("gpu", "A10G") for c in raw), key=lambda g: load_budget()["gpu_prices_usd_per_second"].get(g, 0))
        spend = record_spend(run_id=run_id, gpu=gpu_used, seconds=total_s, note="T1.2 P1 probes 4B-R", run_dir=local_dir)
        _record_task(run_id, spend["cost_usd"], phase="p1")
        results["spend"] = spend
        results["run_id"] = run_id
        write_json(local_dir / "p1_summary.json", results)
        write_json(ROOT / "reports" / "T1.2_p1_probe_sanity.json", results)
        print(f"P1 mapping mean={mapping_mean:.4f}", flush=True)
        if mapping_mean < 0.95:
            raise SystemExit(f"P1 STOP: mapping {mapping_mean:.4f} < 0.95")
    return results


def run_p2_p3(app, template_check_qwen, baseline_fn) -> dict:
    projected = project_cost("L40S", 2 * 90 + 16 * 240)  # 2 template + 16 cells
    _guard(projected, phase="p3")
    out = {"phase": "p3", "runs": []}
    with app.run():
        for model_key in ["qwen3_8b", "qwen3_14b"]:
            tc_path = ROOT / "reports" / f"T1.2_template_check_{model_key}.json"
            # Skip re-billing if this template check already succeeded this task.
            already = False
            if TASK_LEDGER.exists():
                for line in TASK_LEDGER.read_text().splitlines():
                    if line.strip() and json.loads(line).get("run_id") == f"t1.2-template-{model_key}":
                        already = True
                        break
            if already and tc_path.exists():
                print(f"Template/prefill check {model_key} … skipped (already logged)", flush=True)
                tc = json.loads(tc_path.read_text())
            else:
                print(f"Template/prefill check {model_key} …", flush=True)
                tc = template_check_qwen.remote(model_key)
                write_json(tc_path, tc)
                spend_tc = record_spend(
                    run_id=f"t1.2-template-{model_key}",
                    gpu=tc.get("gpu", "A10G"),
                    seconds=float(tc.get("gpu_seconds", 0)),
                    note=f"T1.2 template {model_key}",
                )
                _record_task(f"t1.2-template-{model_key}", spend_tc["cost_usd"], phase="p2")
            assert tc.get("prefill_label_ok") and tc.get("all_single"), tc

        for model_key in ["qwen3_8b", "qwen3_14b"]:
            for variant in ["G", "R"]:
                run_id = make_run_id(f"t12-base-{model_key}-{variant}")
                local_dir = RUNS / run_id
                args = [(model_key, cell, 4000 + cell, run_id, variant) for cell in range(4)]
                print(f"Baseline {model_key} {variant} …", flush=True)
                raw = list(baseline_fn.starmap(args))
                raw.sort(key=lambda r: int(r["run_index"]))
                cell_metrics = []
                total_s = 0.0
                merged = []
                for r in raw:
                    cell = int(r["run_index"])
                    _write_artifacts(local_dir / f"cell{cell}", r)
                    slim = {k: v for k, v in r.items() if not str(k).startswith("artifact_")}
                    cell_metrics.append(slim)
                    total_s += float(r["gpu_seconds"])
                    write_json(local_dir / f"cell{cell}_summary.json", slim)
                    if "artifact_test_results.jsonl" in r:
                        merged.append(r["artifact_test_results.jsonl"])
                (local_dir / "test_results.jsonl").write_text("".join(merged), encoding="utf-8")
                # Recompute paired DS on merged
                rows = load_jsonl(local_dir / "test_results.jsonl")
                agg = _aggregate_cells(model_key, variant, cell_metrics)
                agg["paired_DS_merged"] = paired_ds_rows(rows, set_name="A")
                agg["paired_DS_M1_mean"] = agg["paired_DS_merged"]["DS_M1"]["mean"]
                agg["paired_DS_M2_mean"] = agg["paired_DS_merged"]["DS_M2"]["mean"]
                write_json(local_dir / "gate_aggregate.json", agg)
                write_json(
                    local_dir / "config.json",
                    {
                        "kind": "baseline_inference_t1_2",
                        "model_key": model_key,
                        "variant": variant,
                        "git_commit": git_commit(),
                        "ended_utc": utc_now_iso(),
                    },
                )
                prices = load_budget()["gpu_prices_usd_per_second"]
                gpu_used = max((c.get("gpu", "A10G") for c in cell_metrics), key=lambda g: float(prices.get(g, 0)))
                spend = record_spend(
                    run_id=run_id, gpu=gpu_used, seconds=total_s, note=f"T1.2 baseline {model_key} {variant}", run_dir=local_dir
                )
                _record_task(run_id, spend["cost_usd"], phase="p3")
                out["runs"].append({"run_id": run_id, "aggregate": agg, "spend": spend})
    write_json(ROOT / "reports" / "T1.2_p3_baselines.json", out)
    return out


def run_p5(app, cost_smoke_rl, p3: dict) -> dict:
    """Cost smoke only if Set A paired DS_M1 ≥ 0.5."""
    eligible = []
    for run in p3.get("runs", []):
        agg = run["aggregate"]
        ds = float(agg.get("paired_DS_M1_mean", float("nan")))
        # One entry per model (prefer variant R if both qualify)
        if ds >= 0.5:
            eligible.append((agg["model_key"], ds, agg["variant"]))
    # Unique models; prefer R
    by_model: dict[str, tuple] = {}
    for mk, ds, var in eligible:
        if mk not in by_model or var == "R":
            by_model[mk] = (mk, ds, var)
    models = list(by_model.values())
    out = {"phase": "p5", "eligible": models, "smokes": []}
    if not models:
        print("P5: no model with DS_M1 ≥ 0.5; skipping cost smokes", flush=True)
        write_json(ROOT / "reports" / "T1.2_p5_cost_smokes.json", out)
        return out
    projected = project_cost("L40S", 600 * len(models))
    _guard(projected, phase="p5")
    with app.run():
        for model_key, ds, _var in models:
            run_id = make_run_id(f"t12-cost-{model_key}")
            local_dir = RUNS / run_id
            print(f"P5 cost smoke {model_key} last_mlp 10 updates (DS_M1={ds:.3f}) …", flush=True)
            r = cost_smoke_rl.remote(
                model_key, 5000, 0, run_id, 10, "last_mlp", "R"
            )
            _write_artifacts(local_dir, r)
            slim = {k: v for k, v in r.items() if not str(k).startswith("artifact_")}
            write_json(local_dir / "cost_smoke.json", slim)
            prices = load_budget()["gpu_prices_usd_per_second"]
            usd_per_s = float(prices[r["gpu"]])
            per_update_s = float(r["seconds_per_update"])
            per_update_usd = per_update_s * usd_per_s
            # Project: 300 RL updates + 1 SFT clone (~same as 300) + 5 batteries
            # Battery rough: use baseline cell cost estimate 180s on same GPU
            battery_s = 180.0
            proj_one = (300 + 300) * per_update_usd + 5 * battery_s * usd_per_s
            slim["usd_per_update"] = per_update_usd
            slim["projected_one_run_usd"] = proj_one
            slim["projected_16_runs_usd"] = 16 * proj_one
            spend = record_spend(
                run_id=run_id,
                gpu=r["gpu"],
                seconds=float(r["gpu_seconds"]),
                note=f"T1.2 P5 cost smoke {model_key}",
                run_dir=local_dir,
            )
            _record_task(run_id, spend["cost_usd"], phase="p5")
            slim["spend"] = spend
            out["smokes"].append(slim)
    write_json(ROOT / "reports" / "T1.2_p5_cost_smokes.json", out)
    return out


def run_p6(app, thinking_battery_cell) -> dict:
    # 2 models × 2 variants × 2 cells; thinking is expensive
    projected = project_cost("A10G", 8 * 600)
    _guard(min(projected, 0.95), phase="p6", p6=True)
    out = {"phase": "p6", "runs": []}
    with app.run():
        for model_key in ["qwen3_1_7b", "qwen3_4b"]:
            for variant in ["G", "R"]:
                run_id = make_run_id(f"t12-think-{model_key}-{variant}")
                local_dir = RUNS / run_id
                args = [(model_key, cell, 6000 + cell, run_id, variant, 4, 10) for cell in range(2)]
                print(f"P6 thinking {model_key} {variant} cells 0-1 …", flush=True)
                raw = list(thinking_battery_cell.starmap(args))
                raw.sort(key=lambda r: int(r["run_index"]))
                cell_metrics = []
                total_s = 0.0
                merged = []
                think_stats = []
                for r in raw:
                    cell = int(r["run_index"])
                    _write_artifacts(local_dir / f"cell{cell}", r)
                    slim = {k: v for k, v in r.items() if not str(k).startswith("artifact_")}
                    cell_metrics.append(slim)
                    total_s += float(r["gpu_seconds"])
                    think_stats.append(r.get("thinking_summary") or {})
                    write_json(local_dir / f"cell{cell}_summary.json", slim)
                    if "artifact_thinking_results.jsonl" in r:
                        merged.append(r["artifact_thinking_results.jsonl"])
                (local_dir / "thinking_results.jsonl").write_text("".join(merged), encoding="utf-8")
                rows = [r for r in load_jsonl(local_dir / "thinking_results.jsonl") if r.get("kind") == "dock"]
                agg = _aggregate_cells(model_key, variant, cell_metrics)
                agg["paired_DS_merged"] = paired_ds_rows(rows, set_name="A")
                agg["paired_DS_M1_mean"] = agg["paired_DS_merged"]["DS_M1"]["mean"]
                agg["paired_DS_M2_mean"] = agg["paired_DS_merged"]["DS_M2"]["mean"]
                agg["thinking"] = {
                    "mean_reasoning_tokens": sum(s.get("mean_reasoning_tokens", 0) for s in think_stats) / max(len(think_stats), 1),
                    "pct_hit_cap": sum(s.get("pct_hit_cap", 0) for s in think_stats) / max(len(think_stats), 1),
                    "pct_deval_mention_not_needed": sum(s.get("pct_deval_mention_not_needed", 0) for s in think_stats)
                    / max(len(think_stats), 1),
                    "gen_seconds_per_trace": sum(s.get("gen_seconds_per_trace", 0) for s in think_stats)
                    / max(len(think_stats), 1),
                    "system_render": (think_stats[0] or {}).get("thinking_system"),
                }
                write_json(local_dir / "gate_aggregate.json", agg)
                prices = load_budget()["gpu_prices_usd_per_second"]
                gpu_used = max((c.get("gpu", "A10G") for c in cell_metrics), key=lambda g: float(prices.get(g, 0)))
                spend = record_spend(
                    run_id=run_id, gpu=gpu_used, seconds=total_s, note=f"T1.2 P6 think {model_key} {variant}", run_dir=local_dir
                )
                _record_task(run_id, spend["cost_usd"], phase="p6")
                # Project RL with reasoning: 128 rollouts/update × 300 updates
                s_per_trace = agg["thinking"]["gen_seconds_per_trace"]
                usd_per_s = float(prices[gpu_used])
                # 128 rollouts × 300 updates
                rl_s = 128 * 300 * s_per_trace
                rl_usd = rl_s * usd_per_s
                agg["projected_rl_reasoning"] = {
                    "gpu": gpu_used,
                    "seconds_per_trace": s_per_trace,
                    "usd_per_run_300_updates": rl_usd,
                    "usd_16_runs": 16 * rl_usd,
                }
                out["runs"].append({"run_id": run_id, "aggregate": agg, "spend": spend})
    write_json(ROOT / "reports" / "T1.2_p6_thinking.json", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["p1", "p3", "p5", "p6", "all"],
        default="all",
        help="Which T1.2 phase to run",
    )
    args = parser.parse_args()
    if git_is_dirty():
        raise SystemExit("Live runs require a clean git tree")

    from modal_app import (
        app,
        baseline_battery_cell_large,
        cost_smoke_rl,
        probes_only_cell,
        template_check_qwen,
        thinking_battery_cell,
    )

    index = {"started_utc": utc_now_iso(), "git_commit": git_commit()}
    p1 = p3 = p5 = p6 = None
    if args.phase in ("p1", "all"):
        p1 = run_p1(app, probes_only_cell, template_check_qwen)
        index["p1"] = p1
    if args.phase in ("p3", "all"):
        p3 = run_p2_p3(app, template_check_qwen, baseline_battery_cell_large)
        index["p3"] = p3
    if args.phase in ("p5", "all"):
        if p3 is None:
            p3_path = ROOT / "reports" / "T1.2_p3_baselines.json"
            p3 = json.loads(p3_path.read_text()) if p3_path.exists() else {"runs": []}
        p5 = run_p5(app, cost_smoke_rl, p3)
        index["p5"] = p5
    if args.phase in ("p6", "all"):
        p6 = run_p6(app, thinking_battery_cell)
        index["p6"] = p6

    index["ended_utc"] = utc_now_iso()
    index["t1_2_spent_usd"] = _task_spent()
    index["global_spent_usd"] = spent_usd()
    write_json(ROOT / "reports" / "T1.2_live_results.json", index)
    print(json.dumps({"t1_2_spent_usd": index["t1_2_spent_usd"], "cap": CAP_TOTAL}, indent=2))
    if index["t1_2_spent_usd"] > CAP_TOTAL + 1e-6:
        raise SystemExit("Exceeded T1.2 $4 budget")


if __name__ == "__main__":
    main()
