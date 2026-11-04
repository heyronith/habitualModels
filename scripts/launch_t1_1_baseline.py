#!/usr/bin/env python3
"""T1.1 F7: baseline gate for both models × variants G/R × 4 cells. Inference only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from habit.infra.ledger import assert_within_budget, load_budget, project_cost, record_spend, spent_usd
from habit.infra.logging import git_commit, git_is_dirty, utc_now_iso, write_json
from habit.infra.paths import RUNS
from habit.infra.run_ids import make_run_id


def _write_artifacts(local_dir: Path, payload: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for k, v in payload.items():
        if k.startswith("artifact_") and isinstance(v, str):
            (local_dir / k[len("artifact_") :]).write_text(v, encoding="utf-8")


def main() -> None:
    if git_is_dirty():
        raise SystemExit("Live runs require a clean git tree")
    budget = load_budget()
    # 2 models × 2 variants × 4 cells; include F5 bias eval overhead.
    seconds = 16 * 90  # conservative L4 seconds per cell incl. bias guard
    projected = project_cost("L4", seconds)
    # Also respect remaining room under $1 task cap for T1.1 (fresh task).
    task_spent_marker = ROOT / "data" / "ledger_t1_1.jsonl"
    prior = 0.0
    if task_spent_marker.exists():
        for line in task_spent_marker.read_text().splitlines():
            if line.strip():
                prior += float(json.loads(line)["cost_usd"])
    print(f"Projected ${projected:.3f}; T1.1 prior ${prior:.3f}; global spent ${spent_usd():.3f}")
    if prior + projected > 1.0 + 1e-9:
        raise SystemExit(f"T1.1 budget guard: {prior + projected:.3f} > $1")
    assert_within_budget(projected, task_only=False)  # still respect total cap

    from modal_app import app, baseline_battery_cell, template_check_qwen

    commit = git_commit()
    results_index: dict = {"started_utc": utc_now_iso(), "git_commit": commit, "runs": []}

    with app.run():
        # F2 live template/prefill check on both models
        for model_key in ["qwen3_1_7b", "qwen3_4b"]:
            print(f"Template/prefill check {model_key} …", flush=True)
            # Reuse template_check_qwen for 1.7B; for 4B call same with key.
            tc = template_check_qwen.remote(model_key)
            write_json(ROOT / "reports" / f"T1.1_template_check_{model_key}.json", tc)
            record_spend(
                run_id=f"t1.1-template-{model_key}",
                gpu="L4",
                seconds=float(tc.get("gpu_seconds", 0)),
                note=f"T1.1 template check {model_key}",
            )
            # Mirror into T1.1 ledger
            task_spent_marker.parent.mkdir(parents=True, exist_ok=True)
            with task_spent_marker.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "run_id": f"t1.1-template-{model_key}",
                            "cost_usd": project_cost("L4", float(tc.get("gpu_seconds", 0))),
                        }
                    )
                    + "\n"
                )

        for model_key in ["qwen3_1_7b", "qwen3_4b"]:
            for variant in ["G", "R"]:
                run_id = make_run_id(f"t11-base-{model_key}-{variant}")
                local_dir = RUNS / run_id
                local_dir.mkdir(parents=True, exist_ok=True)
                args = [(model_key, cell, 2000 + cell, run_id, variant) for cell in range(4)]
                print(f"Baseline {model_key} variant {variant} cells 0-3 …", flush=True)
                raw = list(baseline_battery_cell.starmap(args))
                raw.sort(key=lambda r: int(r["run_index"]))
                cell_metrics = []
                total_gpu_s = 0.0
                merged = []
                for r in raw:
                    cell = int(r["run_index"])
                    _write_artifacts(local_dir / f"cell{cell}", r)
                    if "artifact_test_results.jsonl" in r:
                        merged.append(r["artifact_test_results.jsonl"])
                    slim = {k: v for k, v in r.items() if not str(k).startswith("artifact_")}
                    cell_metrics.append(slim)
                    total_gpu_s += float(r["gpu_seconds"])
                    write_json(local_dir / f"cell{cell}_summary.json", slim)

                (local_dir / "test_results.jsonl").write_text("".join(merged), encoding="utf-8")
                write_json(
                    local_dir / "config.json",
                    {
                        "kind": "baseline_inference_t1_1",
                        "model_key": model_key,
                        "variant": variant,
                        "git_commit": commit,
                        "git_dirty": False,
                        "cells": cell_metrics,
                        "gpu_type": "L4",
                        "ended_utc": utc_now_iso(),
                    },
                )

                def mean_key(k: str):
                    vals = [c["metrics"][k] for c in cell_metrics if k in c["metrics"]]
                    return sum(vals) / len(vals) if vals else float("nan")

                agg = {
                    "variant": variant,
                    "model_key": model_key,
                    "p_useful_deval_M1_setA": mean_key("p_useful_deval_M1_setA"),
                    "p_useful_deval_M2_setA": mean_key("p_useful_deval_M2_setA"),
                    "p_useful_deval_M1_setB": mean_key("p_useful_deval_M1_setB"),
                    "p_useful_deval_M2_setB": mean_key("p_useful_deval_M2_setB"),
                    "control_p_dock_a": mean_key("control_p_dock_a"),
                    "sham_shift": mean_key("sham_shift"),
                    "label_mass_control": mean_key("label_mass_control"),
                    "mapping_accuracy": mean_key("mapping_accuracy"),
                    "competence_accuracy": mean_key("competence_accuracy"),
                    "gate_pass_cells": [c["metrics"].get("gate_pass") for c in cell_metrics],
                    "diagnostics": {
                        "control_by_mention_order": [
                            c["metrics"].get("control_by_mention_order") for c in cell_metrics
                        ],
                        "control_by_digit": [c["metrics"].get("control_by_digit") for c in cell_metrics],
                        "control_by_n": [c["metrics"].get("control_by_n") for c in cell_metrics],
                        "useful_by_mention_order_setA": [
                            c["metrics"].get("useful_by_mention_order_setA") for c in cell_metrics
                        ],
                        "efficacy": [c["metrics"].get("efficacy") for c in cell_metrics],
                        "label_mass_by_type": [c["metrics"].get("label_mass_by_type") for c in cell_metrics],
                    },
                    "label_draw_tries": [c.get("label_draw_tries") for c in cell_metrics],
                    "prefill": cell_metrics[0].get("prefill") if cell_metrics else None,
                }
                # Aggregate gate: mean useful ≥0.80 both devals Set A; mention-order means ≥0.70
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
                write_json(local_dir / "gate_aggregate.json", agg)
                (local_dir / "metrics.jsonl").write_text("")
                (local_dir / "actions.jsonl").write_text("")
                write_json(
                    local_dir / "checkpoint_manifest.json",
                    {"checkpoints": [], "note": "C0 base model not saved"},
                )
                spend = record_spend(
                    run_id=run_id,
                    gpu="L4",
                    seconds=total_gpu_s,
                    note=f"T1.1 baseline {model_key} {variant}",
                    run_dir=local_dir,
                )
                with task_spent_marker.open("a") as f:
                    f.write(json.dumps({"run_id": run_id, "cost_usd": spend["cost_usd"]}) + "\n")
                results_index["runs"].append({"run_id": run_id, "aggregate": agg, "spend": spend})

    results_index["ended_utc"] = utc_now_iso()
    results_index["spent_usd_total"] = spent_usd()
    t11 = sum(
        float(json.loads(l)["cost_usd"]) for l in task_spent_marker.read_text().splitlines() if l.strip()
    )
    results_index["t1_1_spent_usd"] = t11
    write_json(ROOT / "reports" / "T1.1_live_results.json", results_index)
    print(json.dumps({"t1_1_spent_usd": t11, "cap": 1.0, "global_spent": spent_usd()}, indent=2))
    if t11 > 1.0 + 1e-6:
        raise SystemExit("Exceeded T1.1 $1 budget")


if __name__ == "__main__":
    main()
