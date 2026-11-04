#!/usr/bin/env python3
"""Launch T1 §9 live Modal runs under the $5 task cap. Requires clean git tree."""
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


def require_clean_git() -> None:
    if git_is_dirty():
        raise SystemExit("Live runs require a clean git tree (git_dirty must be false).")


def _write_artifacts(local_dir: Path, payload: dict) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for k, v in payload.items():
        if k.startswith("artifact_") and isinstance(v, str):
            (local_dir / k[len("artifact_") :]).write_text(v, encoding="utf-8")


def main() -> None:
    require_clean_git()
    budget = load_budget()
    proj = budget["projections"]
    seconds = (
        proj["baseline_battery_gpu_seconds"]["qwen3_1_7b"]
        + proj["baseline_battery_gpu_seconds"]["qwen3_4b"]
        + proj["smoke_update_gpu_seconds"]["qwen3_1_7b_lora_L4"] * 30
        + proj["smoke_update_gpu_seconds"]["qwen3_1_7b_last_mlp_L4"] * 30
        + proj["smoke_sft_update_gpu_seconds"] * 30
        + 3 * proj["smoke_battery_gpu_seconds"]
        + 180  # template check
    )
    projected = project_cost("L4", seconds)
    print(f"Projected cost ${projected:.3f}; spent so far ${spent_usd():.3f}")
    assert_within_budget(projected, task_only=True)

    from modal_app import (
        app,
        baseline_battery_cell,
        smoke_battery_on_ckpt,
        smoke_sft_clone,
        smoke_train,
        template_check_qwen,
    )

    commit = git_commit()
    results_index: dict = {"started_utc": utc_now_iso(), "git_commit": commit, "runs": []}

    with app.run():
        # Template check on real Qwen3-1.7B (thinking disabled).
        print("Template check Qwen3-1.7B …")
        tc = template_check_qwen.remote("qwen3_1_7b")
        write_json(ROOT / "reports" / "T1_template_check_qwen3_1_7b.json", tc)
        md = [
            "# T1 template check (Qwen3-1.7B live)",
            "",
            f"revision: `{tc['revision']}`",
            "",
            "## Rendered string up to answer position",
            "",
            "```",
            tc["prompt"],
            "```",
            "",
            "## Token checks",
            "",
            "| text | token_ids | single |",
            "|---|---|---|",
        ]
        for c in tc["checks"]:
            md.append(f"| {c['text']} | {c['token_ids']} | {c['is_single']} |")
        md += [
            "",
            f"all_single: {tc['all_single']}",
            f"max_abs_diff (restricted vs renorm): {tc['max_abs_diff']}",
            f"equal_atol_1e-5: {tc.get('equal_atol_1e-5')}",
            "",
        ]
        (ROOT / "reports" / "T1_template_check.md").write_text("\n".join(md))
        record_spend(
            run_id="template-check-qwen3-1.7b",
            gpu="L4",
            seconds=float(tc.get("gpu_seconds", 0.0)),
            note="template check Qwen3-1.7B",
        )

        # --- §9(a) baseline gate ---
        for model_key in ["qwen3_1_7b", "qwen3_4b"]:
            run_id = make_run_id(f"baseline-{model_key}")
            local_dir = RUNS / run_id
            local_dir.mkdir(parents=True, exist_ok=True)
            cell_metrics = []
            total_gpu_s = 0.0
            merged_results: list[str] = []
            for cell in range(4):
                print(f"Baseline {model_key} cell {cell} …")
                r = baseline_battery_cell.remote(model_key, cell, 1000 + cell, run_id)
                _write_artifacts(local_dir / f"cell{cell}", r)
                if "artifact_test_results.jsonl" in r:
                    merged_results.append(r["artifact_test_results.jsonl"])
                    (local_dir / f"cell{cell}" / "test_results.jsonl").write_text(
                        r["artifact_test_results.jsonl"], encoding="utf-8"
                    )
                # strip bulky artifact from saved summary
                slim = {k: v for k, v in r.items() if not k.startswith("artifact_")}
                cell_metrics.append(slim)
                total_gpu_s += float(r["gpu_seconds"])
                write_json(local_dir / f"cell{cell}_summary.json", slim)

            (local_dir / "test_results.jsonl").write_text("".join(merged_results), encoding="utf-8")
            write_json(
                local_dir / "config.json",
                {
                    "kind": "baseline_inference",
                    "model_key": model_key,
                    "git_commit": commit,
                    "git_dirty": False,
                    "cells": cell_metrics,
                    "started_utc": results_index["started_utc"],
                    "ended_utc": utc_now_iso(),
                    "gpu_type": "L4",
                    "model_revision": cell_metrics[0].get("model_revision") if cell_metrics else None,
                },
            )

            def mean_key(k: str) -> float:
                vals = [c["metrics"][k] for c in cell_metrics if k in c["metrics"]]
                return sum(vals) / len(vals) if vals else float("nan")

            agg = {
                "p_useful_deval_M1_setA": mean_key("p_useful_deval_M1_setA"),
                "p_useful_deval_M2_setA": mean_key("p_useful_deval_M2_setA"),
                "control_p_dock_a": mean_key("control_p_dock_a"),
                "sham_shift": mean_key("sham_shift"),
                "label_mass": mean_key("label_mass"),
                "mapping_accuracy": mean_key("mapping_accuracy"),
                "competence_accuracy": mean_key("competence_accuracy"),
                "efficacy": {
                    "p_no_deval_M1": sum(c["metrics"]["efficacy"].get("p_no_deval_M1", 0) for c in cell_metrics)
                    / len(cell_metrics),
                    "p_yes_control_M1": sum(c["metrics"]["efficacy"].get("p_yes_control_M1", 0) for c in cell_metrics)
                    / len(cell_metrics),
                    "p_no_deval_M2": sum(c["metrics"]["efficacy"].get("p_no_deval_M2", 0) for c in cell_metrics)
                    / len(cell_metrics),
                    "p_yes_control_M2": sum(c["metrics"]["efficacy"].get("p_yes_control_M2", 0) for c in cell_metrics)
                    / len(cell_metrics),
                },
            }
            agg["gate_pass"] = bool(
                agg["p_useful_deval_M1_setA"] >= 0.80 and agg["p_useful_deval_M2_setA"] >= 0.80
            )
            write_json(local_dir / "gate_aggregate.json", agg)
            (local_dir / "metrics.jsonl").write_text("")
            (local_dir / "actions.jsonl").write_text("")
            write_json(local_dir / "checkpoint_manifest.json", {"checkpoints": [], "note": "C0 base model not saved"})
            spend = record_spend(
                run_id=run_id, gpu="L4", seconds=total_gpu_s, note=f"baseline {model_key}", run_dir=local_dir
            )
            results_index["runs"].append({"run_id": run_id, "kind": "baseline", "aggregate": agg, "spend": spend})

        # --- §9(b) smoke training ---
        smoke_runs = []
        for mode in ["lora", "last_mlp"]:
            run_id = make_run_id(f"smoke-rl-{mode}")
            local_dir = RUNS / run_id
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Smoke RL {mode} …")
            r = smoke_train.remote("qwen3_1_7b", mode, 42, 0, run_id, 30)
            _write_artifacts(local_dir, r)
            slim = {k: v for k, v in r.items() if not k.startswith("artifact_")}
            write_json(local_dir / "modal_result.json", slim)
            bat = smoke_battery_on_ckpt.remote("qwen3_1_7b", mode, 42, 0, run_id, 30)
            _write_artifacts(local_dir, bat)
            bat_slim = {k: v for k, v in bat.items() if not k.startswith("artifact_")}
            write_json(local_dir / "battery_summary.json", bat_slim)
            if "config.json" not in list(local_dir.glob("config.json")):
                write_json(
                    local_dir / "config.json",
                    {
                        "kind": "smoke_rl",
                        "trainable_mode": mode,
                        "model_key": "qwen3_1_7b",
                        "git_commit": commit,
                        "git_dirty": False,
                        "gpu_type": "L4",
                        "modal": slim,
                        "started_utc": results_index["started_utc"],
                        "ended_utc": utc_now_iso(),
                    },
                )
            spend = record_spend(
                run_id=run_id,
                gpu="L4",
                seconds=float(r["gpu_seconds"]) + float(bat["gpu_seconds"]),
                note=f"smoke rl+battery {mode}",
                run_dir=local_dir,
            )
            smoke_runs.append({"run_id": run_id, "mode": mode, "result": slim, "battery": bat_slim, "spend": spend})
            results_index["runs"].append(smoke_runs[-1])

        rl_id = smoke_runs[0]["run_id"]
        clone_id = make_run_id("smoke-sft-lora")
        local_dir = RUNS / clone_id
        local_dir.mkdir(parents=True, exist_ok=True)
        print("Smoke SFT clone …")
        c = smoke_sft_clone.remote("qwen3_1_7b", "lora", 42, 0, rl_id, clone_id, 30)
        _write_artifacts(local_dir, c)
        c_slim = {k: v for k, v in c.items() if not k.startswith("artifact_")}
        write_json(local_dir / "modal_result.json", c_slim)
        write_json(
            local_dir / "config.json",
            {
                "kind": "smoke_sft",
                "rl_run_id": rl_id,
                "git_commit": commit,
                "git_dirty": False,
                "gpu_type": "L4",
                "ended_utc": utc_now_iso(),
                **{k: c_slim.get(k) for k in ("summary", "gpu_seconds", "peak_gpu_mem_gb")},
            },
        )
        (local_dir / "actions.jsonl").write_text("")
        (local_dir / "test_results.jsonl").write_text("")
        spend = record_spend(
            run_id=clone_id, gpu="L4", seconds=float(c["gpu_seconds"]), note="smoke sft clone", run_dir=local_dir
        )
        results_index["runs"].append({"run_id": clone_id, "kind": "sft", "result": c_slim, "spend": spend})

    results_index["ended_utc"] = utc_now_iso()
    results_index["spent_usd_total"] = spent_usd()
    write_json(ROOT / "reports" / "T1_live_results.json", results_index)
    print(json.dumps({"spent_usd": spent_usd(), "cap": budget["cap_task_usd"]}, indent=2))
    if spent_usd() > float(budget["cap_task_usd"]) + 1e-6:
        raise SystemExit("Exceeded task budget")


if __name__ == "__main__":
    main()
