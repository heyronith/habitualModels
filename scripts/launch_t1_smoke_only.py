#!/usr/bin/env python3
"""Continue T1 §9(b) smoke training after baselines already completed."""
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
    # Conservative: assume L40S worst case for smoke if L4 OOMs.
    seconds = 30 * 40 + 30 * 45 + 30 * 35 + 3 * 200
    projected = project_cost("L40S", seconds)
    print(f"Projected worst-case ${projected:.3f}; spent ${spent_usd():.3f}")
    assert_within_budget(projected, task_only=True)

    from modal_app import app, smoke_battery_on_ckpt, smoke_sft_clone, smoke_train

    commit = git_commit()
    results = {"started_utc": utc_now_iso(), "git_commit": commit, "runs": []}
    with app.run():
        smoke_runs = []
        for mode in ["lora", "last_mlp"]:
            run_id = make_run_id(f"smoke-rl-{mode}")
            local_dir = RUNS / run_id
            local_dir.mkdir(parents=True, exist_ok=True)
            print(f"Smoke RL {mode} …", flush=True)
            r = smoke_train.remote("qwen3_1_7b", mode, 42, 0, run_id, 30)
            _write_artifacts(local_dir, r)
            slim = {k: v for k, v in r.items() if not k.startswith("artifact_")}
            write_json(local_dir / "modal_result.json", slim)
            print(
                f"  gpu={slim.get('gpu')} name={slim.get('gpu_name')} "
                f"peak_mem={slim.get('peak_gpu_mem_gb')} sec={slim.get('gpu_seconds')}",
                flush=True,
            )
            bat = smoke_battery_on_ckpt.remote("qwen3_1_7b", mode, 42, 0, run_id, 30)
            _write_artifacts(local_dir, bat)
            bat_slim = {k: v for k, v in bat.items() if not k.startswith("artifact_")}
            write_json(local_dir / "battery_summary.json", bat_slim)
            write_json(
                local_dir / "config.json",
                {
                    "kind": "smoke_rl",
                    "trainable_mode": mode,
                    "model_key": "qwen3_1_7b",
                    "git_commit": commit,
                    "git_dirty": False,
                    "gpu_type": slim.get("gpu"),
                    "gpu_name": slim.get("gpu_name"),
                    "modal": slim,
                    "ended_utc": utc_now_iso(),
                },
            )
            gpu = slim.get("gpu") or "L4"
            spend = record_spend(
                run_id=run_id,
                gpu=gpu,
                seconds=float(r["gpu_seconds"]) + float(bat["gpu_seconds"]),
                note=f"smoke rl+battery {mode}",
                run_dir=local_dir,
            )
            smoke_runs.append({"run_id": run_id, "mode": mode, "result": slim, "battery": bat_slim, "spend": spend})
            results["runs"].append(smoke_runs[-1])

        rl_id = smoke_runs[0]["run_id"]
        clone_id = make_run_id("smoke-sft-lora")
        local_dir = RUNS / clone_id
        local_dir.mkdir(parents=True, exist_ok=True)
        print("Smoke SFT clone …", flush=True)
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
                "gpu_type": c_slim.get("gpu", "L4"),
                "ended_utc": utc_now_iso(),
            },
        )
        (local_dir / "actions.jsonl").write_text("")
        (local_dir / "test_results.jsonl").write_text("")
        spend = record_spend(
            run_id=clone_id,
            gpu=c_slim.get("gpu", "L4"),
            seconds=float(c["gpu_seconds"]),
            note="smoke sft clone",
            run_dir=local_dir,
        )
        results["runs"].append({"run_id": clone_id, "kind": "sft", "result": c_slim, "spend": spend})

    results["ended_utc"] = utc_now_iso()
    results["spent_usd_total"] = spent_usd()
    write_json(ROOT / "reports" / "T1_smoke_results.json", results)
    print(json.dumps({"spent_usd": spent_usd(), "cap": budget["cap_task_usd"]}, indent=2))


if __name__ == "__main__":
    main()
