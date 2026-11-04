#!/usr/bin/env python3
"""CPU integration: RL 20 updates → checkpoint → clone → battery → analysis."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from habit.analysis.metrics import load_jsonl, summarize_battery
from habit.env.counterbalance import draw_run_fixed
from habit.infra.logging import write_json
from habit.infra.paths import CONFIGS, RUNS
from habit.infra.run_ids import make_run_id
from habit.model.loading import load_model
from habit.test.battery import run_battery
from habit.train.rl import train_rl
from habit.train.sft_clone import train_sft_clone


def main() -> None:
    t0 = time.time()
    cfg = yaml.safe_load((CONFIGS / "train" / "smoke.yaml").read_text())
    cfg = {
        **cfg,
        "name": "cpu-integration",
        "model_key": "tiny",
        "trainable_mode": "full_tiny",
        "max_updates": 20,
        "batch_states": 4,
        "group_size": 2,
        "eval_every": 5,
        "eval_n_states": 8,
        "force_checkpoints": [20],
        "optimizer": {**cfg["optimizer"], "lr_lora": 1e-3},
    }
    run_id = make_run_id("cpu-rl")
    run_dir = RUNS / run_id
    model, tokenizer = load_model("tiny", device="cpu")
    summary = train_rl(model, tokenizer, cfg, run_dir, device="cpu", model_key="tiny")
    print("RL summary", summary)

    # Clone
    clone_id = make_run_id("cpu-sft")
    clone_dir = RUNS / clone_id
    model2, tokenizer2 = load_model("tiny", device="cpu")
    clone_summary = train_sft_clone(
        model2, tokenizer2, cfg, run_dir, clone_dir, device="cpu", model_key="tiny", max_updates=20
    )
    print("SFT summary", clone_summary)

    # Battery on final RL checkpoint
    fixed = draw_run_fixed(int(cfg["run_index"]), int(cfg["seed"]), tokenizer=tokenizer)
    # Reload trained weights
    from habit.train.checkpoints import load_checkpoint

    ckpt = run_dir / "checkpoints" / "update_20"
    model3, tokenizer3 = load_model("tiny", device="cpu")
    # Match trainable mode used in training
    from habit.train.params import apply_trainable_mode

    model3, _ = apply_trainable_mode(model3, "full_tiny", cfg)
    load_checkpoint(model3, ckpt, strict=False)
    model3.eval()
    bat = run_battery(
        model3,
        tokenizer3,
        fixed,
        run_id=run_id,
        checkpoint="update_20",
        out_path=run_dir / "test_results.jsonl",
        device="cpu",
    )
    rows = load_jsonl(run_dir / "test_results.jsonl")
    metrics = summarize_battery(rows)
    write_json(run_dir / "analysis_summary.json", metrics)
    elapsed = time.time() - t0
    print("battery", bat)
    print("metrics", metrics)
    print(f"elapsed_sec={elapsed:.1f}")
    if elapsed > 600:
        raise SystemExit(f"CPU integration exceeded 10 min: {elapsed:.1f}s")
    print("OK")


if __name__ == "__main__":
    main()
