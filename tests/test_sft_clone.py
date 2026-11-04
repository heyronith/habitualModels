from __future__ import annotations

import json
from pathlib import Path

import yaml

from habit.infra.paths import CONFIGS, RUNS
from habit.infra.run_ids import make_run_id
from habit.model.loading import load_model
from habit.train.rl import train_rl
from habit.train.sft_clone import hash_actions_file, train_sft_clone


def test_sft_consumes_rl_actions_in_order(tmp_path: Path):
    cfg = yaml.safe_load((CONFIGS / "train" / "smoke.yaml").read_text())
    cfg = {
        **cfg,
        "max_updates": 3,
        "batch_states": 2,
        "group_size": 2,
        "eval_every": 100,
        "force_checkpoints": [3],
        "trainable_mode": "full_tiny",
        "seed": 0,
        "run_index": 0,
    }
    rl_dir = tmp_path / "rl"
    model, tok = load_model("tiny", device="cpu")
    train_rl(model, tok, cfg, rl_dir, device="cpu", model_key="tiny")
    actions_hash = hash_actions_file(rl_dir / "actions.jsonl")

    clone_dir = tmp_path / "clone"
    model2, tok2 = load_model("tiny", device="cpu")
    summary = train_sft_clone(model2, tok2, cfg, rl_dir, clone_dir, device="cpu", model_key="tiny")
    assert summary["actions_sha256"] == actions_hash
    # Order file exists and references source hash
    text = (clone_dir / "actions_consumed_sha256.txt").read_text()
    assert actions_hash in text
