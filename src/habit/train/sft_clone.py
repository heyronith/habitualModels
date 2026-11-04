from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import load_templates, render_prompt
from habit.env.state import generate_states
from habit.infra.logging import append_jsonl, utc_now_iso, write_json
from habit.model.chat import render_chat, tokenize_prompt
from habit.model.scoring import restricted_softmax_probs
from habit.model.tokens import encode_answer_token_ids
from habit.train.checkpoints import save_checkpoint
from habit.train.params import apply_trainable_mode, count_trainable


def hash_actions_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_actions_by_update(actions_path: Path) -> dict[int, list[dict[str, Any]]]:
    by_u: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in actions_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_u[int(row["update"])].append(row)
    return dict(by_u)


def train_sft_clone(
    model,
    tokenizer,
    cfg: dict[str, Any],
    rl_run_dir: Path,
    clone_run_dir: Path,
    *,
    device: str = "cpu",
    model_key: str = "tiny",
    gpu_type: str | None = None,
    max_updates: int | None = None,
) -> dict[str, Any]:
    """No-reward SFT clone: one update per RL update on the same (state, action) pairs."""
    clone_run_dir.mkdir(parents=True, exist_ok=True)
    actions_path = rl_run_dir / "actions.jsonl"
    actions_hash = hash_actions_file(actions_path)
    by_update = load_actions_by_update(actions_path)
    updates = sorted(by_update)
    if max_updates is not None:
        updates = [u for u in updates if u <= max_updates]

    fixed = draw_run_fixed(int(cfg["run_index"]), int(cfg["seed"]), tokenizer=tokenizer)
    # Prefer RL twin's fixed env if present.
    fixed_path = rl_run_dir / "run_fixed.json"
    if fixed_path.exists():
        from habit.env.counterbalance import RunFixed

        d = json.loads(fixed_path.read_text())
        fixed = RunFixed(**d)

    mode = cfg["trainable_mode"]
    if model_key == "tiny" and mode in ("lora", "last_mlp"):
        try:
            model, lr = apply_trainable_mode(model, mode, cfg)
        except Exception:
            model, lr = apply_trainable_mode(model, "full_tiny", cfg)
            mode = "full_tiny"
    else:
        model, lr = apply_trainable_mode(model, mode, cfg)

    model.train()
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=tuple(cfg["optimizer"]["betas"]),
        eps=float(cfg["optimizer"]["eps"]),
    )
    enable_thinking = bool(cfg.get("enable_thinking", False))
    templates = load_templates()
    variant = str(cfg.get("variant", "G"))
    B = int(cfg["batch_states"])
    G = int(cfg["group_size"])

    # Rebuild states the same way as RL for each update to recover prompts.
    # Verify action log order via prompt sha when possible.
    metrics_path = clone_run_dir / "metrics.jsonl"
    manifests: list[dict[str, Any]] = []
    force_ckpts = set(int(x) for x in cfg.get("force_checkpoints", []))
    # Also mirror RL checkpoint updates from twin summary if present.
    twin_summary = rl_run_dir / "train_summary.json"
    if twin_summary.exists():
        for man in json.loads(twin_summary.read_text()).get("checkpoints", []):
            force_ckpts.add(int(man["update"]))

    t0 = time.time()
    update_times: list[float] = []
    write_json(
        clone_run_dir / "config.json",
        {
            **cfg,
            "kind": "sft_clone",
            "rl_run_dir": str(rl_run_dir),
            "actions_sha256": actions_hash,
            "model_key": model_key,
            "trainable_mode_effective": mode,
            "lr": lr,
            "n_trainable": count_trainable(model),
            "fixed": fixed.to_dict(),
            "gpu_type": gpu_type,
            "started_utc": utc_now_iso(),
        },
    )

    id_a = encode_answer_token_ids(tokenizer, str(fixed.dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(fixed.dock_b))[0]

    consumed_hash_parts: list[str] = []
    for update in updates:
        u_t0 = time.time()
        rows = by_update[update]
        states = generate_states(fixed, B, int(cfg["seed"]) + update * 10007, training=True)
        if len(rows) != B * G:
            raise RuntimeError(
                f"SFT clone action count mismatch at update {update}: {len(rows)} vs {B * G}"
            )
        # Rebuild prompts from logged template_id (F6); G rows share one state×template.
        expected = []
        for si, st in enumerate(states):
            block = rows[si * G : (si + 1) * G]
            tid = block[0].get("template_id", "A1")
            var = block[0].get("variant", variant)
            rp = render_prompt(st, tid, prompt_type="control", variant=var, templates=templates)
            for _g in range(G):
                expected.append((st, rp))

        opt.zero_grad(set_to_none=True)
        micro = int(cfg.get("loss_microbatch", 8))
        n = len(rows)
        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, n, micro):
            end = min(start + micro, n)
            log_pis = []
            for row, (st, rp) in zip(rows[start:end], expected[start:end]):
                if row["prompt_sha256"] != rp.sha256:
                    raise RuntimeError(
                        f"Prompt sha mismatch at update {update}: {row['prompt_sha256']} != {rp.sha256}"
                    )
                if row["state_id"] != st.state_id:
                    raise RuntimeError("state_id mismatch in SFT clone")
                action = int(row["action"])
                prompt = render_chat(tokenizer, rp.messages(), enable_thinking=enable_thinking)
                enc = tokenize_prompt(tokenizer, prompt, device=device)
                out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
                logits = out.logits[:, -1, :][0]
                probs = restricted_softmax_probs(logits, [id_a, id_b])
                idx = 0 if action == fixed.dock_a else 1
                log_pis.append(torch.log(probs[idx] + 1e-12))
                consumed_hash_parts.append(f"{update}:{row['state_id']}:{action}")
            chunk_loss = -torch.stack(log_pis).float().sum() / n
            chunk_loss.backward()
            loss_acc = loss_acc + chunk_loss.detach()
        loss = loss_acc
        opt.step()
        dt = time.time() - u_t0
        update_times.append(dt)
        append_jsonl(
            metrics_path,
            {"update": update, "loss": float(loss.item()), "sec": dt, "kind": "sft"},
        )
        if update in force_ckpts:
            man = save_checkpoint(
                model,
                clone_run_dir / "checkpoints" / f"update_{update}",
                update=update,
                meta={"tag": "clone"},
            )
            manifests.append(man)

    consumed_digest = hashlib.sha256("\n".join(consumed_hash_parts).encode()).hexdigest()
    (clone_run_dir / "actions_consumed_sha256.txt").write_text(
        f"source_actions_sha256={actions_hash}\nconsumed_order_sha256={consumed_digest}\n"
    )
    summary = {
        "status": "completed",
        "n_updates": len(updates),
        "actions_sha256": actions_hash,
        "consumed_order_sha256": consumed_digest,
        "mean_sec_per_update": float(np.mean(update_times)) if update_times else None,
        "elapsed_sec": time.time() - t0,
        "checkpoints": manifests,
        "ended_utc": utc_now_iso(),
    }
    write_json(clone_run_dir / "train_summary.json", summary)
    write_json(clone_run_dir / "checkpoint_manifest.json", {"checkpoints": manifests})
    return summary
