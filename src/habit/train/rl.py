from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from habit.env.bias_guard import draw_run_fixed_with_bias_guard
from habit.env.counterbalance import RunFixed, draw_run_fixed
from habit.env.render import load_templates, render_prompt
from habit.env.reward import reward_for_action
from habit.env.state import generate_states
from habit.infra.logging import append_jsonl, utc_now_iso, write_json
from habit.model.chat import render_chat, tokenize_prompt
from habit.model.scoring import label_mass, restricted_softmax_probs
from habit.model.tokens import encode_answer_token_ids
from habit.train.checkpoints import save_checkpoint
from habit.train.criterion import CriterionTracker
from habit.train.params import apply_trainable_mode, count_trainable

SET_A_TEMPLATES = ("A1", "A2", "A3")


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _forward_logpi_batch(
    model,
    tokenizer,
    messages_list: list[list[dict[str, str]]],
    actions: list[int],
    dock_a: int,
    dock_b: int,
    *,
    enable_thinking: bool,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log_pi[a], p_dock_a, label_mass for each item (with grad)."""
    log_pis = []
    p_as = []
    masses = []
    id_a = encode_answer_token_ids(tokenizer, str(dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(dock_b))[0]
    for messages, action in zip(messages_list, actions):
        prompt = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
        enc = tokenize_prompt(tokenizer, prompt, device=device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        logits = out.logits[:, -1, :][0]
        probs = restricted_softmax_probs(logits, [id_a, id_b])
        mass = label_mass(logits.detach(), [id_a, id_b])
        idx = 0 if action == dock_a else 1
        log_pis.append(torch.log(probs[idx] + 1e-12))
        p_as.append(probs[0].detach())
        masses.append(mass)
    return torch.stack(log_pis), torch.stack(p_as), torch.stack(masses)


@torch.no_grad()
def evaluate_mean_p_dock_a(
    model,
    tokenizer,
    fixed: RunFixed,
    states,
    template_id: str,
    *,
    enable_thinking: bool,
    device: str,
    variant: str = "G",
) -> float:
    model.eval()
    ps = []
    id_a = encode_answer_token_ids(tokenizer, str(fixed.dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(fixed.dock_b))[0]
    for st in states:
        rp = render_prompt(st, template_id, prompt_type="control", variant=variant)
        prompt = render_chat(tokenizer, rp.messages(), enable_thinking=enable_thinking)
        enc = tokenize_prompt(tokenizer, prompt, device=device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        logits = out.logits[:, -1, :][0]
        probs = restricted_softmax_probs(logits, [id_a, id_b])
        ps.append(float(probs[0].item()))
    model.train()
    return float(np.mean(ps))


def sample_set_a_template(rng: np.random.Generator) -> str:
    """Uniform over A1–A3 (F6). Set B never used in training."""
    return str(rng.choice(SET_A_TEMPLATES))


def train_rl(
    model,
    tokenizer,
    cfg: dict[str, Any],
    run_dir: Path,
    *,
    device: str = "cpu",
    model_key: str = "tiny",
    gpu_type: str | None = None,
) -> dict[str, Any]:
    """Group REINFORCE training. No KL term and no entropy bonus."""
    _set_seeds(int(cfg["seed"]))
    run_dir.mkdir(parents=True, exist_ok=True)
    templates = load_templates()
    variant = str(cfg.get("variant", "G"))
    eval_template_id = "A1"

    # F5: starting-bias guard on dock labels (skip heavy guard on tiny CPU unless asked).
    bias_log = run_dir / "label_draw_tries.jsonl"
    if model_key == "tiny" and not cfg.get("force_bias_guard", False):
        fixed = draw_run_fixed(int(cfg["run_index"]), int(cfg["seed"]), tokenizer=tokenizer)
        write_json(run_dir / "label_draw_tries.summary.json", {"skipped": True, "reason": "tiny_cpu"})
    else:
        fixed, _tries = draw_run_fixed_with_bias_guard(
            model,
            tokenizer,
            int(cfg["run_index"]),
            int(cfg["seed"]),
            variant=variant,
            device=device,
            enable_thinking=bool(cfg.get("enable_thinking", False)),
            log_path=bias_log,
        )
    write_json(run_dir / "run_fixed.json", fixed.to_dict())

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

    B = int(cfg["batch_states"])
    G = int(cfg["group_size"])
    max_updates = int(cfg["max_updates"])
    eval_every = int(cfg["eval_every"])
    k = int(cfg["k_crit"])
    tracker = CriterionTracker(
        threshold=float(cfg["criterion_threshold"]),
        consecutive_needed=int(cfg["criterion_consecutive"]),
    )

    eval_states = generate_states(fixed, int(cfg["eval_n_states"]), int(cfg["eval_seed"]), training=True)
    enable_thinking = bool(cfg.get("enable_thinking", False))
    tmpl_rng = np.random.default_rng(int(cfg["seed"]) + 17)

    actions_path = run_dir / "actions.jsonl"
    metrics_path = run_dir / "metrics.jsonl"
    manifests: list[dict[str, Any]] = []
    force_ckpts = set(int(x) for x in cfg.get("force_checkpoints", []))
    saved_updates: set[int] = set()
    status = "running"
    t0 = time.time()
    update_times: list[float] = []

    config_resolved = {
        **cfg,
        "model_key": model_key,
        "trainable_mode_effective": mode,
        "lr": lr,
        "n_trainable": count_trainable(model),
        "fixed": fixed.to_dict(),
        "variant": variant,
        "train_templates": list(SET_A_TEMPLATES),
        "algorithm_notes": "group REINFORCE; no KL term; no entropy bonus; Set A templates A1–A3 uniform",
        "gpu_type": gpu_type,
        "started_utc": utc_now_iso(),
    }
    write_json(run_dir / "config.json", config_resolved)

    # F5: update-0 evaluation before any training step.
    mean_p0 = evaluate_mean_p_dock_a(
        model,
        tokenizer,
        fixed,
        eval_states,
        eval_template_id,
        enable_thinking=enable_thinking,
        device=device,
        variant=variant,
    )
    append_jsonl(
        metrics_path,
        {
            "update": 0,
            "mean_p_dock_a": mean_p0,
            "c_crit": None,
            "streak": 0,
            "kind": "eval",
        },
    )

    c_crit = None
    final_update = 0
    for update in range(1, max_updates + 1):
        final_update = update
        u_t0 = time.time()
        states = generate_states(fixed, B, int(cfg["seed"]) + update * 10007, training=True)
        messages_list: list[list[dict[str, str]]] = []
        actions: list[int] = []
        rewards: list[float] = []
        state_ids: list[str] = []
        prompt_shas: list[str] = []

        model.eval()
        with torch.no_grad():
            for st in states:
                tid = sample_set_a_template(tmpl_rng)
                assert tid in SET_A_TEMPLATES and tid[0] == "A"
                rp = render_prompt(
                    st, tid, prompt_type="control", variant=variant, templates=templates
                )
                prompt = render_chat(tokenizer, rp.messages(), enable_thinking=enable_thinking)
                enc = tokenize_prompt(tokenizer, prompt, device=device)
                out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
                logits = out.logits[:, -1, :][0]
                id_a = encode_answer_token_ids(tokenizer, str(fixed.dock_a))[0]
                id_b = encode_answer_token_ids(tokenizer, str(fixed.dock_b))[0]
                probs = restricted_softmax_probs(logits, [id_a, id_b])
                for _g in range(G):
                    idx = int(torch.multinomial(probs, 1).item())
                    action = fixed.dock_a if idx == 0 else fixed.dock_b
                    r = reward_for_action(fixed, action)
                    messages_list.append(rp.messages())
                    actions.append(action)
                    rewards.append(r)
                    state_ids.append(st.state_id)
                    prompt_shas.append(rp.sha256)
                    append_jsonl(
                        actions_path,
                        {
                            "update": update,
                            "state_id": st.state_id,
                            "prompt_sha256": rp.sha256,
                            "template_id": tid,
                            "variant": variant,
                            "action": action,
                            "reward": r,
                        },
                    )
        model.train()

        # Group advantages
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
        # groups are contiguous blocks of G
        adv = rewards_t.clone()
        for i in range(B):
            sl = slice(i * G, (i + 1) * G)
            adv[sl] = rewards_t[sl] - rewards_t[sl].mean()

        # Microbatch the policy-gradient forward to limit activation memory
        # (B×G can be 128); accumulate mean loss then one optimizer step.
        opt.zero_grad(set_to_none=True)
        micro = int(cfg.get("loss_microbatch", 8))
        n = len(actions)
        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        for start in range(0, n, micro):
            end = min(start + micro, n)
            log_pis, _, _ = _forward_logpi_batch(
                model,
                tokenizer,
                messages_list[start:end],
                actions[start:end],
                fixed.dock_a,
                fixed.dock_b,
                enable_thinking=enable_thinking,
                device=device,
            )
            # Weight by chunk size so overall mean matches full-batch mean.
            chunk_loss = -(adv[start:end].detach() * log_pis.float()).sum() / n
            chunk_loss.backward()
            loss_acc = loss_acc + chunk_loss.detach()
        loss = loss_acc
        opt.step()

        dt = time.time() - u_t0
        update_times.append(dt)
        append_jsonl(
            metrics_path,
            {
                "update": update,
                "loss": float(loss.item()),
                "mean_reward": float(rewards_t.mean().item()),
                "sec": dt,
                "kind": "train",
            },
        )

        if update % eval_every == 0:
            mean_p = evaluate_mean_p_dock_a(
                model,
                tokenizer,
                fixed,
                eval_states,
                eval_template_id,
                enable_thinking=enable_thinking,
                device=device,
                variant=variant,
            )
            c_crit = tracker.observe(update, mean_p)
            append_jsonl(
                metrics_path,
                {
                    "update": update,
                    "mean_p_dock_a": mean_p,
                    "c_crit": c_crit,
                    "streak": tracker.streak,
                    "kind": "eval",
                },
            )

        # Checkpoint schedule
        targets = set(force_ckpts)
        if c_crit is not None:
            targets |= set(tracker.checkpoint_updates(k))
            # Stop at k * C_crit
            if update >= k * c_crit:
                status = "completed"
                if update not in saved_updates:
                    man = save_checkpoint(
                        model,
                        run_dir / "checkpoints" / f"update_{update}",
                        update=update,
                        meta={"tag": "k_c_crit"},
                    )
                    manifests.append(man)
                    saved_updates.add(update)
                break

        for t in sorted(targets):
            if update == t and update not in saved_updates:
                man = save_checkpoint(
                    model,
                    run_dir / "checkpoints" / f"update_{update}",
                    update=update,
                    meta={"tag": "schedule"},
                )
                manifests.append(man)
                saved_updates.add(update)

    else:
        if c_crit is None:
            status = "no_criterion"
        else:
            status = "completed_max_updates"

    # Ensure final force checkpoint if needed
    if final_update in force_ckpts and final_update not in saved_updates:
        man = save_checkpoint(
            model,
            run_dir / "checkpoints" / f"update_{final_update}",
            update=final_update,
            meta={"tag": "force_final"},
        )
        manifests.append(man)

    summary = {
        "status": status,
        "c_crit": c_crit,
        "final_update": final_update,
        "n_updates": final_update,
        "mean_sec_per_update": float(np.mean(update_times)) if update_times else None,
        "elapsed_sec": time.time() - t0,
        "checkpoints": manifests,
        "ended_utc": utc_now_iso(),
    }
    write_json(run_dir / "train_summary.json", summary)
    write_json(run_dir / "checkpoint_manifest.json", {"checkpoints": manifests})
    config_resolved["ended_utc"] = summary["ended_utc"]
    config_resolved["status"] = status
    write_json(run_dir / "config.json", config_resolved)
    return summary
