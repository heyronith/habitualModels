from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from habit.env.counterbalance import RunFixed, cell_for_run
from habit.env.names import load_env_config, validated_pools
from habit.env.render import render_prompt
from habit.env.state import generate_states
from habit.infra.logging import append_jsonl, write_json
from habit.infra.paths import CONFIGS
from habit.model.chat import render_chat, tokenize_prompt
from habit.model.scoring import restricted_softmax_probs
from habit.model.tokens import encode_answer_token_ids


def _starting_bias_cfg() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "env.yaml").read_text())["starting_bias"]


_SET_A_TEMPLATES = ("A1", "A2", "A3")


def mean_control_p_dock_a(
    model,
    tokenizer,
    fixed: RunFixed,
    *,
    n_states: int,
    seed: int,
    variant: str = "G",
    enable_thinking: bool = False,
    device: str = "cpu",
) -> float:
    """Mean control p(dock_a) on Set A states (uniform A1–A3 mix), with Dock prefill."""
    import torch

    states = generate_states(fixed, n_states, seed, training=True)
    id_a = encode_answer_token_ids(tokenizer, str(fixed.dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(fixed.dock_b))[0]
    ps = []
    model.eval()
    with torch.no_grad():
        for i, st in enumerate(states):
            tid = _SET_A_TEMPLATES[i % len(_SET_A_TEMPLATES)]
            rp = render_prompt(st, tid, prompt_type="control", variant=variant)
            prompt = render_chat(tokenizer, rp.messages(), enable_thinking=enable_thinking)
            enc = tokenize_prompt(tokenizer, prompt, device=device)
            logits = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask")).logits[:, -1, :][0]
            probs = restricted_softmax_probs(logits, [id_a, id_b])
            ps.append(float(probs[0].item()))
    return float(np.mean(ps))


def draw_labels_for_cell(
    rng: np.random.Generator,
    cell: int,
    tokenizer=None,
) -> tuple[int, int]:
    cfg = load_env_config()
    cell_cfg = cfg["counterbalance_cells"][cell]
    labels = list(cfg["dock_label_pool"])
    for _ in range(10_000):
        a, b = (int(x) for x in rng.choice(labels, size=2, replace=False))
        if a == b:
            continue
        if tokenizer is not None:
            ok = True
            for lab in (a, b):
                ids = tokenizer.encode(str(lab), add_special_tokens=False)
                if len(ids) != 1:
                    ok = False
                    break
            if not ok:
                continue
        lower, higher = (a, b) if a < b else (b, a)
        if cell_cfg["dock_a_lower"]:
            return lower, higher
        return higher, lower
    raise RuntimeError("Could not sample single-token dock labels")


def draw_materials(
    rng: np.random.Generator,
    cell: int,
    tokenizer=None,
) -> tuple[str, str]:
    cfg = load_env_config()
    materials, _, _ = validated_pools()
    cell_cfg = cfg["counterbalance_cells"][cell]
    for _ in range(10_000):
        pair = rng.choice(materials, size=2, replace=False)
        m_first, m_second = str(pair[0]), str(pair[1])
        if tokenizer is not None:
            t1 = tokenizer.encode(m_first, add_special_tokens=False)
            t2 = tokenizer.encode(m_second, add_special_tokens=False)
            if not t1 or not t2 or t1[0] == t2[0]:
                continue
        break
    else:
        raise RuntimeError("Could not sample materials with distinct first tokens")
    if cell_cfg["m1_name_slot"] == 0:
        return m_first, m_second
    return m_second, m_first


def draw_run_fixed_with_bias_guard(
    model,
    tokenizer,
    run_index: int,
    seed: int,
    *,
    variant: str = "G",
    device: str = "cpu",
    enable_thinking: bool = False,
    log_path: Path | None = None,
) -> tuple[RunFixed, list[dict[str, Any]]]:
    """Draw materials once, then redraw dock labels until control p(dock_a) ∈ [lo, hi] (F5)."""
    cfg = load_env_config()
    bias = _starting_bias_cfg()
    cell = cell_for_run(run_index)
    rng = np.random.default_rng(seed)
    M1, M2 = draw_materials(rng, cell, tokenizer=tokenizer)

    tries: list[dict[str, Any]] = []
    fixed: RunFixed | None = None
    for attempt in range(1, int(bias["max_redraws"]) + 1):
        dock_a, dock_b = draw_labels_for_cell(rng, cell, tokenizer=tokenizer)
        candidate = RunFixed(
            run_index=run_index,
            cell=cell,
            seed=seed,
            M1=M1,
            M2=M2,
            dock_a=dock_a,
            dock_b=dock_b,
            reward_a=float(cfg["reward"]["dock_a"]),
            reward_b=float(cfg["reward"]["dock_b"]),
        )
        mean_p = mean_control_p_dock_a(
            model,
            tokenizer,
            candidate,
            n_states=int(bias["n_states"]),
            seed=int(bias["seed"]),
            variant=variant,
            enable_thinking=enable_thinking,
            device=device,
        )
        row = {
            "attempt": attempt,
            "dock_a": dock_a,
            "dock_b": dock_b,
            "mean_control_p_dock_a": mean_p,
            "accepted": bool(bias["lo"] <= mean_p <= bias["hi"]),
        }
        tries.append(row)
        if log_path is not None:
            append_jsonl(log_path, row)
        if row["accepted"]:
            fixed = candidate
            break
    if fixed is None:
        # Fall back to last try but mark failure in log.
        fixed = candidate  # type: ignore[name-defined]
        if log_path is not None:
            append_jsonl(log_path, {"attempt": "fallback", "note": "max_redraws exhausted", **tries[-1]})
    if log_path is not None:
        write_json(log_path.with_suffix(".summary.json"), {"tries": tries, "fixed": fixed.to_dict()})
    return fixed, tries
