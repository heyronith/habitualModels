"""P6: reasoning-enabled Set A battery (control/deval/sham), exact post-think scoring."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from habit.env.counterbalance import RunFixed
from habit.env.render import load_templates, render_prompt
from habit.env.state import derive_state_for_type, generate_states
from habit.infra.logging import append_jsonl, write_json
from habit.model.thinking import (
    DEFAULT_REASONING_CAP,
    mentions_devalued_not_needed,
    sample_think_continuation,
    score_after_think,
    thinking_system_render_example,
)


def generate_thinking_items(
    fixed: RunFixed,
    *,
    variant: str = "G",
    n_instances: int = 10,
    types: list[str] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Paired Set A (A1–A3) items for thinking baseline."""
    types = types or ["control", "deval_M1", "deval_M2", "sham"]
    templates = load_templates()
    tids = ["A1", "A2", "A3"]
    n_base = len(tids) * n_instances
    bases = generate_states(fixed, n_base, seed, training=True)
    base_map = {}
    bi = 0
    for tid in tids:
        for inst in range(n_instances):
            base_map[(tid, inst)] = bases[bi]
            bi += 1
    items = []
    sham_flip = 0
    for ptype in types:
        for tid in tids:
            for inst in range(n_instances):
                base = base_map[(tid, inst)]
                dock_x = None
                if ptype == "sham":
                    dock_x = fixed.dock_a if (sham_flip % 2 == 0) else fixed.dock_b
                    sham_flip += 1
                typed = derive_state_for_type(base, ptype)
                rp = render_prompt(
                    base, tid, prompt_type=ptype, variant=variant, dock_x=dock_x, templates=templates
                )
                items.append(
                    {
                        "kind": "dock_thinking",
                        "type": ptype,
                        "template_id": tid,
                        "set": "A",
                        "instance_id": inst,
                        "variant": variant,
                        "base_state": base.to_dict(),
                        "state": typed.to_dict(),
                        "messages": rp.messages(),
                        "docks_order": base.docks_order,
                    }
                )
    return items


def run_thinking_battery(
    model,
    tokenizer,
    fixed: RunFixed,
    *,
    run_id: str,
    checkpoint: str,
    out_path: Path,
    device: str = "cpu",
    variant: str = "G",
    k_traces: int = 4,
    n_instances: int = 10,
    max_new_tokens: int = DEFAULT_REASONING_CAP,
    seed: int = 0,
) -> dict[str, Any]:
    items = generate_thinking_items(
        fixed, variant=variant, n_instances=n_instances, seed=seed
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    t_gen_total = 0.0
    n_traces = 0
    n_cap = 0
    reasoning_lens: list[int] = []
    deval_mention = 0
    deval_traces = 0

    for it in items:
        trace_ps_a = []
        trace_ps_b = []
        for k in range(k_traces):
            t0 = time.time()
            prompt, generated, gen_meta = sample_think_continuation(
                model,
                tokenizer,
                it["messages"],
                max_new_tokens=max_new_tokens,
                device=device,
            )
            t_gen = time.time() - t0
            t_gen_total += t_gen
            scored = score_after_think(
                model,
                tokenizer,
                prompt,
                generated,
                fixed.dock_a,
                fixed.dock_b,
                max_new_tokens=max_new_tokens,
                hit_cap=gen_meta["hit_cap"],
                device=device,
            )
            n_traces += 1
            if gen_meta["hit_cap"] or scored.get("forced_close"):
                if gen_meta["hit_cap"]:
                    n_cap += 1
            reasoning_lens.append(int(gen_meta["n_new_tokens"]))
            trace_ps_a.append(scored["p_dock_a"])
            trace_ps_b.append(scored["p_dock_b"])

            material = None
            if it["type"] == "deval_M1":
                material = fixed.M1
            elif it["type"] == "deval_M2":
                material = fixed.M2
            mentioned = False
            if material is not None:
                deval_traces += 1
                mentioned = mentions_devalued_not_needed(generated, material)
                if mentioned:
                    deval_mention += 1

            append_jsonl(
                out_path,
                {
                    "run_id": run_id,
                    "checkpoint": checkpoint,
                    "kind": "dock_thinking_trace",
                    "type": it["type"],
                    "template_id": it["template_id"],
                    "set": "A",
                    "instance_id": it["instance_id"],
                    "variant": variant,
                    "trace_k": k,
                    "p_dock_a": scored["p_dock_a"],
                    "p_dock_b": scored["p_dock_b"],
                    "label_mass": scored["label_mass"],
                    "n_new_tokens": gen_meta["n_new_tokens"],
                    "hit_cap": gen_meta["hit_cap"],
                    "forced_close": scored["forced_close"],
                    "gen_seconds": t_gen,
                    "mentions_devalued_not_needed": mentioned,
                    "docks_order": it["docks_order"],
                    "dock_a": fixed.dock_a,
                    "dock_b": fixed.dock_b,
                    "trace_preview": generated[:400],
                },
            )

        row = {
            "run_id": run_id,
            "checkpoint": checkpoint,
            "kind": "dock",  # same kind so summarize_battery / paired_ds work
            "type": it["type"],
            "template_id": it["template_id"],
            "set": "A",
            "instance_id": it["instance_id"],
            "variant": variant,
            "p_dock_a": float(np.mean(trace_ps_a)),
            "p_dock_b": float(np.mean(trace_ps_b)),
            "label_mass": float("nan"),
            "docks_order": it["docks_order"],
            "dock_a": fixed.dock_a,
            "dock_b": fixed.dock_b,
            "k_traces": k_traces,
            "n1": it["state"]["n1"],
            "n2": it["state"]["n2"],
        }
        append_jsonl(out_path, row)

    summary = {
        "n_items": len(items),
        "n_traces": n_traces,
        "mean_reasoning_tokens": float(np.mean(reasoning_lens)) if reasoning_lens else float("nan"),
        "pct_hit_cap": n_cap / max(n_traces, 1),
        "pct_deval_mention_not_needed": deval_mention / max(deval_traces, 1),
        "gen_seconds_total": t_gen_total,
        "gen_seconds_per_trace": t_gen_total / max(n_traces, 1),
        "variant": variant,
        "thinking_system": thinking_system_render_example(variant),
        "max_new_tokens": max_new_tokens,
        "k_traces": k_traces,
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    return summary
