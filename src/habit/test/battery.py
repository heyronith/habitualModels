from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from habit.env.counterbalance import RunFixed
from habit.env.render import load_templates, render_prompt
from habit.env.state import State, derive_state_for_type, generate_states
from habit.infra.logging import append_jsonl, write_json
from habit.infra.paths import CONFIGS, PROMPTS
from habit.model.scoring import dock_policy_probs_batch, score_candidates


def load_battery_cfg() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "test" / "battery.yaml").read_text())


def load_probes() -> dict[str, Any]:
    return yaml.safe_load((PROMPTS / "probes.yaml").read_text())


def expected_prompt_count(cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_battery_cfg()
    n_types = len(cfg["types"])
    n_templates = len(cfg["templates"]["set_a"]) + len(cfg["templates"]["set_b"])
    n_inst = int(cfg["n_instances_per_type_template"])
    return n_types * n_templates * n_inst


def generate_battery_items(
    fixed: RunFixed,
    *,
    cfg: dict[str, Any] | None = None,
    variant: str = "G",
) -> list[dict[str, Any]]:
    """Paired battery (F3): one base state per (template, instance); all types share it."""
    cfg = cfg or load_battery_cfg()
    templates = load_templates()
    seed = int(cfg["seed"])
    n_inst = int(cfg["n_instances_per_type_template"])
    all_tids = list(cfg["templates"]["set_a"]) + list(cfg["templates"]["set_b"])
    n_base = len(all_tids) * n_inst
    bases = generate_states(fixed, n_base, seed, training=True)

    items: list[dict[str, Any]] = []
    sham_flip = 0
    bi = 0
    # Index bases by (template, instance) in stable order.
    base_map: dict[tuple[str, int], State] = {}
    for tid in all_tids:
        for inst in range(n_inst):
            base_map[(tid, inst)] = bases[bi]
            bi += 1

    for ptype in cfg["types"]:
        for tid in all_tids:
            for inst in range(n_inst):
                base = base_map[(tid, inst)]
                dock_x = None
                if ptype == "sham":
                    dock_x = fixed.dock_a if (sham_flip % 2 == 0) else fixed.dock_b
                    sham_flip += 1
                typed = derive_state_for_type(base, ptype)
                rp = render_prompt(
                    base,
                    tid,
                    prompt_type=ptype,
                    variant=variant,
                    dock_x=dock_x,
                    templates=templates,
                )
                items.append(
                    {
                        "kind": "dock",
                        "type": ptype,
                        "template_id": tid,
                        "set": templates[tid]["set"],
                        "instance_id": inst,
                        "variant": variant,
                        "base_state": base.to_dict(),
                        "state": typed.to_dict(),
                        "dock_x": dock_x,
                        "prompt_sha256": rp.sha256,
                        "messages": rp.messages(),
                    }
                )
    return items


def generate_competence_items(cfg: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = cfg or load_battery_cfg()
    probes = load_probes()
    rng = np.random.default_rng(int(cfg["competence"]["seed"]))
    items = []
    n_arith = int(cfg["competence"]["n_arithmetic"])
    n_alpha = int(cfg["competence"]["n_alphabetical"])
    for i in range(n_arith):
        a = int(rng.integers(1, 10))
        b = int(rng.integers(1, 10))
        correct = a + b
        wrong = correct + (1 if rng.random() < 0.5 else -1)
        if wrong == correct:
            wrong = correct + 1
        if rng.random() < 0.5:
            c1, c2 = correct, wrong
        else:
            c1, c2 = wrong, correct
        q = f"What is {a} + {b}? Reply {c1} or {c2}."
        items.append(
            {
                "kind": "competence",
                "subtype": "arithmetic",
                "instance_id": i,
                "question": q,
                "candidates": [str(c1), str(c2)],
                "answer": str(correct),
            }
        )
    pairs = list(probes["competence"]["word_pairs"])
    for i in range(n_alpha):
        w1, w2 = pairs[i % len(pairs)]
        shown = [w1, w2] if rng.random() < 0.5 else [w2, w1]
        answer = w1 if w1.lower() < w2.lower() else w2
        q = f"Which word comes first alphabetically: {shown[0]} or {shown[1]}? Reply with the word."
        items.append(
            {
                "kind": "competence",
                "subtype": "alphabetical",
                "instance_id": i,
                "question": q,
                "candidates": shown,
                "answer": answer,
            }
        )
    return items


def _append_probe_to_user(messages: list[dict[str, str]], question: str) -> list[dict[str, str]]:
    """F4: append probe inside the same user message (no consecutive user turns)."""
    out = [dict(m) for m in messages]
    assert out[-1]["role"] == "user"
    out[-1]["content"] = out[-1]["content"].rstrip() + "\n" + question
    return out


def run_battery(
    model,
    tokenizer,
    fixed: RunFixed,
    *,
    run_id: str,
    checkpoint: str,
    out_path: Path,
    device: str = "cpu",
    enable_thinking: bool = False,
    variant: str = "G",
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or load_battery_cfg()
    probes = load_probes()
    items = generate_battery_items(fixed, cfg=cfg, variant=variant)
    if len(items) != expected_prompt_count(cfg):
        raise RuntimeError(f"Expected {expected_prompt_count(cfg)} items, got {len(items)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    controls = [it for it in items if it["type"] == "control"]
    deval_m1 = [it for it in items if it["type"] == "deval_M1"]
    deval_m2 = [it for it in items if it["type"] == "deval_M2"]

    microbatch = 16 if device != "cpu" else 4
    all_probs = dock_policy_probs_batch(
        model,
        tokenizer,
        [it["messages"] for it in items],
        fixed.dock_a,
        fixed.dock_b,
        enable_thinking=enable_thinking,
        device=device,
        microbatch=microbatch,
    )
    n_dock = 0
    for it, probs in zip(items, all_probs):
        base = it["base_state"]
        row = {
            "run_id": run_id,
            "checkpoint": checkpoint,
            "kind": "dock",
            "type": it["type"],
            "template_id": it["template_id"],
            "set": it["set"],
            "instance_id": it["instance_id"],
            "variant": variant,
            "p_dock_a": probs["p_dock_a"],
            "p_dock_b": probs["p_dock_b"],
            "label_mass": probs["label_mass"],
            "prompt_sha256": it["prompt_sha256"],
            "docks_order": base["docks_order"],
            "dock_a": fixed.dock_a,
            "dock_b": fixed.dock_b,
            "n1": it["state"]["n1"],
            "n2": it["state"]["n2"],
            "q1": base["q1"],
            "q2": base["q2"],
            "s1": it["state"]["s1"],
            "s2": it["state"]["s2"],
            "order_id": base["order_id"],
        }
        append_jsonl(out_path, row)
        n_dock += 1

    def _ask_efficacy(base_item: dict, material: str) -> dict[str, float]:
        q = probes["efficacy"]["question"].format(material=material)
        messages = _append_probe_to_user(base_item["messages"], q)
        return score_candidates(
            model,
            tokenizer,
            messages,
            probes["efficacy"]["candidates"],
            enable_thinking=enable_thinking,
            device=device,
            prefill="",  # yes/no, not dock labels
        )

    # Pair efficacy: same (template, instance) control ↔ deval
    for it in deval_m1[:20]:
        pr = _ask_efficacy(it, fixed.M1)
        append_jsonl(
            out_path,
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "kind": "probe_efficacy",
                "context_type": "deval_M1",
                "material": "M1",
                "p_yes": pr["yes"],
                "p_no": pr["no"],
                "template_id": it["template_id"],
                "instance_id": it["instance_id"],
                "variant": variant,
            },
        )
    # Paired controls for those deval instances
    ctrl_by_key = {(c["template_id"], c["instance_id"]): c for c in controls}
    for it in deval_m1[:20]:
        ctrl = ctrl_by_key[(it["template_id"], it["instance_id"])]
        pr = _ask_efficacy(ctrl, fixed.M1)
        append_jsonl(
            out_path,
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "kind": "probe_efficacy",
                "context_type": "control",
                "material": "M1",
                "p_yes": pr["yes"],
                "p_no": pr["no"],
                "template_id": it["template_id"],
                "instance_id": it["instance_id"],
                "variant": variant,
            },
        )
    for it in deval_m2[:20]:
        pr = _ask_efficacy(it, fixed.M2)
        append_jsonl(
            out_path,
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "kind": "probe_efficacy",
                "context_type": "deval_M2",
                "material": "M2",
                "p_yes": pr["yes"],
                "p_no": pr["no"],
                "template_id": it["template_id"],
                "instance_id": it["instance_id"],
                "variant": variant,
            },
        )
    for it in deval_m2[:20]:
        ctrl = ctrl_by_key[(it["template_id"], it["instance_id"])]
        pr = _ask_efficacy(ctrl, fixed.M2)
        append_jsonl(
            out_path,
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "kind": "probe_efficacy",
                "context_type": "control",
                "material": "M2",
                "p_yes": pr["yes"],
                "p_no": pr["no"],
                "template_id": it["template_id"],
                "instance_id": it["instance_id"],
                "variant": variant,
            },
        )

    for it in controls[:40]:
        for dock, material_name in [
            (fixed.dock_a, fixed.M1),
            (fixed.dock_b, fixed.M2),
        ]:
            q = probes["mapping"]["question"].format(dock=dock)
            messages = _append_probe_to_user(it["messages"], q)
            pr = score_candidates(
                model,
                tokenizer,
                messages,
                [fixed.M1, fixed.M2],
                enable_thinking=enable_thinking,
                device=device,
                prefill="",
            )
            append_jsonl(
                out_path,
                {
                    "run_id": run_id,
                    "checkpoint": checkpoint,
                    "kind": "probe_mapping",
                    "dock": dock,
                    "expected": material_name,
                    "p_M1": pr[fixed.M1],
                    "p_M2": pr[fixed.M2],
                    "correct_p": pr[material_name],
                    "template_id": it["template_id"],
                    "instance_id": it["instance_id"],
                    "variant": variant,
                },
            )

    correct = 0
    comp_items = generate_competence_items(cfg)
    for it in comp_items:
        messages = [
            {
                "role": "system",
                "content": "Answer the question. Reply with one of the options only.",
            },
            {"role": "user", "content": it["question"]},
        ]
        pr = score_candidates(
            model,
            tokenizer,
            messages,
            it["candidates"],
            enable_thinking=enable_thinking,
            device=device,
            prefill="",
        )
        pred = max(pr, key=pr.get)
        ok = pred == it["answer"]
        correct += int(ok)
        append_jsonl(
            out_path,
            {
                "run_id": run_id,
                "checkpoint": checkpoint,
                "kind": "probe_competence",
                "subtype": it["subtype"],
                "instance_id": it["instance_id"],
                "answer": it["answer"],
                "pred": pred,
                "correct": ok,
                "probs": pr,
            },
        )

    summary = {
        "n_dock_prompts": n_dock,
        "n_competence": len(comp_items),
        "competence_accuracy": correct / max(len(comp_items), 1),
        "variant": variant,
        "out_path": str(out_path),
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    return summary
