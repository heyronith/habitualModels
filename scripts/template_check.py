#!/usr/bin/env python3
"""Render one prompt and verify single-token labels / restricted softmax."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import render_prompt
from habit.env.state import generate_states
from habit.infra.paths import REPORTS
from habit.model.chat import render_chat
from habit.model.loading import load_model
from habit.model.scoring import full_softmax_renorm_probs, next_token_logits, restricted_softmax_probs
from habit.model.tokens import assert_single_tokens, check_single_token
from habit.model.chat import tokenize_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", default="tiny")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model_key, device=args.device)
    fixed = draw_run_fixed(0, 42, tokenizer=tokenizer)
    st = generate_states(fixed, 1, 0, training=True)[0]
    rp = render_prompt(st, "A1", prompt_type="control")
    prompt = render_chat(tokenizer, rp.messages(), enable_thinking=False)

    labels = [str(i) for i in range(2, 10)] + ["yes", "no"]
    checks = check_single_token(tokenizer, labels)
    ids = {c.text: c.token_ids for c in checks}

    # Softmax equality on this prompt for dock labels
    enc = tokenize_prompt(tokenizer, prompt, device=args.device)
    import torch

    with torch.no_grad():
        logits = next_token_logits(model, enc["input_ids"], enc.get("attention_mask"))[0]
    dock_ids = [
        assert_single_tokens(tokenizer, [str(fixed.dock_a)])[str(fixed.dock_a)],
        assert_single_tokens(tokenizer, [str(fixed.dock_b)])[str(fixed.dock_b)],
    ]
    # For tiny model, yes/no/labels may fail single-token — report honestly.
    single_ok = all(c.is_single for c in checks)

    if single_ok:
        r = restricted_softmax_probs(logits, dock_ids)
        f = full_softmax_renorm_probs(logits, dock_ids)
        max_diff = float((r - f).abs().max().item())
    else:
        max_diff = float("nan")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "T1_template_check.md"
    lines = [
        "# T1 template check",
        "",
        f"model_key: `{args.model_key}`",
        "",
        "## Rendered string up to answer position",
        "",
        "```",
        prompt,
        "```",
        "",
        f"length_chars: {len(prompt)}",
        "",
        "## Token checks (answer-position candidates)",
        "",
        "| text | token_ids | single |",
        "|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {c.text} | {c.token_ids} | {c.is_single} |")
    lines += [
        "",
        f"all_single: {single_ok}",
        "",
        "## Restricted vs renormalised full softmax",
        "",
        f"dock_a={fixed.dock_a}, dock_b={fixed.dock_b}",
        f"max_abs_diff: {max_diff}",
        f"equal_atol_1e-5: {bool(max_diff == max_diff and max_diff < 1e-5)}",
        "",
    ]
    out.write_text("\n".join(lines))
    print(out.read_text())


if __name__ == "__main__":
    main()
