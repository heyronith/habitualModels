from __future__ import annotations

from typing import Any

import yaml

from habit.infra.paths import CONFIGS


def default_prefill() -> str:
    cfg = yaml.safe_load((CONFIGS / "env.yaml").read_text())
    return str(cfg.get("answer_prefill", "Dock "))


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
    prefill: str | None = None,
) -> str:
    """Render chat template up to the answer position, then optional assistant prefill.

    Prefill (default ``Dock ``) is appended after the empty think block so the
    next-token distribution is over dock labels.
    """
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    try:
        text = tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
    if add_generation_prompt:
        pf = default_prefill() if prefill is None else prefill
        text = text + pf
    return text


def tokenize_prompt(tokenizer, prompt_text: str, device: str = "cpu"):
    import torch

    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def choose_prefill_for_labels(tokenizer, labels: list[str] | None = None) -> str:
    """Pick ``Dock`` vs ``Dock `` so each label is a single token at the next position."""
    labels = labels or [str(i) for i in range(2, 10)]
    for candidate in ("Dock ", "Dock"):
        ok = True
        for lab in labels:
            ids = tokenizer.encode(lab, add_special_tokens=False)
            if len(ids) != 1:
                ok = False
                break
        if ok:
            # Also require that encoding the prefill+label starts with prefill tokens
            # then the single label token.
            pref_ids = tokenizer.encode(candidate, add_special_tokens=False)
            for lab in labels:
                full = tokenizer.encode(candidate + lab, add_special_tokens=False)
                lab_ids = tokenizer.encode(lab, add_special_tokens=False)
                if full != pref_ids + lab_ids:
                    ok = False
                    break
        if ok:
            return candidate
    raise RuntimeError("Could not find a Dock prefill that keeps labels 2–9 single-token")
