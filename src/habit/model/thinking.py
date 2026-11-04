"""Thinking-mode generation with hard cap and exact post-think scoring (T1.2 P6)."""
from __future__ import annotations

import re
from typing import Any

from habit.env.render import load_variants
from habit.model.chat import default_prefill, render_chat, tokenize_prompt
from habit.model.scoring import restricted_softmax_probs
from habit.model.tokens import encode_answer_token_ids

THINK_CLOSE = "\n</think>\n\n"
THINK_OPEN = "<think>"
# Qwen model-card thinking-mode sampling (temperature 0.6, top_p 0.95).
THINKING_TEMPERATURE = 0.6
THINKING_TOP_P = 0.95
THINKING_TOP_K = 20
DEFAULT_REASONING_CAP = 256

DOCK_REPLY_SENTENCE = "Reply with the dock number only."


def system_for_thinking(system: str) -> str:
    """Drop the dock-number-only constraint; it conflicts with thinking then answering."""
    if DOCK_REPLY_SENTENCE in system:
        # Remove the sentence (and preceding space if present).
        out = system.replace(" " + DOCK_REPLY_SENTENCE, "").replace(DOCK_REPLY_SENTENCE, "")
        return out.rstrip()
    return system


def messages_for_thinking(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    out = [dict(m) for m in messages]
    assert out[0]["role"] == "system"
    out[0]["content"] = system_for_thinking(out[0]["content"])
    return out


def cap_and_close_think(
    generated_text: str,
    *,
    max_new_tokens: int = DEFAULT_REASONING_CAP,
    hit_cap: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Take generated continuation up through the think block; close if needed.

    Returns (prefix_for_scoring, meta) where prefix ends at ``</think>\\n\\n``
    (or after forced close) and is ready for ``Dock `` prefill + exact scoring.
    """
    text = generated_text
    closed = "</think>" in text
    # If an open think exists without close, or we hit the hard cap, force-close.
    force = False
    if hit_cap is True and not closed:
        force = True
    elif THINK_OPEN in text and not closed:
        # Incomplete think block even without an explicit hit_cap flag.
        force = True

    if closed:
        # Keep everything through the first </think>, then normalize trailing newlines.
        idx = text.index("</think>") + len("</think>")
        prefix = text[:idx]
        # Ensure exactly two newlines after </think> before answer prefill.
        prefix = prefix.rstrip("\n") + "\n\n"
        meta = {
            "closed_naturally": True,
            "forced_close": False,
            "reasoning_chars": len(prefix),
            "hit_cap": bool(hit_cap),
        }
        return prefix, meta

    # Truncate to the generated text and append close.
    prefix = text.rstrip("\n") + THINK_CLOSE
    meta = {
        "closed_naturally": False,
        "forced_close": True,
        "reasoning_chars": len(prefix),
        "hit_cap": True if hit_cap is None else bool(hit_cap),
    }
    return prefix, meta


def mentions_devalued_not_needed(trace: str, material: str) -> bool:
    """Exploratory regex: deval material described as not needed / already stocked."""
    m = re.escape(material)
    patterns = [
        rf"\b{m}\b.{{0,40}}(not needed|already|enough|sufficient|in (the )?storeroom|stocked|zero still)",
        rf"(not need|don't need|do not need|no longer need).{{0,40}}\b{m}\b",
        rf"(still needed).{{0,20}}0.{{0,5}}\b{m}\b",
        rf"\b{m}\b.{{0,20}}(is |are )?(full|complete|satisfied)",
    ]
    low = trace.lower()
    mat_low = material.lower()
    # Search case-insensitively on original by lowercasing both.
    for pat in patterns:
        if re.search(pat, low, flags=re.IGNORECASE) or re.search(
            pat.replace(m, re.escape(mat_low)), low, flags=re.IGNORECASE
        ):
            return True
    return False


def sample_think_continuation(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int = DEFAULT_REASONING_CAP,
    temperature: float = THINKING_TEMPERATURE,
    top_p: float = THINKING_TOP_P,
    top_k: int = THINKING_TOP_K,
    device: str = "cpu",
) -> tuple[str, str, dict[str, Any]]:
    """Generate a thinking continuation; return (prompt_text, generated_text, meta)."""
    import torch

    think_msgs = messages_for_thinking(messages)
    # enable_thinking=True: model opens a think block; no answer prefill yet.
    prompt = render_chat(
        tokenizer,
        think_msgs,
        enable_thinking=True,
        add_generation_prompt=True,
        prefill="",  # no Dock prefill during reasoning
    )
    enc = tokenize_prompt(tokenizer, prompt, device=device)
    input_len = int(enc["input_ids"].shape[1])
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0, input_len:]
    n_new = int(new_ids.shape[0])
    hit_cap = n_new >= max_new_tokens
    generated = tokenizer.decode(new_ids, skip_special_tokens=False)
    meta = {
        "n_new_tokens": n_new,
        "hit_cap": hit_cap,
        "prompt_n_tokens": input_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
    }
    return prompt, generated, meta


def score_after_think(
    model,
    tokenizer,
    prompt: str,
    generated: str,
    dock_a: int,
    dock_b: int,
    *,
    max_new_tokens: int = DEFAULT_REASONING_CAP,
    hit_cap: bool | None = None,
    device: str = "cpu",
    prefill: str | None = None,
) -> dict[str, Any]:
    """Cap/close think block, append Dock prefill, read restricted p(dock_a/b)."""
    import torch

    closed, close_meta = cap_and_close_think(
        generated, max_new_tokens=max_new_tokens, hit_cap=hit_cap
    )
    pref = default_prefill() if prefill is None else prefill
    full = prompt + closed + pref
    enc = tokenize_prompt(tokenizer, full, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask"),
        ).logits[:, -1, :][0]
    id_a = encode_answer_token_ids(tokenizer, str(dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(dock_b))[0]
    # Also get label mass over 2–9
    label_ids = [encode_answer_token_ids(tokenizer, str(i))[0] for i in range(2, 10)]
    import torch.nn.functional as F

    full_probs = F.softmax(logits.float(), dim=-1)
    label_mass = float(full_probs[label_ids].sum().item())
    probs = restricted_softmax_probs(logits, [id_a, id_b])
    return {
        "p_dock_a": float(probs[0].item()),
        "p_dock_b": float(probs[1].item()),
        "label_mass": label_mass,
        "closed_prefix": closed,
        "full_prompt_tail": full[-200:],
        **close_meta,
    }


def thinking_system_render_example(variant: str = "G") -> dict[str, str]:
    """Document exact system strings used under thinking mode."""
    variants = load_variants()
    original = variants[variant]["system"]
    return {"original": original, "thinking": system_for_thinking(original)}
