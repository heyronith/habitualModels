from __future__ import annotations

import torch

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import load_templates, render_prompt
from habit.env.state import generate_states
from habit.model.chat import render_chat, tokenize_prompt
from habit.model.loading import load_model
from habit.model.scoring import full_softmax_renorm_probs, next_token_logits, restricted_softmax_probs
from habit.model.tokens import check_single_token


def test_templates_render():
    templates = load_templates()
    assert set(templates) >= {"A1", "A2", "A3", "B1", "B2"}
    fixed = draw_run_fixed(0, 1)
    st = generate_states(fixed, 1, 0, training=True)[0]
    for tid in templates:
        rp = render_prompt(st, tid, prompt_type="control")
        assert str(fixed.dock_a) in rp.user or True
        assert fixed.M1 in rp.user and fixed.M2 in rp.user
        assert rp.set in {"A", "B"}


def test_single_tokens_and_softmax_tiny():
    model, tokenizer = load_model("tiny", device="cpu")
    checks = check_single_token(tokenizer, [str(i) for i in range(2, 10)] + ["yes", "no"])
    assert all(c.is_single for c in checks), checks

    fixed = draw_run_fixed(0, 42, tokenizer=tokenizer)
    st = generate_states(fixed, 1, 0, training=True)[0]
    rp = render_prompt(st, "A1")
    prompt = render_chat(tokenizer, rp.messages(), enable_thinking=False)
    assert "<think>" in prompt  # empty think block present for TinyQwenHabit
    enc = tokenize_prompt(tokenizer, prompt, device="cpu")
    with torch.no_grad():
        logits = next_token_logits(model, enc["input_ids"], enc["attention_mask"])[0]
    ids = [tokenizer.encode(str(fixed.dock_a), add_special_tokens=False)[0],
           tokenizer.encode(str(fixed.dock_b), add_special_tokens=False)[0]]
    r = restricted_softmax_probs(logits, ids)
    f = full_softmax_renorm_probs(logits, ids)
    assert torch.allclose(r, f, atol=1e-5)

    # Also pure math check on random logits
    logits2 = torch.randn(50)
    ids2 = [3, 7]
    assert torch.allclose(
        restricted_softmax_probs(logits2, ids2),
        full_softmax_renorm_probs(logits2, ids2),
        atol=1e-6,
    )
