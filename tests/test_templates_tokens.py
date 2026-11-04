from __future__ import annotations

import torch

from habit.env.counterbalance import draw_run_fixed
from habit.env.render import load_templates, render_prompt
from habit.env.state import generate_states
from habit.model.chat import choose_prefill_for_labels, default_prefill, render_chat, tokenize_prompt
from habit.model.loading import load_model
from habit.model.scoring import full_softmax_renorm_probs, next_token_logits, restricted_softmax_probs
from habit.model.tokens import check_single_token


def test_templates_render():
    templates = load_templates()
    assert set(templates) >= {"A1", "A2", "A3", "B1", "B2"}
    fixed = draw_run_fixed(0, 1)
    st = generate_states(fixed, 1, 0, training=True)[0]
    for tid in templates:
        rp = render_prompt(st, tid, prompt_type="control", variant="G")
        assert fixed.M1 in rp.user and fixed.M2 in rp.user
        assert "Still needed" in rp.user or "still needed" in rp.user.lower() or "Remaining" in rp.user
        assert "/20" not in rp.user
        assert rp.set in {"A", "B"}


def test_prefill_and_single_tokens_tiny():
    model, tokenizer = load_model("tiny", device="cpu")
    pref = choose_prefill_for_labels(tokenizer)
    assert pref in ("Dock ", "Dock")
    assert default_prefill() == "Dock "

    checks = check_single_token(tokenizer, [str(i) for i in range(2, 10)] + ["yes", "no"])
    assert all(c.is_single for c in checks), checks

    fixed = draw_run_fixed(0, 42, tokenizer=tokenizer)
    st = generate_states(fixed, 1, 0, training=True)[0]
    rp = render_prompt(st, "A1", variant="G")
    prompt = render_chat(tokenizer, rp.messages(), enable_thinking=False, prefill=pref)
    assert "<think>" in prompt
    assert prompt.endswith(pref) or prompt.rstrip().endswith("Dock")
    enc = tokenize_prompt(tokenizer, prompt, device="cpu")
    with torch.no_grad():
        logits = next_token_logits(model, enc["input_ids"], enc["attention_mask"])[0]
    ids = [
        tokenizer.encode(str(fixed.dock_a), add_special_tokens=False)[0],
        tokenizer.encode(str(fixed.dock_b), add_special_tokens=False)[0],
    ]
    r = restricted_softmax_probs(logits, ids)
    f = full_softmax_renorm_probs(logits, ids)
    assert torch.allclose(r, f, atol=1e-5)

    logits2 = torch.randn(50)
    assert torch.allclose(
        restricted_softmax_probs(logits2, [3, 7]),
        full_softmax_renorm_probs(logits2, [3, 7]),
        atol=1e-6,
    )


def test_variants_differ_only_in_system():
    fixed = draw_run_fixed(0, 3)
    st = generate_states(fixed, 1, 0)[0]
    g = render_prompt(st, "A1", variant="G")
    r = render_prompt(st, "A1", variant="R")
    assert g.user == r.user
    assert g.system != r.system
    assert "Only send the cart for a material the order still needs" in r.system
    assert "Only send the cart for a material the order still needs" not in g.system
