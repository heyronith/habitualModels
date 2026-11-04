from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from habit.model.chat import render_chat, tokenize_prompt
from habit.model.tokens import encode_answer_token_ids


@torch.no_grad()
def next_token_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    return out.logits[:, -1, :]  # [B, V]


def restricted_softmax_probs(
    logits: torch.Tensor,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Softmax over a subset of vocab indices. logits: [B, V] or [V]."""
    idx = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    if logits.dim() == 1:
        sub = logits.index_select(0, idx)
        return F.softmax(sub.float(), dim=0)
    sub = logits.index_select(-1, idx)
    return F.softmax(sub.float(), dim=-1)


def full_softmax_renorm_probs(
    logits: torch.Tensor,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Renormalised full-vocabulary softmax restricted to token_ids."""
    if logits.dim() == 1:
        full = F.softmax(logits.float(), dim=0)
        sub = full[list(token_ids)]
        return sub / sub.sum()
    full = F.softmax(logits.float(), dim=-1)
    idx = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    sub = full.index_select(-1, idx)
    return sub / sub.sum(dim=-1, keepdim=True)


def label_mass(logits: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    """Sum of full-vocab probabilities of the label tokens."""
    if logits.dim() == 1:
        full = F.softmax(logits.float(), dim=0)
        return full[list(token_ids)].sum()
    full = F.softmax(logits.float(), dim=-1)
    idx = torch.tensor(list(token_ids), device=logits.device, dtype=torch.long)
    return full.index_select(-1, idx).sum(dim=-1)


def score_single_token_candidates(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    candidates: Sequence[str],
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    prompt = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    enc = tokenize_prompt(tokenizer, prompt, device=device)
    logits = next_token_logits(model, enc["input_ids"], enc.get("attention_mask"))[0]
    ids = []
    for c in candidates:
        tok_ids = encode_answer_token_ids(tokenizer, c)
        if len(tok_ids) != 1:
            raise ValueError(f"Candidate {c!r} is not a single token: {tok_ids}")
        ids.append(tok_ids[0])
    probs = restricted_softmax_probs(logits, ids)
    return {c: float(probs[i].item()) for i, c in enumerate(candidates)}


def score_multi_token_candidate_logprob(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    answer: str,
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
) -> float:
    """Sum of conditional log-probs of answer tokens given the prompt."""
    prompt = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    ans_ids = encode_answer_token_ids(tokenizer, answer)
    if not ans_ids:
        raise ValueError(f"Empty tokenization for {answer!r}")
    prompt_ids = tokenize_prompt(tokenizer, prompt, device=device)["input_ids"]
    ans = torch.tensor([ans_ids], device=device)
    full = torch.cat([prompt_ids, ans], dim=1)
    with torch.no_grad():
        out = model(input_ids=full)
        # logits at positions predicting ans tokens
        logits = out.logits[:, prompt_ids.size(1) - 1 : -1, :]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        gathered = log_probs.gather(-1, ans.unsqueeze(-1)).squeeze(-1)
        return float(gathered.sum().item())


def score_candidates(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    candidates: Sequence[str],
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    """Exact candidate scoring; multi-token answers use sum of token log-probs then softmax."""
    lengths = [len(encode_answer_token_ids(tokenizer, c)) for c in candidates]
    if all(L == 1 for L in lengths):
        return score_single_token_candidates(
            model, tokenizer, messages, candidates, enable_thinking=enable_thinking, device=device
        )
    logps = []
    for c in candidates:
        logps.append(
            score_multi_token_candidate_logprob(
                model, tokenizer, messages, c, enable_thinking=enable_thinking, device=device
            )
        )
    t = torch.tensor(logps, dtype=torch.float32)
    probs = F.softmax(t, dim=0)
    return {c: float(probs[i].item()) for i, c in enumerate(candidates)}


def dock_policy_probs(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    dock_a: int,
    dock_b: int,
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
) -> dict[str, float]:
    return dock_policy_probs_batch(
        model,
        tokenizer,
        [messages],
        dock_a,
        dock_b,
        enable_thinking=enable_thinking,
        device=device,
    )[0]


def dock_policy_probs_batch(
    model,
    tokenizer,
    messages_batch: list[list[dict[str, str]]],
    dock_a: int,
    dock_b: int,
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
    microbatch: int = 8,
) -> list[dict[str, float]]:
    """Batched exact dock probabilities (pad to max length within microbatch)."""
    id_a = encode_answer_token_ids(tokenizer, str(dock_a))[0]
    id_b = encode_answer_token_ids(tokenizer, str(dock_b))[0]
    out_rows: list[dict[str, float]] = []
    for start in range(0, len(messages_batch), microbatch):
        chunk = messages_batch[start : start + microbatch]
        prompts = [render_chat(tokenizer, m, enable_thinking=enable_thinking) for m in chunk]
        id_lists = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
        max_len = max(len(x) for x in id_lists)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(id_lists), max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(id_lists):
            input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
            attention_mask[i, : len(ids)] = 1
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            # Gather logit at last non-pad position per row
            lengths = attention_mask.sum(dim=1) - 1
            row_idx = torch.arange(logits.size(0), device=device)
            last_logits = logits[row_idx, lengths, :]
        probs = restricted_softmax_probs(last_logits, [id_a, id_b])
        masses = label_mass(last_logits, [id_a, id_b])
        for i in range(len(chunk)):
            out_rows.append(
                {
                    "p_dock_a": float(probs[i, 0].item()),
                    "p_dock_b": float(probs[i, 1].item()),
                    "label_mass": float(masses[i].item()),
                }
            )
    return out_rows


def sample_dock_action(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    dock_a: int,
    dock_b: int,
    *,
    enable_thinking: bool = False,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[int, float, float, float]:
    """Sample from 2-way softmax. Returns (label, log_pi, p_a, label_mass)."""
    prompt = render_chat(tokenizer, messages, enable_thinking=enable_thinking)
    enc = tokenize_prompt(tokenizer, prompt, device=device)
    # Need grad for training — caller controls torch.set_grad_enabled
    out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
    logits = out.logits[:, -1, :][0]
    ids = [
        encode_answer_token_ids(tokenizer, str(dock_a))[0],
        encode_answer_token_ids(tokenizer, str(dock_b))[0],
    ]
    probs = restricted_softmax_probs(logits, ids)
    mass = label_mass(logits.detach(), ids)
    # sample
    if generator is None:
        idx = int(torch.multinomial(probs.detach(), 1).item())
    else:
        idx = int(torch.multinomial(probs.detach(), 1, generator=generator).item())
    label = dock_a if idx == 0 else dock_b
    log_pi = torch.log(probs[idx] + 1e-12)
    return label, float(log_pi.detach().item()), float(probs[0].detach().item()), float(mass.item())
