"""Tiny random Qwen2-architecture model with a Qwen-style chat template for CPU tests.

Documented name: TinyQwenHabit (see configs/models.yaml).
Tokenizer: character-level with atomic multi-char specials (digits, yes/no, chat markers)
so labels 2–9 and yes/no are single tokens at the answer position.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from transformers import Qwen2Config, Qwen2ForCausalLM, PreTrainedTokenizerFast

from habit.infra.paths import CONFIGS

TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\\n' + message['content'] + '<|im_end|>\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\\n' }}"
    "{% if enable_thinking is defined and enable_thinking %}"
    "{{ '' }}"
    "{% else %}"
    "{{ '<think>\\n\\n</think>\\n\\n' }}"
    "{% endif %}"
    "{% endif %}"
)


def tiny_config_dict() -> dict:
    return yaml.safe_load((CONFIGS / "models.yaml").read_text())["tiny"]


def build_tiny_tokenizer(vocab_size: int = 1024) -> PreTrainedTokenizerFast:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Sequence, Split, WhitespaceSplit

    # Atomic tokens first (longest match via WordLevel after splitting to atoms).
    specials = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<pad>", "<unk>"]
    atoms = [str(i) for i in range(10)] + ["yes", "no"]
    # Remaining printable ASCII as single-char tokens.
    chars = [chr(i) for i in range(32, 127)]
    vocab: dict[str, int] = {}
    for tok in specials + atoms + chars:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    while len(vocab) < vocab_size:
        vocab[f"<extra{len(vocab)}>"] = len(vocab)

    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    # Split keeping specials/digits/yes/no as units; otherwise characters.
    # Practical approach: WhitespaceSplit then character fallback via our encode helper.
    tok.pre_tokenizer = WhitespaceSplit()

    wrapper = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
    )
    wrapper.chat_template = TINY_CHAT_TEMPLATE
    wrapper.add_special_tokens({})

    # Monkey-patch encode for candidate strings to guarantee atomic lookups.
    vocab_map = dict(vocab)

    def _encode_atomic(text: str, add_special_tokens: bool = False, **kwargs):
        # Prefer whole-string vocab hit (digits, yes, no).
        if text in vocab_map:
            return [vocab_map[text]]
        # Otherwise: greedy longest match over vocab keys.
        ids: list[int] = []
        i = 0
        keys = sorted(vocab_map.keys(), key=len, reverse=True)
        while i < len(text):
            matched = False
            for k in keys:
                if text.startswith(k, i):
                    ids.append(vocab_map[k])
                    i += len(k)
                    matched = True
                    break
            if not matched:
                ids.append(vocab_map["<unk>"])
                i += 1
        return ids

    # Bind methods used by our scoring utilities.
    wrapper._habit_encode_atomic = _encode_atomic  # type: ignore[attr-defined]

    original_encode = wrapper.encode

    def encode(text: str, add_special_tokens: bool = True, **kwargs):  # type: ignore[no-untyped-def]
        # For short answer candidates, use atomic encoder.
        if isinstance(text, str) and (" " not in text) and (len(text) <= 8):
            return _encode_atomic(text, add_special_tokens=add_special_tokens)
        # For longer prompts: character/atomic greedy over full string (no whitespace reliance).
        return _encode_atomic(text, add_special_tokens=False)

    wrapper.encode = encode  # type: ignore[method-assign]
    return wrapper


def build_tiny_model(seed: int = 0, device: str = "cpu"):
    cfg = tiny_config_dict()
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"],
        tie_word_embeddings=cfg["tie_word_embeddings"],
        pad_token_id=4,
    )
    model = Qwen2ForCausalLM(config)
    model.to(device)
    model.eval()
    tokenizer = build_tiny_tokenizer(cfg["vocab_size"])
    return model, tokenizer


def save_tiny_pretrained(path: Path, seed: int = 0) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    model, tokenizer = build_tiny_model(seed=seed)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "CHAT_TEMPLATE.txt").write_text(TINY_CHAT_TEMPLATE)
    return path
