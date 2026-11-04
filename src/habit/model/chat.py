from __future__ import annotations

from typing import Any


def render_chat(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool = False,
    add_generation_prompt: bool = True,
) -> str:
    """Render chat template up to the answer position (generation prompt included)."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    # Qwen3 accepts enable_thinking; tiny template also reads it.
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        # Older/tiny tokenizers without the kwarg — inject via context if needed.
        return tokenizer.apply_chat_template(messages, **kwargs)


def tokenize_prompt(tokenizer, prompt_text: str, device: str = "cpu"):
    import torch

    # Prefer encode() so TinyQwenHabit's atomic encoder is used.
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
