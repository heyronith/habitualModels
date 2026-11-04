from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenCheck:
    text: str
    token_ids: list[int]
    is_single: bool


def encode_answer_token_ids(tokenizer, text: str) -> list[int]:
    """Token ids for a candidate answer string at the answer position."""
    return list(tokenizer.encode(text, add_special_tokens=False))


def check_single_token(tokenizer, texts: list[str]) -> list[TokenCheck]:
    out = []
    for t in texts:
        ids = encode_answer_token_ids(tokenizer, t)
        out.append(TokenCheck(text=t, token_ids=ids, is_single=len(ids) == 1))
    return out


def assert_single_tokens(tokenizer, texts: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for c in check_single_token(tokenizer, texts):
        if not c.is_single:
            raise AssertionError(f"{c.text!r} is not a single token: ids={c.token_ids}")
        mapping[c.text] = c.token_ids[0]
    return mapping
