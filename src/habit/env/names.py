from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml

from habit.infra.paths import CONFIGS

_WORDLIST_CACHE: set[str] | None = None


def load_env_config() -> dict:
    return yaml.safe_load((CONFIGS / "env.yaml").read_text())


def english_wordlist() -> set[str]:
    global _WORDLIST_CACHE
    if _WORDLIST_CACHE is not None:
        return _WORDLIST_CACHE
    candidates = [
        Path("/usr/share/dict/words"),
        Path("/usr/dict/words"),
    ]
    words: set[str] = set()
    for path in candidates:
        if path.exists():
            words = {w.strip().lower() for w in path.read_text().splitlines() if w.strip()}
            break
    _WORDLIST_CACHE = words
    return words


def is_real_english(word: str) -> bool:
    return word.lower() in english_wordlist()


def filter_nonce_pool(pool: Iterable[str]) -> list[str]:
    out = []
    for w in pool:
        if is_real_english(w):
            raise ValueError(f"Material/product pool entry {w!r} is a real English word")
        out.append(w)
    return out


def validated_pools() -> tuple[list[str], list[str], list[str]]:
    cfg = load_env_config()
    materials = filter_nonce_pool(cfg["material_pool"])
    products = filter_nonce_pool(cfg["product_pool"])
    material_set = set(materials)
    for p in products:
        if p in material_set:
            raise ValueError(f"Product {p!r} overlaps material pool")
    customers = list(cfg["customer_pool"])
    if len(materials) < 12:
        raise ValueError("Need ≥12 material nonce words")
    if len(products) < 20:
        raise ValueError("Need ≥20 product nonce words")
    if len(customers) < 30:
        raise ValueError("Need ≥30 customer names")
    return materials, products, customers
