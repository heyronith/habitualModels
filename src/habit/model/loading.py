from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from habit.infra.paths import CONFIGS
from habit.model.tiny import build_tiny_model


def load_models_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "models.yaml").read_text())


def resolve_model_spec(model_key: str) -> dict[str, str]:
    cfg = load_models_config()
    if model_key == "tiny":
        return {"id": "tiny", "revision": "local"}
    if model_key not in cfg["models"]:
        raise KeyError(f"Unknown model_key {model_key}")
    return dict(cfg["models"][model_key])


def load_tokenizer(model_key: str, *, cache_dir: str | Path | None = None):
    if model_key == "tiny":
        _, tok = build_tiny_model()
        return tok
    spec = resolve_model_spec(model_key)
    return AutoTokenizer.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


def load_model(
    model_key: str,
    *,
    device: str = "cpu",
    dtype: torch.dtype | None = None,
    cache_dir: str | Path | None = None,
):
    if model_key == "tiny":
        model, tokenizer = build_tiny_model(device=device)
        return model, tokenizer
    spec = resolve_model_spec(model_key)
    if dtype is None:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model = AutoModelForCausalLM.from_pretrained(
        spec["id"],
        revision=spec["revision"],
        torch_dtype=dtype,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    model.to(device)
    model.eval()
    return model, tokenizer
