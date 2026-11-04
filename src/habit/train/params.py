from __future__ import annotations

from typing import Any

import torch.nn as nn


def apply_trainable_mode(model, mode: str, cfg: dict[str, Any]):
    """Configure LoRA or last-MLP training. Returns (model, default_lr)."""
    if mode == "lora":
        from peft import LoraConfig, get_peft_model, TaskType

        lcfg = cfg["lora"]
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lcfg["r"]),
            lora_alpha=int(lcfg["alpha"]),
            lora_dropout=float(lcfg["dropout"]),
            target_modules=list(lcfg["target_modules"]),
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        lr = float(cfg["optimizer"]["lr_lora"])
        return model, lr

    if mode == "last_mlp":
        for p in model.parameters():
            p.requires_grad = False
        n = int(cfg["last_mlp"]["n_layers"])
        # Qwen2/3: model.model.layers
        layers = model.model.layers
        for layer in layers[-n:]:
            for p in layer.mlp.parameters():
                p.requires_grad = True
        lr = float(cfg["optimizer"]["lr_last_mlp"])
        return model, lr

    if mode == "full_tiny":
        for p in model.parameters():
            p.requires_grad = True
        lr = float(cfg["optimizer"].get("lr_lora", 1e-4))
        return model, lr

    raise ValueError(f"Unknown trainable mode {mode}")


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
