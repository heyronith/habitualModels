from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(
    model,
    path: Path,
    *,
    update: int,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    weights = path / "adapter_or_weights.pt"
    # Save trainable state dict preferentially (LoRA / last MLP).
    trainable = {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}
    if not trainable:
        trainable = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"update": update, "state_dict": trainable, "meta": meta or {}}, weights)
    digest = sha256_file(weights)
    manifest = {
        "update": update,
        "path": str(weights),
        "sha256": digest,
        "meta": meta or {},
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_checkpoint(model, path: Path, *, strict: bool = False):
    weights = path / "adapter_or_weights.pt"
    blob = torch.load(weights, map_location="cpu", weights_only=False)
    missing = model.load_state_dict(blob["state_dict"], strict=strict)
    return blob, missing
