"""Modal app for baseline-inference gate and smoke training (T1 §9)."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get("HABIT_MODAL_APP", "habitual-models")
VOLUME_NAME = os.environ.get("HABIT_VOLUME", "habitual-checkpoints")
_REPO = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers==4.51.3",
        "peft==0.15.2",
        "numpy==2.2.4",
        "scipy==1.15.2",
        "pandas==2.2.3",
        "pyyaml==6.0.2",
        "pydantic==2.11.3",
        "accelerate==1.6.0",
        "safetensors==0.5.3",
        "huggingface-hub==0.30.2",
        "tqdm==4.67.1",
    )
    .env({"HF_HOME": "/root/.cache/huggingface", "PYTHONPATH": "/root/project/src"})
    .add_local_dir(str(_REPO / "src"), remote_path="/root/project/src")
    .add_local_dir(str(_REPO / "configs"), remote_path="/root/project/configs")
    .add_local_dir(str(_REPO / "prompts"), remote_path="/root/project/prompts")
)

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
# Workspace secret stores HF_TOKEN (created as `hf-token` in this account).
secrets = [modal.Secret.from_name("hf-token")]
_FUNCTION_KW = dict(
    image=image,
    volumes={"/vol": vol},
    secrets=secrets,
    memory=32768,
)


def _gpu_mem_peak_gb() -> float | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.max_memory_allocated() / (1024**3))
    except Exception:
        return None


@app.function(gpu=["L4", "A10G", "T4"], timeout=60 * 30, **_FUNCTION_KW)
def template_check_qwen(model_key: str = "qwen3_1_7b") -> dict[str, Any]:
    """Render prompt + verify single-token labels on a pinned Qwen3 revision."""
    import sys

    sys.path.insert(0, "/root/project/src")
    import torch

    from habit.env.counterbalance import draw_run_fixed
    from habit.env.render import render_prompt
    from habit.env.state import generate_states
    from habit.model.chat import render_chat, tokenize_prompt
    from habit.model.loading import load_model, resolve_model_spec
    from habit.model.scoring import full_softmax_renorm_probs, next_token_logits, restricted_softmax_probs
    from habit.model.tokens import check_single_token

    from habit.model.chat import choose_prefill_for_labels, default_prefill

    t0 = time.time()
    model, tokenizer = load_model(model_key, device="cuda", dtype=torch.bfloat16)
    pref = choose_prefill_for_labels(tokenizer)
    fixed = draw_run_fixed(0, 42, tokenizer=tokenizer)
    st = generate_states(fixed, 1, 0, training=True)[0]
    prompts = {}
    for variant in ("G", "R"):
        rp = render_prompt(st, "A1", prompt_type="control", variant=variant)
        prompts[variant] = {
            "system": rp.system,
            "user": rp.user,
            "rendered": render_chat(
                tokenizer, rp.messages(), enable_thinking=False, prefill=pref
            ),
        }
    labels = [str(i) for i in range(2, 10)] + ["yes", "no"]
    checks = [
        {"text": c.text, "token_ids": c.token_ids, "is_single": c.is_single}
        for c in check_single_token(tokenizer, labels)
    ]
    # Prefill+label continuity
    pref_ok = True
    pref_ids = tokenizer.encode(pref, add_special_tokens=False)
    for lab in [str(i) for i in range(2, 10)]:
        full = tokenizer.encode(pref + lab, add_special_tokens=False)
        lab_ids = tokenizer.encode(lab, add_special_tokens=False)
        if full != pref_ids + lab_ids or len(lab_ids) != 1:
            pref_ok = False
            break
    enc = tokenize_prompt(tokenizer, prompts["G"]["rendered"], device="cuda")
    with torch.no_grad():
        logits = next_token_logits(model, enc["input_ids"], enc.get("attention_mask"))[0]
    dock_ids = [
        tokenizer.encode(str(fixed.dock_a), add_special_tokens=False)[0],
        tokenizer.encode(str(fixed.dock_b), add_special_tokens=False)[0],
    ]
    r = restricted_softmax_probs(logits, dock_ids)
    f = full_softmax_renorm_probs(logits, dock_ids)
    max_diff = float((r - f).abs().max().item())
    result = {
        "model_key": model_key,
        "revision": resolve_model_spec(model_key)["revision"],
        "prefill": pref,
        "default_prefill": default_prefill(),
        "prefill_label_ok": pref_ok,
        "prompts": prompts,
        "prompt": prompts["G"]["rendered"],
        "checks": checks,
        "all_single": all(c["is_single"] for c in checks),
        "dock_a": fixed.dock_a,
        "dock_b": fixed.dock_b,
        "max_abs_diff": max_diff,
        "equal_atol_1e-5": max_diff < 1e-5,
        "fixed": fixed.to_dict(),
        "gpu": (
            "L40S"
            if "L40" in (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
            else "A10G"
            if "A10" in (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
            else "T4"
            if "T4" in (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
            else "L4"
        ),
        "gpu_seconds": time.time() - t0,
    }
    out = Path(f"/vol/reports/T1.1_template_check_{model_key}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    vol.commit()
    return result


@app.function(gpu=["L4", "A10G", "T4"], timeout=60 * 60 * 3, **_FUNCTION_KW)
def baseline_battery_cell(
    model_key: str,
    run_index: int,
    seed: int,
    run_id: str,
    variant: str = "G",
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/project/src")
    import torch

    from habit.analysis.metrics import load_jsonl, summarize_battery
    from habit.env.bias_guard import draw_run_fixed_with_bias_guard
    from habit.infra.logging import write_json
    from habit.model.chat import choose_prefill_for_labels, default_prefill
    from habit.model.loading import load_model, resolve_model_spec
    from habit.model.tokens import check_single_token
    from habit.test.battery import run_battery

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = load_model(model_key, device="cuda", dtype=torch.bfloat16)
    # Verify prefill keeps labels single-token (F2).
    pref = choose_prefill_for_labels(tokenizer)
    assert pref == default_prefill(), f"prefill mismatch: {pref!r} vs {default_prefill()!r}"
    label_checks = [
        {"text": c.text, "token_ids": c.token_ids, "is_single": c.is_single}
        for c in check_single_token(tokenizer, [str(i) for i in range(2, 10)])
    ]
    assert all(c["is_single"] for c in label_checks)

    out_dir = Path(f"/vol/runs/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    bias_log = out_dir / f"label_draw_tries_cell{run_index}.jsonl"
    fixed, tries = draw_run_fixed_with_bias_guard(
        model,
        tokenizer,
        run_index,
        seed,
        variant=variant,
        device="cuda",
        enable_thinking=False,
        log_path=bias_log,
    )
    local_out = out_dir / f"test_results_cell{run_index}.jsonl"
    summary = run_battery(
        model,
        tokenizer,
        fixed,
        run_id=run_id,
        checkpoint="C0",
        out_path=local_out,
        device="cuda",
        enable_thinking=False,
        variant=variant,
    )
    rows = load_jsonl(local_out)
    metrics = summarize_battery(rows)
    spec = resolve_model_spec(model_key)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if "L40" in gpu_name:
        gpu_key = "L40S"
    elif "A10" in gpu_name:
        gpu_key = "A10G"
    elif "T4" in gpu_name:
        gpu_key = "T4"
    else:
        gpu_key = "L4"
    result = {
        "run_id": run_id,
        "model_key": model_key,
        "model_revision": spec["revision"],
        "run_index": run_index,
        "variant": variant,
        "cell": fixed.cell,
        "fixed": fixed.to_dict(),
        "prefill": pref,
        "label_token_checks": label_checks,
        "label_draw_tries": tries,
        "battery_summary": summary,
        "metrics": metrics,
        "gpu": gpu_key,
        "gpu_name": gpu_name,
        "gpu_seconds": time.time() - t0,
        "peak_gpu_mem_gb": _gpu_mem_peak_gb(),
        "out_path": str(local_out),
    }
    if local_out.exists():
        result["artifact_test_results.jsonl"] = local_out.read_text(encoding="utf-8")
    if bias_log.exists():
        result["artifact_label_draw_tries.jsonl"] = bias_log.read_text(encoding="utf-8")
    write_json(out_dir / f"cell{run_index}_summary.json", result)
    vol.commit()
    return result


@app.function(gpu=["L4", "A10G", "L40S"], timeout=60 * 60 * 2, **_FUNCTION_KW)
def smoke_train(
    model_key: str,
    trainable_mode: str,
    seed: int,
    run_index: int,
    run_id: str,
    max_updates: int = 30,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/project/src")
    import torch
    import yaml

    from habit.infra.logging import write_json
    from habit.infra.paths import CONFIGS
    from habit.model.loading import load_model, resolve_model_spec
    from habit.train.rl import train_rl

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cfg = yaml.safe_load(Path("/root/project/configs/train/smoke.yaml").read_text())
    cfg = {
        **cfg,
        "trainable_mode": trainable_mode,
        "seed": seed,
        "run_index": run_index,
        "max_updates": max_updates,
        "force_checkpoints": [max_updates],
        "name": f"smoke-{trainable_mode}",
    }
    import torch as _torch

    gpu_name = _torch.cuda.get_device_name(0) if _torch.cuda.is_available() else "cpu"
    # Map device name to budget.yaml key.
    if "L40" in gpu_name:
        gpu_key = "L40S"
    elif "A10" in gpu_name:
        gpu_key = "A10G"
    elif "L4" in gpu_name:
        gpu_key = "L4"
    else:
        gpu_key = "L4"
    model, tokenizer = load_model(model_key, device="cuda", dtype=torch.bfloat16)
    out_dir = Path(f"/vol/runs/{run_id}")
    summary = train_rl(
        model,
        tokenizer,
        cfg,
        out_dir,
        device="cuda",
        model_key=model_key,
        gpu_type=gpu_key,
    )
    # Copy checkpoint bytes digest already in manifest; ensure volume persistence.
    result = {
        "run_id": run_id,
        "model_key": model_key,
        "model_revision": resolve_model_spec(model_key)["revision"],
        "trainable_mode": trainable_mode,
        "summary": summary,
        "gpu": gpu_key,
        "gpu_name": gpu_name,
        "gpu_seconds": time.time() - t0,
        "peak_gpu_mem_gb": _gpu_mem_peak_gb(),
        "volume_path": f"/vol/runs/{run_id}",
    }
    # Include small text artifacts so the local launcher can commit them.
    for name in ["actions.jsonl", "metrics.jsonl", "config.json", "run_fixed.json", "train_summary.json", "checkpoint_manifest.json"]:
        p = out_dir / name
        if p.exists():
            result[f"artifact_{name}"] = p.read_text(encoding="utf-8")
    write_json(out_dir / "modal_result.json", result)
    vol.commit()
    return result


@app.function(gpu=["L4", "A10G", "L40S"], timeout=60 * 60 * 2, **_FUNCTION_KW)
def smoke_sft_clone(
    model_key: str,
    trainable_mode: str,
    seed: int,
    run_index: int,
    rl_run_id: str,
    clone_run_id: str,
    max_updates: int = 30,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/project/src")
    import torch
    import yaml

    from habit.infra.logging import write_json
    from habit.model.loading import load_model, resolve_model_spec
    from habit.train.sft_clone import train_sft_clone

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cfg = yaml.safe_load(Path("/root/project/configs/train/smoke.yaml").read_text())
    cfg = {
        **cfg,
        "trainable_mode": trainable_mode,
        "seed": seed,
        "run_index": run_index,
        "max_updates": max_updates,
        "force_checkpoints": [max_updates],
    }
    model, tokenizer = load_model(model_key, device="cuda", dtype=torch.bfloat16)
    rl_dir = Path(f"/vol/runs/{rl_run_id}")
    clone_dir = Path(f"/vol/runs/{clone_run_id}")
    summary = train_sft_clone(
        model,
        tokenizer,
        cfg,
        rl_dir,
        clone_dir,
        device="cuda",
        model_key=model_key,
        gpu_type="L4",
        max_updates=max_updates,
    )
    result = {
        "run_id": clone_run_id,
        "rl_run_id": rl_run_id,
        "summary": summary,
        "gpu": "L4",
        "gpu_seconds": time.time() - t0,
        "peak_gpu_mem_gb": _gpu_mem_peak_gb(),
        "model_revision": resolve_model_spec(model_key)["revision"],
    }
    for name in ["metrics.jsonl", "config.json", "train_summary.json", "checkpoint_manifest.json", "actions_consumed_sha256.txt"]:
        p = clone_dir / name
        if p.exists():
            result[f"artifact_{name}"] = p.read_text(encoding="utf-8")
    write_json(clone_dir / "modal_result.json", result)
    vol.commit()
    return result


@app.function(gpu=["L4", "A10G", "L40S"], timeout=60 * 60 * 2, **_FUNCTION_KW)
def smoke_battery_on_ckpt(
    model_key: str,
    trainable_mode: str,
    seed: int,
    run_index: int,
    run_id: str,
    update: int = 30,
) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/root/project/src")
    import torch
    import yaml

    from habit.analysis.metrics import load_jsonl, summarize_battery
    from habit.env.counterbalance import draw_run_fixed
    from habit.infra.logging import write_json
    from habit.model.loading import load_model
    from habit.test.battery import run_battery
    from habit.train.checkpoints import load_checkpoint
    from habit.train.params import apply_trainable_mode

    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cfg = yaml.safe_load(Path("/root/project/configs/train/smoke.yaml").read_text())
    cfg = {**cfg, "trainable_mode": trainable_mode, "seed": seed, "run_index": run_index}
    model, tokenizer = load_model(model_key, device="cuda", dtype=torch.bfloat16)
    model, _ = apply_trainable_mode(model, trainable_mode, cfg)
    ckpt = Path(f"/vol/runs/{run_id}/checkpoints/update_{update}")
    load_checkpoint(model, ckpt, strict=False)
    model.eval()
    fixed_path = Path(f"/vol/runs/{run_id}/run_fixed.json")
    if fixed_path.exists():
        from habit.env.counterbalance import RunFixed

        fixed = RunFixed(**json.loads(fixed_path.read_text()))
    else:
        fixed = draw_run_fixed(run_index, seed, tokenizer=tokenizer)
    out = Path(f"/vol/runs/{run_id}/test_results.jsonl")
    summary = run_battery(
        model,
        tokenizer,
        fixed,
        run_id=run_id,
        checkpoint=f"update_{update}",
        out_path=out,
        device="cuda",
    )
    metrics = summarize_battery(load_jsonl(out))
    result = {
        "run_id": run_id,
        "battery_summary": summary,
        "metrics": metrics,
        "gpu": "L4",
        "gpu_seconds": time.time() - t0,
        "peak_gpu_mem_gb": _gpu_mem_peak_gb(),
    }
    if out.exists():
        result["artifact_test_results.jsonl"] = out.read_text(encoding="utf-8")
    write_json(Path(f"/vol/runs/{run_id}/battery_summary.json"), result)
    vol.commit()
    return result
