from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from habit.infra.logging import append_jsonl, utc_now_iso
from habit.infra.paths import CONFIGS, LEDGER_TOTAL


def load_budget() -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / "budget.yaml").read_text())


def spent_usd(ledger_path: Path | None = None) -> float:
    path = ledger_path or LEDGER_TOTAL
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        total += float(row.get("cost_usd", 0.0))
    return total


def gpu_price(gpu: str) -> float:
    budget = load_budget()
    prices = budget["gpu_prices_usd_per_second"]
    if gpu not in prices:
        raise KeyError(f"Unknown GPU type {gpu!r}; known={sorted(prices)}")
    return float(prices[gpu])


def project_cost(gpu: str, seconds: float) -> float:
    return gpu_price(gpu) * float(seconds)


def assert_within_budget(projected_usd: float, *, task_only: bool = True) -> None:
    budget = load_budget()
    spent = spent_usd()
    cap = float(budget["cap_task_usd"] if task_only else budget["cap_total_usd"])
    total_cap = float(budget["cap_total_usd"])
    if spent + projected_usd > cap + 1e-9:
        raise RuntimeError(
            f"Budget guard: projected ${projected_usd:.4f} + spent ${spent:.4f} "
            f"exceeds task/total cap ${cap:.2f}"
        )
    if spent + projected_usd > total_cap + 1e-9:
        raise RuntimeError(
            f"Budget guard: projected ${projected_usd:.4f} + spent ${spent:.4f} "
            f"exceeds cap_total_usd ${total_cap:.2f}"
        )


def record_spend(
    *,
    run_id: str,
    gpu: str,
    seconds: float,
    note: str,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    cost = project_cost(gpu, seconds)
    row = {
        "ts": utc_now_iso(),
        "run_id": run_id,
        "gpu": gpu,
        "gpu_seconds": seconds,
        "usd_per_second": gpu_price(gpu),
        "cost_usd": cost,
        "note": note,
    }
    append_jsonl(LEDGER_TOTAL, row)
    if run_dir is not None:
        append_jsonl(run_dir / "ledger.jsonl", row)
    return row
