from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"
PROMPTS = REPO_ROOT / "prompts"
DATA = REPO_ROOT / "data"
RUNS = DATA / "runs"
REPORTS = REPO_ROOT / "reports"
LEDGER_TOTAL = DATA / "ledger_total.jsonl"
