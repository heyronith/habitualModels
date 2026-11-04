# Habitual Models

Harness for RL vs matched no-reward SFT on a simulated workshop dock-choice task, with a fixed exact-probability test battery and preregistered analysis.

## Setup (Python 3.11)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env   # fill HF_TOKEN if needed; Modal via `modal token set` / ~/.modal.toml
```

Pinned dependency versions live in `pyproject.toml` and are recorded in `reports/T1_report.md`.

## Reproduce

```bash
# Unit / acceptance tests (offline)
pytest -q

# Template + token check (tiny CPU model)
python scripts/template_check.py --model-key tiny

# CPU integration (<10 min): RL → clone → battery → analysis on TinyQwenHabit
python scripts/run_cpu_integration.py

# Live T1 §9 Modal runs (requires clean git tree; ≤ $5 task cap)
python scripts/launch_t1_live.py
```

## CPU test model

`TinyQwenHabit`: random `Qwen2ForCausalLM` (2 layers, hidden 64) with a Qwen-style chat template that emits an empty `<think>` block when `enable_thinking=False`. See `configs/models.yaml` and `src/habit/model/tiny.py`.

## Layout

See the T1 brief. Run artifacts: `data/runs/<run_id>/` (logs/results committed; weights on Modal Volume). Budget: `configs/budget.yaml`.
