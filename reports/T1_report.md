# T1 report — full harness, smoke tests, baseline-inference gate

## What was built

End-to-end study harness under `src/habit/`:

- **env** — nonce name pools (English-wordlist filtered), 4-cell counterbalancing, state generator, Set A/B templates, rewards (+1.0 / +0.6)
- **model** — Qwen3 load with pinned revisions, `enable_thinking=False`, exact candidate scoring, TinyQwenHabit for CPU
- **train** — group REINFORCE (no KL, no entropy), LoRA / last-4-MLP modes, criterion tracker, matched SFT clone
- **test** — 600-prompt battery + efficacy/mapping/competence probes (exact probs only)
- **analysis** — DS / ΔDS, sign-flip, paired bootstrap CI, TOST, Holm; `simulate.py` planted-effect checks
- **infra** — run ids, JSONL logs, Modal volume checkpoints, budget guard + ledger
- **Modal** — `modal_app.py` + launchers for §9 live runs

### Dependency versions (pinned in `pyproject.toml`)

| Package | Version |
|---|---|
| Python | 3.11.15 |
| torch | 2.6.0 |
| transformers | 4.51.3 |
| peft | 0.15.2 |
| modal | 1.5.5 |
| numpy | 2.2.4 |
| scipy | 1.15.2 |
| pandas | 2.2.3 |
| pyyaml | 6.0.2 |
| pydantic | 2.11.3 |
| pytest | 8.3.5 |
| accelerate | 1.6.0 |
| safetensors | 0.5.3 |
| huggingface-hub | 0.30.2 |

### Models

| Key | HF id | Pinned revision |
|---|---|---|
| qwen3_1_7b | Qwen/Qwen3-1.7B | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| qwen3_4b | Qwen/Qwen3-4B | `1cfa9a7208912126459214e8b04321603b3df60c` |

**CPU test model:** `TinyQwenHabit` — random `Qwen2ForCausalLM` (2 layers, hidden 64, vocab 1024) with a Qwen-style chat template that emits an empty `<think>\n\n</think>\n\n` block when thinking is disabled. See `configs/models.yaml` and `src/habit/model/tiny.py`.

## Template check and token ids

Full dump: `reports/T1_template_check.md` (live Qwen3-1.7B) and `reports/T1_template_check_tiny.md`.

With `enable_thinking=False`, the generation prompt still ends in an empty think block (Qwen3 chat template), then the answer position. Labels **2–9**, **yes**, **no** are single tokens:

| text | token_id |
|---|---|
| 2…9 | 17…24 |
| yes | 9693 |
| no | 2152 |

Restricted 2-way softmax equals renormalised full-vocab softmax (`max_abs_diff ≈ 6e-8`).

## Pytest summary

```
.................                                                        [100%]
17 passed in 203.48s (0:03:23)
```

(Saved also in `reports/T1_pytest_summary.txt`.) CPU integration (`scripts/run_cpu_integration.py`) completed in **31.7 s** (RL 20 → clone → battery → analysis on TinyQwenHabit).

## §9(a) Baseline-inference gate (C0)

Runs: `data/runs/20261104-052122-baseline-qwen3_1_7b-c8b657`, `data/runs/20261104-052455-baseline-qwen3_4b-8d2456`  
GPU: **L4**. Means over 4 counterbalancing cells.

| Metric | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|
| p(useful dock) deval_M1 (Set A) | 0.498 | 0.289 |
| p(useful dock) deval_M2 (Set A) | 0.500 | 0.657 |
| control p_dock_a | 0.470 | 0.790 |
| sham shift | −0.025 | 0.004 |
| label mass | 0.451 | 0.957 |
| efficacy p(no)\|deval_M1 | 0.806 | 0.854 |
| efficacy p(yes)\|control_M1 | 0.324 | 0.537 |
| efficacy p(no)\|deval_M2 | 0.531 | 0.846 |
| efficacy p(yes)\|control_M2 | 0.098 | 0.396 |
| mapping accuracy | 0.981 | 1.000 |
| competence accuracy | 0.750 | 0.875 |
| **Gate (≥0.80 useful in both devals, Set A)** | **FAIL** | **FAIL** |

Gate is reported only; no action taken.

## §9(b) Smoke training

| Run | Mode | GPU | s/update | $/update | Peak GPU mem | Battery after update 30 |
|---|---|---|---:|---:|---:|---|
| `…smoke-rl-lora-152c89` | LoRA | A10G | 26.56 | $0.0081 | 7.52 GB | control p_a≈1.00 |
| `…smoke-rl-last_mlp-ef7229` | last_mlp | A10G | 6.75 | $0.0021 | 4.74 GB | control p_a≈0.97 |
| `…smoke-sft-lora-31ace7` | SFT clone of LoRA | L4 | 22.94 | $0.0051 | 7.49 GB | (no battery required) |

Weights stayed on Modal volume `habitual-checkpoints` (manifests + sha256 in each run dir).

## Ledger (task spend)

`data/ledger_total.jsonl` total: **$0.597** (cap_task_usd = $5). Breakdown: template check $0.007; baseline 1.7B $0.025; baseline 4B $0.040; smoke LoRA+battery $0.282; smoke last_mlp+battery $0.081; SFT clone $0.162.

## Projected cost of one full pilot run

Assumptions from brief: C_crit ≈ 300, k = 6 → **1800** RL updates; + 1800 SFT clone updates; **5** battery passes. Use smoke LoRA timing/cost on A10G as the rate:

- RL: 1800 × 26.56 s ≈ 47 800 GPU-s ≈ **$14.6** (A10G @ $0.000306/s)
- Clone: 1800 × 22.94 s ≈ 41 300 GPU-s ≈ **$9.2** (L4 rate) or ≈ **$12.6** if A10G
- Batteries: 5 × ~50 s (batched 1.7B) ≈ 250 GPU-s ≈ **$0.06** (L4)

**≈ $24–28** for one 1.7B LoRA pilot twin (RL+clone+5 batteries) at smoke rates. last_mlp would be cheaper per update (~4×). Scale up for 4B / L40S if needed.

## Deviations from the brief (with reasons)

1. **Nonce material/product spellings** — Replaced brief examples that appear in `/usr/share/dict/words` (`vell`, `toma`, `rusk`, `tav`, `gribble`) with other pronounceable nonces so the English-wordlist exclusion test passes.
2. **Paired bootstrap CI** — Implemented as bootstrap-SE × t critical value (not percentile interval). Percentile under-covered (~92%) vs the acceptance band 93–97% at n=16; t+boot-SE recovers ~94.5% coverage under the planted-noise sim.
3. **Smoke GPU** — First LoRA attempt on L4 OOM’d before microbatching. After microbatched loss (`loss_microbatch=8`), Modal scheduled **A10G** from the preference list `[L4, A10G, L40S]` (brief allows A10 after L4; L40S not required). SFT clone ran on **L4**.
4. **Modal launch must be a real `.py` file** — `python -c` / stdin hangs because Modal tries to mount `<stdin>`. Launchers are file-based scripts.
5. **HF secret name** — Workspace secret is `hf-token` (not `huggingface`); documented in `.env.example` / `modal_app.py`.
6. **Training template** — RL uses Set A template **A1** as the training surface (Set A only; A2/A3 still in eval battery). Brief requires Set A in training; did not invent multi-template mixing without specification.
7. **Baseline gate FAIL** — Reported as required; not tuned.
8. **Empty think block with `enable_thinking=False`** — Observed on Qwen3-1.7B pinned revision; matches the brief’s “empty think block” answer position. Not altered.
