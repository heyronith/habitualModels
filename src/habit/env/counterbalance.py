from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from habit.env.names import load_env_config, validated_pools


@dataclass(frozen=True)
class RunFixed:
    """Fixed per-run environment drawn from seed + counterbalancing cell."""

    run_index: int
    cell: int
    seed: int
    M1: str
    M2: str
    dock_a: int
    dock_b: int
    reward_a: float
    reward_b: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cell_for_run(run_index: int) -> int:
    return int(run_index) % 4


def draw_run_fixed(
    run_index: int,
    seed: int,
    *,
    tokenizer=None,
) -> RunFixed:
    """Draw M1/M2 and dock labels for a run.

    If tokenizer is provided, enforce distinct first tokens for material names
    and single-token dock labels at the answer position.
    """
    cfg = load_env_config()
    materials, _, _ = validated_pools()
    cell = cell_for_run(run_index)
    cell_cfg = cfg["counterbalance_cells"][cell]
    rng = np.random.default_rng(seed)

    # Sample two distinct materials; optionally require different first tokens.
    for _ in range(10_000):
        pair = rng.choice(materials, size=2, replace=False)
        m_first, m_second = str(pair[0]), str(pair[1])
        if tokenizer is not None:
            t1 = tokenizer.encode(m_first, add_special_tokens=False)
            t2 = tokenizer.encode(m_second, add_special_tokens=False)
            if not t1 or not t2 or t1[0] == t2[0]:
                continue
        break
    else:
        raise RuntimeError("Could not sample materials with distinct first tokens")

    if cell_cfg["m1_name_slot"] == 0:
        M1, M2 = m_first, m_second
    else:
        M1, M2 = m_second, m_first

    labels = list(cfg["dock_label_pool"])
    for _ in range(10_000):
        a, b = (int(x) for x in rng.choice(labels, size=2, replace=False))
        if a == b:
            continue
        if tokenizer is not None:
            # Single token when appearing as the answer string.
            for lab in (a, b):
                ids = tokenizer.encode(str(lab), add_special_tokens=False)
                if len(ids) != 1:
                    break
            else:
                break
        else:
            break
    else:
        raise RuntimeError("Could not sample single-token dock labels")

    lower, higher = (a, b) if a < b else (b, a)
    if cell_cfg["dock_a_lower"]:
        dock_a, dock_b = lower, higher
    else:
        dock_a, dock_b = higher, lower

    return RunFixed(
        run_index=run_index,
        cell=cell,
        seed=seed,
        M1=M1,
        M2=M2,
        dock_a=dock_a,
        dock_b=dock_b,
        reward_a=float(cfg["reward"]["dock_a"]),
        reward_b=float(cfg["reward"]["dock_b"]),
    )
