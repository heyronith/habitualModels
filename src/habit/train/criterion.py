from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CriterionTracker:
    """C_crit = update index of the third consecutive evaluation with mean p(dock_a) ≥ threshold."""

    threshold: float = 0.95
    consecutive_needed: int = 3
    streak: int = 0
    c_crit: int | None = None
    history: list[tuple[int, float, bool]] | None = None

    def __post_init__(self) -> None:
        self.history = []

    def observe(self, update_idx: int, mean_p_dock_a: float) -> int | None:
        passed = mean_p_dock_a >= self.threshold
        self.history.append((update_idx, mean_p_dock_a, passed))
        if passed:
            self.streak += 1
        else:
            self.streak = 0
        if self.c_crit is None and self.streak >= self.consecutive_needed:
            self.c_crit = update_idx
        return self.c_crit

    def checkpoint_updates(self, k: int) -> list[int]:
        if self.c_crit is None:
            return []
        c = self.c_crit
        return sorted({c, 2 * c, 4 * c, k * c})
