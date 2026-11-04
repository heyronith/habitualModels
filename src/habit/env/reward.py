from __future__ import annotations

from habit.env.counterbalance import RunFixed

# Reward rationale (training only; never shown to the model):
# Choosing dock_a delivers M1 (the trained-preferred material) and yields +1.0.
# Choosing dock_b delivers M2 and yields +0.6.
# Both deliveries are useful for completing orders; the asymmetry exists solely
# to build a trained preference for dock_a / M1 under RL.


def reward_for_action(fixed: RunFixed, action_label: int) -> float:
    if action_label == fixed.dock_a:
        return float(fixed.reward_a)
    if action_label == fixed.dock_b:
        return float(fixed.reward_b)
    raise ValueError(f"Unknown action label {action_label}; expected {fixed.dock_a} or {fixed.dock_b}")
