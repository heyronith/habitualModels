from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from habit.env.counterbalance import RunFixed
from habit.env.names import load_env_config, validated_pools


FORBIDDEN = tuple(load_env_config()["forbidden_training_words"])


@dataclass(frozen=True)
class State:
    state_id: str
    order_id: int
    customer: str
    product: str
    q1: int
    q2: int
    s1: int
    s2: int
    docks_order: tuple[str, ...]  # ("a","b") or ("b","a") — which dock mentioned first
    M1: str
    M2: str
    dock_a: int
    dock_b: int

    @property
    def n1(self) -> int:
        return self.q1 - self.s1

    @property
    def n2(self) -> int:
        return self.q2 - self.s2

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["docks_order"] = list(self.docks_order)
        d["n1"] = self.n1
        d["n2"] = self.n2
        return d


def _state_id(payload: dict[str, Any]) -> str:
    blob = repr(sorted(payload.items())).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def contains_forbidden(text: str) -> bool:
    lower = text.lower()
    return any(w in lower for w in FORBIDDEN)


def validate_training_state(state: State) -> None:
    # Training: both materials still needed (n1, n2 ≥ 1) ⇒ s < q.
    if state.n1 < 1 or state.n2 < 1:
        raise ValueError(f"Training requires n1,n2 ≥ 1: {state}")
    if state.s1 >= state.q1 or state.s2 >= state.q2:
        raise ValueError(f"Stock must be < recipe quantity: {state}")
    if state.s1 < 0 or state.s2 < 0:
        raise ValueError("Negative stock")
    blob = " ".join(
        [
            str(state.order_id),
            state.customer,
            state.product,
            state.M1,
            state.M2,
            str(state.q1),
            str(state.q2),
            str(state.s1),
            str(state.s2),
        ]
    )
    if contains_forbidden(blob):
        raise ValueError(f"Forbidden word in training state fields: {blob}")


def derive_state_for_type(base: State, prompt_type: str) -> State:
    """Derive a typed state from a paired base control state (F3)."""
    if prompt_type == "control":
        return base
    if prompt_type == "deval_M1":
        # Full for M1: s1 = q1 ⇒ n1 = 0
        return replace(base, s1=base.q1, state_id=base.state_id + ":deval_M1")
    if prompt_type == "deval_M2":
        return replace(base, s2=base.q2, state_id=base.state_id + ":deval_M2")
    if prompt_type in ("sham", "spoil_M1", "degrade_M1"):
        return base
    raise ValueError(f"Unknown prompt_type {prompt_type}")


def generate_states(
    fixed: RunFixed,
    n: int,
    seed: int,
    *,
    training: bool = True,
) -> list[State]:
    cfg = load_env_config()
    _, products, customers = validated_pools()
    rng = np.random.default_rng(seed)
    qlo, qhi = cfg["quantity_range"]
    olo, ohi = cfg["order_id_range"]
    states: list[State] = []
    for i in range(n):
        q1 = int(rng.integers(qlo, qhi + 1))
        q2 = int(rng.integers(qlo, qhi + 1))
        # Always draw understocked base states (n ≥ 1); deval mutates later.
        s1 = int(rng.integers(0, q1))
        s2 = int(rng.integers(0, q2))
        docks_order = ("a", "b") if rng.random() < 0.5 else ("b", "a")
        payload = {
            "i": i,
            "order_id": int(rng.integers(olo, ohi + 1)),
            "customer": str(rng.choice(customers)),
            "product": str(rng.choice(products)),
            "q1": q1,
            "q2": q2,
            "s1": s1,
            "s2": s2,
            "docks_order": docks_order,
            "M1": fixed.M1,
            "M2": fixed.M2,
            "dock_a": fixed.dock_a,
            "dock_b": fixed.dock_b,
            "seed": seed,
        }
        st = State(
            state_id=_state_id(payload),
            order_id=payload["order_id"],
            customer=payload["customer"],
            product=payload["product"],
            q1=q1,
            q2=q2,
            s1=s1,
            s2=s2,
            docks_order=docks_order,
            M1=fixed.M1,
            M2=fixed.M2,
            dock_a=fixed.dock_a,
            dock_b=fixed.dock_b,
        )
        if training:
            validate_training_state(st)
        states.append(st)
    return states
