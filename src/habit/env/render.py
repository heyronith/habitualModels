from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import yaml

from habit.env.state import State, contains_forbidden, derive_state_for_type
from habit.infra.paths import PROMPTS


def load_variants() -> dict[str, dict[str, Any]]:
    return yaml.safe_load((PROMPTS / "variants.yaml").read_text())


def load_templates() -> dict[str, dict[str, Any]]:
    train = yaml.safe_load((PROMPTS / "train_templates.yaml").read_text())
    held = yaml.safe_load((PROMPTS / "heldout_templates.yaml").read_text())
    out: dict[str, dict[str, Any]] = {}
    for tid, t in train["templates"].items():
        out[tid] = {**t, "set": "A", "template_id": tid}
    for tid, t in held["templates"].items():
        out[tid] = {**t, "set": "B", "template_id": tid}
    return out


def load_test_lines() -> dict[str, Any]:
    return yaml.safe_load((PROMPTS / "test_lines.yaml").read_text())["types"]


def docks_line(state: State) -> str:
    parts = []
    for which in state.docks_order:
        if which == "a":
            parts.append(f"Dock {state.dock_a} supplies {state.M1}")
        else:
            parts.append(f"Dock {state.dock_b} supplies {state.M2}")
    return ". ".join(parts)


def render_fields(
    state: State,
    *,
    dock_x: int | None = None,
) -> dict[str, Any]:
    return {
        "order_id": state.order_id,
        "customer": state.customer,
        "product": state.product,
        "q1": state.q1,
        "q2": state.q2,
        "s1": state.s1,
        "s2": state.s2,
        "n1": state.n1,
        "n2": state.n2,
        "M1": state.M1,
        "M2": state.M2,
        "dock_a": state.dock_a,
        "dock_b": state.dock_b,
        "dock_x": dock_x if dock_x is not None else state.dock_a,
        "docks_line": docks_line(state),
    }


@dataclass
class RenderedPrompt:
    template_id: str
    set: str
    variant: str
    system: str
    user: str
    prompt_type: str
    sha256: str

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def _sha(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()


def apply_test_type(
    user: str,
    fields: dict[str, Any],
    prompt_type: str,
    test_lines: dict[str, Any] | None = None,
) -> str:
    """Apply additive test-only lines (sham/spoil/degrade). Deval is state mutation."""
    test_lines = test_lines or load_test_lines()
    spec = test_lines.get(prompt_type, {})
    out = user
    if "add_line" in spec:
        out = out.rstrip() + "\n" + spec["add_line"].format(**fields)
    return out


def render_prompt(
    state: State,
    template_id: str,
    *,
    prompt_type: str = "control",
    variant: str = "G",
    dock_x: int | None = None,
    templates: dict[str, dict[str, Any]] | None = None,
    variants: dict[str, dict[str, Any]] | None = None,
) -> RenderedPrompt:
    templates = templates or load_templates()
    variants = variants or load_variants()
    if variant not in variants:
        raise KeyError(f"Unknown variant {variant}; known={sorted(variants)}")
    t = templates[template_id]
    typed = derive_state_for_type(state, prompt_type)
    fields = render_fields(typed, dock_x=dock_x)
    system = variants[variant]["system"]
    user = t["user"].format(**fields).rstrip()
    user = apply_test_type(user, fields, prompt_type)
    if prompt_type == "control":
        if contains_forbidden(user):
            raise ValueError(f"Forbidden word leaked into control prompt: {user}")
        # Training/control must never show zero still-needed for either material.
        if typed.n1 < 1 or typed.n2 < 1:
            raise ValueError("Control/training prompt has n=0")
        zero_pat_m1 = f"0 {typed.M1}"
        zero_pat_m2 = f"0 {typed.M2}"
        # Still-needed line must not claim 0 of a material on control.
        for line in user.splitlines():
            low = line.lower()
            if "still needed" in low or low.startswith("remaining") or low.startswith("still needed"):
                if zero_pat_m1 in line or zero_pat_m2 in line:
                    raise ValueError(f"Zero still-needed leaked into control: {line}")
    return RenderedPrompt(
        template_id=template_id,
        set=t["set"],
        variant=variant,
        system=system,
        user=user,
        prompt_type=prompt_type,
        sha256=_sha(system, user),
    )
