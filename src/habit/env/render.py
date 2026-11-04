from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from habit.env.state import State, contains_forbidden
from habit.infra.paths import PROMPTS


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
    test_lines = test_lines or load_test_lines()
    spec = test_lines.get(prompt_type, {})
    out = user
    if "storeroom_override" in spec:
        override = spec["storeroom_override"].format(**fields).rstrip()
        # Replace any Storeroom / Stock line variants.
        lines = out.splitlines()
        replaced = False
        new_lines = []
        for line in lines:
            if line.startswith("Storeroom:") or line.startswith("Stock on hand") or line.startswith(
                "Current storeroom:"
            ) or line.startswith("Storeroom levels:") or line.startswith("Storeroom reading"):
                new_lines.append(override)
                replaced = True
            else:
                new_lines.append(line)
        if not replaced:
            new_lines.append(override)
        out = "\n".join(new_lines)
    if "add_line" in spec:
        out = out.rstrip() + "\n" + spec["add_line"].format(**fields)
    return out


def render_prompt(
    state: State,
    template_id: str,
    *,
    prompt_type: str = "control",
    dock_x: int | None = None,
    templates: dict[str, dict[str, Any]] | None = None,
) -> RenderedPrompt:
    templates = templates or load_templates()
    t = templates[template_id]
    fields = render_fields(state, dock_x=dock_x)
    system = t["system"].format(**fields)
    user = t["user"].format(**fields).rstrip()
    user = apply_test_type(user, fields, prompt_type)
    if prompt_type == "control" and contains_forbidden(user):
        raise ValueError(f"Forbidden word leaked into control prompt: {user}")
    return RenderedPrompt(
        template_id=template_id,
        set=t["set"],
        system=system,
        user=user,
        prompt_type=prompt_type,
        sha256=_sha(system, user),
    )
