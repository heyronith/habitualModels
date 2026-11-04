from __future__ import annotations

from habit.model.thinking import (
    DOCK_REPLY_SENTENCE,
    cap_and_close_think,
    messages_for_thinking,
    mentions_devalued_not_needed,
    system_for_thinking,
    thinking_system_render_example,
)


def test_cap_and_close_natural():
    gen = "<think>\nNeed dock 5.\n</think>\n\nSomething"
    prefix, meta = cap_and_close_think(gen, hit_cap=False)
    assert prefix.endswith("</think>\n\n")
    assert meta["closed_naturally"] is True
    assert meta["forced_close"] is False
    assert "Something" not in prefix


def test_cap_and_close_forced_on_cap():
    gen = "<think>\nStill reasoning about materials"
    prefix, meta = cap_and_close_think(gen, max_new_tokens=256, hit_cap=True)
    assert prefix.endswith("\n</think>\n\n")
    assert meta["forced_close"] is True
    assert meta["hit_cap"] is True
    assert "Still reasoning" in prefix


def test_cap_and_close_incomplete_without_flag():
    gen = "<think>\nincomplete"
    prefix, meta = cap_and_close_think(gen)
    assert "</think>" in prefix
    assert meta["forced_close"] is True


def test_thinking_system_drops_dock_only():
    ex = thinking_system_render_example("R")
    assert DOCK_REPLY_SENTENCE in ex["original"]
    assert DOCK_REPLY_SENTENCE not in ex["thinking"]
    assert "Only send the cart for a material the order still needs" in ex["thinking"]
    msgs = messages_for_thinking(
        [
            {"role": "system", "content": ex["original"]},
            {"role": "user", "content": "hi"},
        ]
    )
    assert DOCK_REPLY_SENTENCE not in msgs[0]["content"]


def test_mentions_devalued_regex():
    assert mentions_devalued_not_needed("dax is not needed because storeroom is full", "dax")
    assert not mentions_devalued_not_needed("we should fetch more dax from the dock", "dax")
