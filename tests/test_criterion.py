from __future__ import annotations

from habit.train.criterion import CriterionTracker


def test_third_consecutive():
    t = CriterionTracker(threshold=0.95, consecutive_needed=3)
    seq = [0.5, 0.96, 0.97, 0.99, 0.5]
    updates = [10, 20, 30, 40, 50]
    c = None
    for u, p in zip(updates, seq):
        c = t.observe(u, p)
    assert c == 40  # third consecutive pass at update 40


def test_non_consecutive_resets():
    t = CriterionTracker()
    t.observe(10, 0.96)
    t.observe(20, 0.97)
    t.observe(30, 0.50)  # break
    t.observe(40, 0.99)
    t.observe(50, 0.99)
    assert t.c_crit is None
    t.observe(60, 0.99)
    assert t.c_crit == 60


def test_never_reaches():
    t = CriterionTracker()
    for u in range(1, 21):
        t.observe(u * 10, 0.5)
    assert t.c_crit is None
    assert t.checkpoint_updates(6) == []


def test_checkpoint_schedule():
    t = CriterionTracker()
    t.observe(10, 0.96)
    t.observe(20, 0.97)
    t.observe(30, 0.98)
    assert t.c_crit == 30
    assert t.checkpoint_updates(6) == [30, 60, 120, 180]
