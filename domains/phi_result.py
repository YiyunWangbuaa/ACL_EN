# -*- coding: utf-8 -*-
"""Separate a defined reference value from a logical contradiction.

``None`` means the rule did not run: no calculator, an unparseable parent, or
an infrastructure failure. ``Contradiction`` means the rule is defined and
rejects this parent assignment, including a formula whose inputs leave its
domain. Callers that only need a string keep using ``answer_under``.
"""
from __future__ import annotations


class Contradiction:
    def __init__(self, reason: str = ""):
        self.reason = reason


_INSTRUMENT = ("无法解析", "没有计算器", "不在 1–", "不是 3×3", "不是三维")


def from_execution(execution):
    """String value, ``Contradiction``, or ``None`` when the rule did not apply."""
    if execution is None:
        return None
    if execution.ok:
        return execution.matrix
    reason = execution.reason or ""
    if execution.status != "contradictory":
        return None
    if any(token in reason for token in _INSTRUMENT):
        return None
    return Contradiction(reason)
