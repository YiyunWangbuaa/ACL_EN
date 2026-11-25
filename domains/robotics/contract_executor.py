# -*- coding: utf-8 -*-
"""Shared execution result for robotics calculators."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Execution:
    status: str
    matrix: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "derived"
