# -*- coding: utf-8 -*-
"""Observation fault sets for the retired robotics-theory slice."""
from __future__ import annotations

from domains.observation import Case, TargetFix, annotate_trajectory, audit_control, observation_fixes
from domains.robotics.aligned import view
from domains.robotics.signed_lists import T_FINAL


def annotate(answers: dict[str, str], *, prefix: str = "model") -> tuple[Case, ...]:
    return annotate_trajectory(view(), answers, T_FINAL, prefix=prefix)


def repairs(answers: dict[str, str], *, prefix: str = "model") -> tuple[TargetFix, ...]:
    return observation_fixes(view(), answers, T_FINAL, prefix=prefix)


def check() -> dict:
    return audit_control(view(), T_FINAL)


if __name__ == "__main__":
    check()
