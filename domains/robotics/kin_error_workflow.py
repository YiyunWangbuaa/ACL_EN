# -*- coding: utf-8 -*-
"""Kinematics attribution of a real model trajectory.

Stage 5 checks the shared candidate report on control trajectories. It does
not build a value-level signature matrix.
"""
from __future__ import annotations


def diagnose_observations(answers: dict[str, str], *, prefix: str = "model"):
    """Model-prefix residual, then the candidate-set report."""
    from domains.robotics.kin_fault_diagnosis import annotate

    return annotate(answers, prefix=prefix)


def check_attribution() -> None:
    from domains.robotics.kin_aligned import check_stage5

    check_stage5()


if __name__ == "__main__":
    check_attribution()
