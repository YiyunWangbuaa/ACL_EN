# -*- coding: utf-8 -*-
"""Observation fault sets for the vision object-and-stereo section."""
from __future__ import annotations

from domains.computer_vision.scene_aligned import view
from domains.computer_vision.scene_signed_lists import T_FINAL
from domains.observation import Case, TargetFix, annotate_trajectory, audit_control, observation_fixes


def _baseline_gold() -> dict[str, str]:
    import domains.computer_vision.scene_manuscript as manuscript

    saved = manuscript.SAMPLE_ID
    manuscript.SAMPLE_ID = "01"
    manuscript.questions.cache_clear()
    manuscript.gold_texts.cache_clear()
    manuscript.gold_scored.cache_clear()
    try:
        return dict(manuscript.gold_texts())
    finally:
        manuscript.SAMPLE_ID = saved
        manuscript.questions.cache_clear()
        manuscript.gold_texts.cache_clear()
        manuscript.gold_scored.cache_clear()


def annotate(answers: dict[str, str], *, prefix: str = "model") -> tuple[Case, ...]:
    return annotate_trajectory(view(), answers, T_FINAL, prefix=prefix, baseline=_baseline_gold())


def repairs(answers: dict[str, str], *, prefix: str = "model") -> tuple[TargetFix, ...]:
    return observation_fixes(view(), answers, T_FINAL, prefix=prefix)


def check() -> dict:
    return audit_control(view(), T_FINAL)


if __name__ == "__main__":
    check()
