# -*- coding: utf-8 -*-
"""Stage-4 entry for kinematics. The D matrix is binary graph reach.

This module does not inject a wrong answer or store a predicted value.
"""
from __future__ import annotations


def check_dmatrix() -> None:
    from domains.robotics.kin_aligned import check_stage4

    check_stage4()


if __name__ == "__main__":
    check_dmatrix()
