# -*- coding: utf-8 -*-
"""Stage-4 entry for the retired robotics-theory slice.

The live kinematics manuscript uses ``kin_dmatrix``. This module keeps the
same entry name as the other primary manuscripts.
"""
from __future__ import annotations


def check_dmatrix() -> None:
    from domains.robotics.aligned import check_stage4

    check_stage4()


if __name__ == "__main__":
    check_dmatrix()
