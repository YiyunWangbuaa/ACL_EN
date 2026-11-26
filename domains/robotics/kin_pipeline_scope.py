# -*- coding: utf-8 -*-
"""Robotics kinematics block: forward F1–F140 + inverse I1–I16."""
from __future__ import annotations

SAMPLE_ID = "01"
FORWARD_IDS = tuple(f"F{n}" for n in range(1, 141))
INVERSE_IDS = tuple(f"I{n}" for n in range(1, 17))
NODE_IDS = FORWARD_IDS + INVERSE_IDS
PROCESS_FORWARD = "forward kinematics"
PROCESS_INVERSE = "inverse kinematics"
