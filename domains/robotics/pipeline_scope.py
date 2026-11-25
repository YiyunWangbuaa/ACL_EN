# -*- coding: utf-8 -*-
"""First robotics block: theory forward + inverse on sample_01."""
from __future__ import annotations

SAMPLE_ID = "01"
FORWARD_IDS = tuple(f"F{n}" for n in range(1, 14))
INVERSE_IDS = tuple(f"I{n}" for n in range(1, 11))
NODE_IDS = FORWARD_IDS + INVERSE_IDS
PROCESS_FORWARD = "正运动学"
PROCESS_INVERSE = "逆运动学"
