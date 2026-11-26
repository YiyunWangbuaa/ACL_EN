# -*- coding: utf-8 -*-
"""Full flight manuscript: sample_01, blanks 1–131."""
from __future__ import annotations

SAMPLE_ID = "01"
FIRST_ID = 1
LAST_ID = 131
COORD_LAST = 59
LOAD_FIRST = 60
NODE_IDS = tuple(str(number) for number in range(FIRST_ID, LAST_ID + 1))
COORD_IDS = tuple(str(number) for number in range(FIRST_ID, COORD_LAST + 1))
LOAD_IDS = tuple(str(number) for number in range(LOAD_FIRST, LAST_ID + 1))
PROCESS = "spatial transforms"
