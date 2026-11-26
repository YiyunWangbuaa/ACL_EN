# -*- coding: utf-8 -*-
"""Default theory block lives in the preceding section of scene sample_03.

That section is the former basic-01 (shared default). Scene sample_01's preceding
section follows 025-001's cameras, so the theory gate does not read it.
"""
from __future__ import annotations

SAMPLE_ID = "03"
FIRST_ID = 1
LAST_ID = 66
NODE_IDS = tuple(str(number) for number in range(FIRST_ID, LAST_ID + 1))
PROCESS = "basics"
