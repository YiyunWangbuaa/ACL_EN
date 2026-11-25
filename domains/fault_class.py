# -*- coding: utf-8 -*-
"""Fault classes. The token is the abbreviation of the English keyword.

Recognition is one ability at two depths. Execution and algebra are later
abilities. Residual has no injector and is not a library row.

DIR  Direct   the contract states the mapping; reading it is enough
RSN  Reason   the contract implies a route; it has to be understood
EXE  Execute  the recognized contract is carried out
ALG  Algebra  a written quantity is calculated wrong
RES  Residual unmodeled; nothing is injected
"""
from __future__ import annotations

DIR = "DIR"
RSN = "RSN"
EXE = "EXE"
ALG = "ALG"
RES = "RES"

RECOGNITION = frozenset({DIR, RSN})
ROW_CLASSES = frozenset({DIR, RSN, EXE, ALG})
MODES = (DIR, RSN, EXE, ALG, RES)
# Classes that can write the same blank. Algebra is not compared with them.
SITE_CLASSES = frozenset({DIR, RSN, EXE})
