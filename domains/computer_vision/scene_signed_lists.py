# -*- coding: utf-8 -*-
"""Human-signed checklists for vision scene sample_01."""
from __future__ import annotations

from domains.computer_vision.scene_pipeline_scope import NODE_IDS

DERIVED_NODES = frozenset({
    "18", "19", "24", "25", "30", "31",
    "33", "34", "35", "36",
    "37", "38", "39", "40", "41", "43",
    "61", "62", "67", "68", "73", "74",
    "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "86",
    "100",
})

DECLARATION_ROOTS = frozenset(node for node in NODE_IDS if node not in DERIVED_NODES)

T_FINAL = ("41", "43", "84", "86", "165")

EQUIVALENT_PAIRS = (
    ("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[1,0,0],[0,1,0],[0,0,1]]"),
    ("[0, L, 0]", "[0,L,0]"),
    ("[[0, 1, 0], [0, 0, 1], [1, 0, 0]]", "[[0,1,0],[0,0,1],[1,0,0]]"),
    ("X_w = X_c1", "X_w=X_c1"),
)
INEQUIVALENT_PAIRS = (
    ("主动", "被动"),
    ("[0, L, 0]", "[L, 0, 0]"),
    ("A", "B"),
    ("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[0, 1, 0], [1, 0, 0], [0, 0, 1]]"),
    ("正向安置", "反向安置"),
)

DERIVED_EXCEPTIONS: tuple[tuple[str, str], ...] = ()
