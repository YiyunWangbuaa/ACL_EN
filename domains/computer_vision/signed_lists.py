# -*- coding: utf-8 -*-
"""Human-signed checklists for vision basic sample_01."""
from __future__ import annotations

from domains.computer_vision.pipeline_scope import NODE_IDS

DERIVED_NODES = frozenset({
    "18", "19", "24", "25", "30", "31",
    "33", "34", "35", "36",
    "37", "38", "39", "40", "41",
    "47", "51", "53", "54", "55",
})

DECLARATION_ROOTS = frozenset(node for node in NODE_IDS if node not in DERIVED_NODES)

T_FINAL = ("41", "47", "53", "66")

EQUIVALENT_PAIRS = (
    ("[0, L, 0]", "[0,L,0]"),
    ("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[1,0,0],[0,1,0],[0,0,1]]"),
    ("[0, 0, 1]", "[0,0,1]"),
    ("X_w = X_c1", "X_w=X_c1"),
)
INEQUIVALENT_PAIRS = (
    ("主动", "被动"),
    ("[0, L, 0]", "[L, 0, 0]"),
    ("A", "B"),
    ("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[0, 1, 0], [1, 0, 0], [0, 0, 1]]"),
    ("X_w = X_c1", "X_w = R21·X_c1 + t21"),
)

DERIVED_EXCEPTIONS: tuple[tuple[str, str], ...] = ()
