# -*- coding: utf-8 -*-
"""Human-signed checklists for robotics theory sample_01."""
from __future__ import annotations

from domains.robotics.pipeline_scope import NODE_IDS

DERIVED_NODES = frozenset({
    "F4", "F7", "F8", "F9", "F11", "F13",
    "I6", "I8", "I10",
})

DECLARATION_ROOTS = frozenset(node for node in NODE_IDS if node not in DERIVED_NODES)

T_FINAL = ("F4", "F11", "F13", "I8", "I10")

EQUIVALENT_PAIRS = (
    ("[1, 0, 0]", "[1,0,0]"),
    ("D*[L, 0, 0]", "D*[L,0,0]"),
    (
        "[[sqrt(3)/2, -sqrt(3)/4, 1/4], [1/2, 3/4, -sqrt(3)/4], [0, 1/2, sqrt(3)/2]]",
        "[[sqrt(3)/2,-sqrt(3)/4,1/4],[1/2,3/4,-sqrt(3)/4],[0,1/2,sqrt(3)/2]]",
    ),
    ("arctan(1/2), arcsin(1/4), arcsin(1/2)", "arctan(1/2),arcsin(1/4),arcsin(1/2)"),
)
INEQUIVALENT_PAIRS = (
    ("positive", "negative"),
    ("[1, 0, 0]", "[0, 1, 0]"),
    ("A", "B"),
    ("[0, 30, -60]", "[0, -30, 60]"),
    ("D*[L, 0, 0]", "D^T*[L, 0, 0]"),
)

DERIVED_EXCEPTIONS: tuple[tuple[str, str], ...] = ()
