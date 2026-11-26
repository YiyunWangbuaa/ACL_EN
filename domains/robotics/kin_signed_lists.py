# -*- coding: utf-8 -*-
"""Human-signed checklists for robotics kinematics sample_01."""
from __future__ import annotations

from domains.robotics.kin_pipeline_scope import NODE_IDS

# Declaration roots stay the blanks that restate the definition.
# An application blank is derived when a later blank reads the object it writes:
# the joint axis, an alignment column, a component sign, or the assembled matrix.
DERIVED_NODES = frozenset({
    "F11", "F13", "F14", "F16", "F18", "F19", "F21", "F23", "F24",
    "F27", "F28", "F29", "F30", "F31", "F32", "F33", "F34", "F35", "F39", "F40",
    "F41", "F42", "F43", "F44",
    "F45", "F48", "F50", "F52", "F53", "F59", "F60", "F61", "F62", "F63",
    "F72", "F73", "F74", "F75", "F77", "F78", "F80", "F81", "F82", "F84", "F85",
    "F92", "F94", "F96", "F98",
    "I7", "I9", "I12", "I13", "I16",
})

DECLARATION_ROOTS = frozenset(node for node in NODE_IDS if node not in DERIVED_NODES)

T_FINAL = ("F45", "F48", "F98", "I9", "I16")

EQUIVALENT_PAIRS = (
    ("[1, 0, 0]", "[1,0,0]"),
    ("D*[L + L_mount, 0, 0]", "D*[L+L_mount,0,0]"),
    (
        "[[sqrt(3)/2, -sqrt(3)/4, 1/4], [1/2, 3/4, -sqrt(3)/4], [0, 1/2, sqrt(3)/2]]",
        "[[sqrt(3)/2,-sqrt(3)/4,1/4],[1/2,3/4,-sqrt(3)/4],[0,1/2,sqrt(3)/2]]",
    ),
    ("L + L_mount", "L+L_mount"),
)
INEQUIVALENT_PAIRS = (
    ("positive", "negative"),
    ("[1, 0, 0]", "[0, 1, 0]"),
    ("A", "B"),
    ("[0, 30, -60]", "[0, -30, 60]"),
    ("D*[L + L_mount, 0, 0]", "D^T*[L + L_mount, 0, 0]"),
)

DERIVED_EXCEPTIONS: tuple[tuple[str, str], ...] = ()
