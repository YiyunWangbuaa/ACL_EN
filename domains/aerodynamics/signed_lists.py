# -*- coding: utf-8 -*-
"""Human-signed checklists for the full flight manuscript (1–131)."""
from __future__ import annotations

from domains.aerodynamics.pipeline_scope import NODE_IDS

# Blanks whose calculator reads earlier answers.
DERIVED_NODES = frozenset({
    "15", "16", "28", "29", "31", "32", "33", "34", "45",
    "46", "47", "49", "50", "55", "56",
    "72", "73", "75", "76",
    "79", "80", "81", "83", "91", "96",
    "107", "109", "110",
    "114", "115", "116", "117", "118", "119",
    "124", "125", "126",
})

# Everything else is a declaration root or a fixed open answer from the sample.
DECLARATION_ROOTS = frozenset(node for node in NODE_IDS if node not in DERIVED_NODES)

# Endpoints used by system repair (coordinate + loads + reference).
T_FINAL = ("15", "34", "45", "55", "59", "81", "83", "90", "120", "131")

EQUIVALENT_PAIRS = (
    ("[1, 0, 0]", "[1,0,0]"),
    ("Cz(-ψ)*Cy(-θ)*Cx(-Φ)", "Cz(-ψ) * Cy(-θ) * Cx(-Φ)"),
    ("atan2(vz, vx)", "atan2(vz,vx)"),
    ("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[1,0,0],[0,1,0],[0,0,1]]"),
)
INEQUIVALENT_PAIRS = (
    ("上方", "下方"),
    ("[1, 0, 0]", "[0, 1, 0]"),
    ("Cy(α)", "Cy(-α)"),
    ("0", "1"),
    ("A", "G"),
)

DERIVED_EXCEPTIONS: tuple[tuple[str, str], ...] = ()
