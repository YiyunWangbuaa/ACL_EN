# -*- coding: utf-8 -*-
"""Which fault is in scope for a question. The ability decides, not the answer shape.

DIR is direct recognition. RSN is a route the contract implies. EXE writes a
recognized contract into its object: a rotation, an alignment, a slot code, a
sign rule. ALG is only calculation: products, component extraction, and
simplification of objects execution already produced.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.contract_verifiers import input_ids, validator_of
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.error_workflow import PHI
from domains.wind_tunnel.fault_layers import LEVELS
from domains.wind_tunnel.multi_slot import _BRANCH_SLOTS

_BRANCH = frozenset(slot for slots in _BRANCH_SLOTS.values() for slot in slots)

# Contract values in, the first mathematical object out.
_APPLICATION = frozenset({
    "axis_alignment",
    "mount_matrix",
    "installation_matrix",
    "balance_forward",
    "reference_coordinate",
    "roll_difference",
    "sense_sign",
    "moment_sign",
    "expression_sign",
})

# Combining or simplifying objects that were already applied.
_CALCULATION = frozenset({
    "total_matrix_product",
    "airspeed_component",
    "angle_formula",
    "load_component",
    "moment_coefficient",
    "wind_component",
    "elastic_angle",
    "wind_ratio",
    "level_airspeed",
    "level_airspeed_unit",
    "deformed_axis",
})

# Identification checked by a calculator. The fault stays on the recognition layer.
_RECOGNITION_CHECK = frozenset({
    "axis_choice",
    "turn_paired",
    "declared_direction",
    "path_order",
})

_SCOPE_NAME = {
    DIR: "direct recognition",
    RSN: "reasoning recognition",
    EXE: "execution",
    ALG: "algebra",
}


def _bare(value: str) -> str:
    return str(value).removeprefix("ID")


def scope_of(node_id: str) -> str:
    """The only fault mode this question is allowed to carry locally."""
    node = _bare(node_id)
    if node in _BRANCH:
        return RSN
    if node in PHI:
        return EXE
    kind = validator_of(node)
    # A slot list with no earlier order question is the order recognition itself.
    if kind == "product_order":
        return EXE if input_ids(node) else DIR
    if kind in _APPLICATION:
        return EXE
    if kind == "formula_part":
        answer = NODE_BY_ID[node].answer_type
        if answer in {"choice", "sign"}:
            # No parent means this blank is the first statement of that token.
            return EXE if NODE_BY_ID[node].depends_on else DIR
        return ALG
    if kind == "fixed_term":
        # The distance is a sum. A token with no parent is the recognition of that token.
        if node == "182":
            return ALG
        return EXE if NODE_BY_ID[node].depends_on else DIR
    if kind in _CALCULATION:
        return ALG
    if kind in _RECOGNITION_CHECK or kind is None:
        return DIR
    raise RuntimeError(f"ID{node} validator {kind} has no evaluation layer")


def scope_name(node_id: str) -> str:
    return _SCOPE_NAME[scope_of(node_id)]


def audit_scopes() -> dict[str, int]:
    """Every question has exactly one evaluation layer."""
    counts = {name: 0 for name in _SCOPE_NAME}
    for node in LEVELS:
        if node not in TARGETS:
            raise RuntimeError(f"ID{node} is in the layer table but not in the question table.")
        counts[scope_of(node)] += 1
    missing = sorted(set(TARGETS) - set(LEVELS), key=int)
    if missing:
        raise RuntimeError(f"These questions have no evaluation layer: {missing}")
    if sum(counts.values()) != len(LEVELS):
        raise RuntimeError("The evaluation layers do not add up to the question count.")
    return counts


if __name__ == "__main__":
    found = audit_scopes()
    print("scopes", found)
    samples = ("8", "26", "43", "47", "55", "69", "75", "80", "85", "111", "132", "182", "195", "23", "24", "4")
    for node in samples:
        print(f"ID{node} {scope_of(node)} {scope_name(node)} {NODE_BY_ID[node].role}")
