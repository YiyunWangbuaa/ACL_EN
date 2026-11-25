# -*- coding: utf-8 -*-
"""Recognition blanks whose answers form the wind-tunnel contract cognition.

A blank is recognition when no earlier answer is an input: either it has no
calculator, or its calculator only restates the sample declaration. Attitude
matrices are not recognition; they read this layer.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def recognition_ids() -> frozenset[str]:
    from domains.wind_tunnel.contract_verifiers import input_ids, validator_of
    from domains.wind_tunnel.error_graph import NODE_BY_ID
    from domains.wind_tunnel.error_workflow import PHI

    found = []
    for node in NODE_BY_ID:
        if node in PHI:
            continue
        if validator_of(node) is None or not input_ids(node):
            found.append(node)
    return frozenset(found)
