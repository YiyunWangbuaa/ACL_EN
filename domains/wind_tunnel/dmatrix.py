# -*- coding: utf-8 -*-
"""Stage-4 entry for the wind tunnel. The D matrix is binary graph reach.

The support rows live in ``signature``. Ambiguity groups live in
``dmatrix_graph``. This module does not inject a wrong answer or store a
predicted value.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN


def check_dmatrix() -> None:
    from domains.wind_tunnel.signature import audit

    reports = audit()
    for item in reports:
        counts = item.summary["counts"]
        empty = sum(item.summary["det0"].values())
        print(
            f"stage 4 binary support: {item.sample} rows {len(item.results)} ("
            + " ".join(f"{key} {counts.get(key, 0)}" for key in (DIR, RSN, EXE, ALG))
            + f"), reachable {item.summary['det1']}, empty reach {empty}, "
            + f"DIR ambiguity groups {len(item.summary['e1_blocks'])} (largest {item.summary['e1_max_block']}).",
            flush=True,
        )


if __name__ == "__main__":
    check_dmatrix()
