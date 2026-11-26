# -*- coding: utf-8 -*-
"""Print the T_final table and check the repair-assignment shape. Does not score CR."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.error_graph import NODE_BY_ID  # noqa: E402
from domains.wind_tunnel.t_final import T_FINAL, audit_finals, audit_repair_shape  # noqa: E402


def main() -> None:
    audit_finals()
    print("T_final is the expressions with no downstream, 17 of them.")
    print()
    print("| process | node | depth | question |")
    print("| --- | --- | ---: | --- |")
    for process, nodes in T_FINAL.items():
        for node in nodes:
            item = NODE_BY_ID[node]
            print(f"| {process} | ID{node} | {item.dependency_depth} | {item.role} |")
    print()
    audit_repair_shape()
    print("Repair skeleton: a single repair of ID47 keeps the model text at ID48; a joint repair of ID47 and ID48 keeps the model text at ID2. ID55/59/60/80 on the path are left to the calculator.")


if __name__ == "__main__":
    main()
