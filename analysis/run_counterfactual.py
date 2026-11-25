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
    print("T_final 是没有下游的表达式，共 17 个。")
    print()
    print("| 过程 | 节点 | 深度 | 题目 |")
    print("| --- | --- | ---: | --- |")
    for process, nodes in T_FINAL.items():
        for node in nodes:
            item = NODE_BY_ID[node]
            print(f"| {process} | ID{node} | {item.dependency_depth} | {item.role} |")
    print()
    audit_repair_shape()
    print("修复骨架：单修 ID47 时 ID48 保持模型原文；联修 ID47 与 ID48 时 ID2 保持模型原文。路径上的 ID55/59/60/80 都留给计算器。")


if __name__ == "__main__":
    main()
