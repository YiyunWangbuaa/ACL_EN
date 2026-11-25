# -*- coding: utf-8 -*-
"""Fixed question layers for fault diagnosis. One node, one level."""
from __future__ import annotations

from domains.wind_tunnel.contract_labels import REMOVED, TARGETS
from domains.wind_tunnel.error_graph import NODE_BY_ID

# Inclusive public ids. Removed expansion slots are not questions.
_RANGES = (
    (1, 1, "L0-CPATH"),
    (2, 7, "L0-CDEF"),
    (8, 8, "L1-MATRIX"),
    (9, 10, "L0-CPATH"),
    (11, 11, "L0-CDEF"),
    (12, 12, "L0-CPATH"),
    (13, 13, "L1-MATRIX"),
    (14, 15, "L0-CPATH"),
    (16, 18, "L0-CPATH"),
    (19, 19, "L0-CDEF"),
    (20, 22, "L1-SPACE"),
    (23, 23, "L0-CPATH"),
    (24, 25, "L0-CDEF"),
    (26, 26, "L1-MATRIX"),
    (27, 27, "L0-CDEF"),
    (28, 30, "L1-SPACE"),
    (31, 31, "L0-CPATH"),
    (32, 33, "L0-CDEF"),
    (34, 34, "L1-MATRIX"),
    (35, 35, "L0-CDEF"),
    (36, 36, "L1-SPACE"),
    (37, 37, "L0-CDEF"),
    (38, 39, "L1-SPACE"),
    (40, 40, "L0-CPATH"),
    (41, 42, "L0-CDEF"),
    (43, 43, "L1-MATRIX"),
    (44, 46, "L0-CPATH"),
    (47, 51, "L2a"),
    (55, 55, "L2b"),
    (56, 56, "L2c"),
    (57, 57, "L0-CPATH"),
    (58, 67, "L2c"),
    (68, 79, "L3-EXPRESSION-COMPONENT"),
    (80, 81, "L3-TARGET"),
    (82, 82, "L0-CPATH"),
    (83, 84, "L0-CDEF"),
    (85, 85, "L0-CPATH"),
    (86, 87, "L0-CDEF"),
    (88, 88, "L1-MATRIX"),
    (89, 94, "L0-CDEF"),
    (95, 95, "L1-MATRIX"),
    (96, 96, "L0-CPATH"),
    (97, 97, "L1-MATRIX"),
    (98, 100, "L2a"),
    (104, 104, "L2b"),
    (105, 110, "L0-CDEF"),
    (111, 116, "L3-TARGET"),
    (117, 117, "L0-CPATH"),
    (118, 125, "L0-CDEF"),
    (126, 126, "L1-MATRIX"),
    (127, 129, "L0-CPATH"),
    (130, 131, "L0-CDEF"),
    (132, 132, "L1-MATRIX"),
    (133, 134, "L0-CPATH"),
    (135, 135, "L2c"),
    (136, 141, "L0-CDEF"),
    (142, 148, "L2c"),
    (149, 163, "L0-CDEF"),
    (164, 173, "L3-EXPRESSION-COMPONENT"),
    (174, 178, "L3-EXPRESSION"),
    (179, 180, "L3-COMPENSATION-SIGN"),
    (181, 181, "L0-CDEF"),
    (182, 182, "L1-VECTOR"),
    (183, 183, "L0-CDEF"),
    (184, 184, "L1-VECTOR"),
    (185, 188, "L0-CDEF"),
    (189, 194, "L2c"),
    (195, 200, "L3-TARGET"),
    (201, 213, "L3-EXPRESSION-COMPONENT"),
)

JOINT_GROUPS = (
    ("L0", "姿态锚点", ("23", "31", "40")),
    ("L0", "变换约定", ("44", "45", "46")),
    ("L0", "力矩起终点", ("185", "186", "187", "188")),
    ("L2", "风轴到体轴阶段顺序", ("47", "48", "49", "50", "51")),
    ("L2", "天平到体轴阶段顺序", ("98", "99", "100")),
)


def _table() -> dict[str, str]:
    found: dict[str, str] = {}
    for start, end, level in _RANGES:
        for number in range(start, end + 1):
            blank_id = str(number)
            if blank_id in REMOVED:
                continue
            if blank_id in found:
                raise RuntimeError(f"ID{blank_id} 的层级重复。")
            found[blank_id] = level
    return found


LEVELS = _table()


def major(level: str) -> str:
    if level.startswith("L2"):
        return "L2"
    return level.split("-", 1)[0]


def level_of(node_id: str) -> str:
    return LEVELS[str(node_id).removeprefix("ID")]


def closure(node_id: str) -> tuple[str, ...]:
    """Asked-or-not ancestors that can reach this node. The node itself is excluded."""
    blank_id = str(node_id).removeprefix("ID")
    node = NODE_BY_ID.get(blank_id)
    if node is None:
        return ()
    return tuple(item for item in node.ancestors if item != blank_id)


def audit_layers() -> dict:
    missing = sorted(set(TARGETS) - set(LEVELS), key=int)
    extra = sorted(set(LEVELS) - set(TARGETS), key=int)
    conflict = [blank_id for blank_id, level in LEVELS.items() if major(level) not in {"L0", "L1", "L2", "L3"}]
    if missing or extra or conflict:
        raise RuntimeError(f"层级未闭合。缺少 {missing}，多出 {extra}，冲突 {conflict}。")
    counts = {name: 0 for name in ("L0", "L1", "L2", "L3")}
    for level in LEVELS.values():
        counts[major(level)] += 1
    if sum(counts.values()) != len(LEVELS):
        raise RuntimeError("四层之和不等于可分层节点数。")
    return {"nodes": len(LEVELS), **counts, "unassigned": 0, "conflict": 0}
