# -*- coding: utf-8 -*-
"""Fine labels for the wind-tunnel cloze.

Every formal blank stays a target. A probe is only an extra question that is
not written back. The legacy C1/C2/A1/A2 label remains the scoring label.
"""
from __future__ import annotations

import json
from pathlib import Path

REMOVED = frozenset({"52", "53", "54", "101", "102", "103"})

# Inclusive public-id ranges: label, subtype.
_RANGES = (
    (1, 1, "C-PATH-DIRECTION", "task_direction"),
    (2, 7, "C-AXIS", "absolute_axis_binding"),
    (8, 8, "A-FULL-EXEC", "alignment_matrix"),
    (9, 9, "C-PATH-EDGE", "source_state"),
    (10, 10, "C-PATH-EDGE", "target_state"),
    (11, 11, "C-DEF", "installation"),
    (12, 12, "C-PATH-BRANCH", "installation_branch"),
    (13, 13, "A-FULL-EXEC", "alignment_matrix"),
    (14, 14, "C-PATH-EDGE", "source_state"),
    (15, 15, "C-PATH-EDGE", "target_state"),
    (16, 18, "C-PATH-SEQUENCE", "angle_sequence"),
    (19, 19, "C-DEF", "installation"),
    (20, 22, "A-SPATIAL", "spatial_relation"),
    (23, 23, "C-PATH-ANCHOR", "process_anchor"),
    (24, 24, "C-AXIS", "rotation_axis_binding"),
    (25, 25, "C-SIGN", "matrix_sign"),
    (26, 26, "A-FULL-EXEC", "rotation_matrix_generation"),
    (27, 27, "C-DEF", "installation"),
    (28, 30, "A-SPATIAL", "spatial_relation"),
    (31, 31, "C-PATH-ANCHOR", "process_anchor"),
    (32, 32, "C-AXIS", "rotation_axis_binding"),
    (33, 33, "C-SIGN", "matrix_sign"),
    (34, 34, "A-FULL-EXEC", "rotation_matrix_generation"),
    (35, 35, "C-SIGN", "definition_sign"),
    (36, 36, "A-SPATIAL", "spatial_relation"),
    (37, 37, "C-SIGN", "theory_sign"),
    (38, 39, "A-SPATIAL", "spatial_relation"),
    (40, 40, "C-PATH-ANCHOR", "process_anchor"),
    (41, 41, "C-AXIS", "rotation_axis_binding"),
    (42, 42, "C-SIGN", "matrix_sign"),
    (43, 43, "A-FULL-EXEC", "rotation_matrix_generation"),
    (44, 46, "C-PATH-CONVENTION", "transform_convention"),
    (47, 51, "C-PATH-SEQUENCE", "stage_sequence"),
    (55, 55, "A-FULL-EXEC", "rotation_matrix_generation"),
    (56, 56, "C-PATH-STATE-TRANSITION", "vector_state"),
    (57, 57, "C-PATH-STATE", "intermediate_state"),
    (58, 58, "C-PATH-STATE", "vector_state"),
    (59, 61, "A-MATH", "component_expression"),
    (62, 67, "A-SPATIAL", "component_sign"),
    (68, 79, "A-MATH", "formula_part"),
    (80, 81, "A-FULL-EXEC", "formula_generation"),
    (82, 82, "C-PATH-DIRECTION", "task_direction"),
    (83, 84, "C-SIGN", "definition_sign"),
    (85, 85, "C-PATH-RELATION", "relative_roll"),
    (86, 86, "C-AXIS", "relative_rotation_axis"),
    (87, 87, "C-SIGN", "relative_rotation_turn"),
    (88, 88, "A-FULL-EXEC", "rotation_matrix_generation"),
    (89, 94, "C-AXIS", "absolute_axis_binding"),
    (95, 95, "A-FULL-EXEC", "alignment_matrix"),
    (96, 96, "C-PATH-BRANCH", "installation_branch"),
    (97, 97, "A-FULL-EXEC", "alignment_matrix"),
    (98, 100, "C-PATH-SEQUENCE", "task_specific"),
    (104, 104, "A-FULL-EXEC", "rotation_matrix_generation"),
    (105, 110, "C-ASSOC", "load_binding"),
    (111, 116, "A-FULL-EXEC", "formula_generation"),
    (117, 117, "C-PATH-DIRECTION", "task_direction"),
    (118, 125, "C-AXIS", "absolute_axis_binding"),
    (126, 126, "A-FULL-EXEC", "alignment_matrix"),
    (127, 127, "C-PATH-EDGE", "source_state"),
    (128, 128, "C-PATH-EDGE", "target_state"),
    (129, 129, "C-PATH-ANCHOR", "attitude_correction_anchor"),
    (130, 130, "C-AXIS", "rotation_axis_binding"),
    (131, 131, "C-SIGN", "matrix_sign"),
    (132, 132, "A-FULL-EXEC", "rotation_matrix_generation"),
    (133, 134, "C-PATH-SEQUENCE", "task_specific"),
    (135, 135, "A-MATH", "vector_expression"),
    (136, 137, "C-DEF", "elastic_definition"),
    (138, 138, "C-SIGN", "definition_sign"),
    (139, 141, "C-AXIS", "rotation_axis_binding"),
    (142, 144, "A-SPATIAL", "component_sign"),
    (145, 148, "A-MATH", "component_expression"),
    (149, 150, "C-AXIS", "component_binding"),
    (151, 151, "C-DEF", "installation"),
    (152, 153, "A-SPATIAL", "spatial_relation"),
    (154, 154, "C-SIGN", "definition_sign"),
    (155, 156, "A-SPATIAL", "spatial_relation"),
    (157, 157, "C-SIGN", "theory_sign"),
    (158, 163, "A-SPATIAL", "component_sign"),
    (164, 173, "A-MATH", "formula_part"),
    (174, 178, "A-FULL-EXEC", "formula_generation"),
    (179, 180, "A-SPATIAL", "spatial_relation"),
    (181, 181, "C-PATH-REFERENCE", "origin"),
    (182, 182, "A-MATH", "distance_expression"),
    (183, 183, "C-PATH-REPRESENTATION", "component_difference"),
    (184, 184, "A-LOCAL-EXEC", "coordinate_normalization"),
    (185, 188, "C-PATH-REFERENCE", "moment_reference"),
    (189, 194, "A-SPATIAL", "moment_sign"),
    (195, 200, "A-FULL-EXEC", "formula_generation"),
    (201, 206, "A-MATH", "formula_part"),
    (207, 212, "A-MATH", "formula_part"),
    (213, 213, "A-MATH", "formula_part"),
)

# Live probes only. Shallow context positions are not in the current instrument.
PROBES = (
    {
        "probe_id": "beta_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "target_id": "34",
        "context_depth": "target_local",
        "writes_back": False,
    },
    {
        "probe_id": "gamma_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "target_id": "43",
        "context_depth": "target_local",
        "writes_back": False,
    },
    {
        "probe_id": "roll_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "target_id": "88",
        "context_depth": "target_local",
        "writes_back": False,
    },
    {
        "probe_id": "mcn_cz_sign",
        "node_role": "probe",
        "new_label": "A-SPATIAL",
        "subtype": "moment_sign",
        "target_id": "198",
        "context_depth": "target_local",
        "writes_back": False,
    },
    {
        "probe_id": "mcn_cz_place",
        "node_role": "probe",
        "new_label": "A-MATH-FORMAT",
        "subtype": "sign_placement",
        "target_id": "198",
        "context_depth": "target_local",
        "writes_back": False,
    },
)


def _table() -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for start, end, label, subtype in _RANGES:
        for number in range(start, end + 1):
            blank_id = str(number)
            if blank_id in REMOVED:
                raise RuntimeError(f"已删除的 ID{blank_id} 被重新标了标签。")
            if blank_id in found:
                raise RuntimeError(f"ID{blank_id} 的新标签重复。")
            found[blank_id] = (label, subtype)
    expected = {str(number) for number in range(1, 214)} - REMOVED
    if set(found) != expected:
        missing = sorted(expected - set(found), key=int)
        extra = sorted(set(found) - expected, key=int)
        raise RuntimeError(f"新标签没有覆盖全部目标空。缺少 {missing}，多出 {extra}。")
    return found


TARGETS = _table()


def annotation_for(blank_id: str) -> dict:
    label, subtype = TARGETS[str(blank_id)]
    return {
        "node_role": "target",
        "new_label": label,
        "subtype": subtype,
    }


def export_document(path: Path) -> None:
    from domains.wind_tunnel.error_graph import GRAPH

    graph_ids = {
        node.blank_id
        for process_nodes in GRAPH.values()
        for node in process_nodes.values()
    }
    if graph_ids != set(TARGETS):
        raise RuntimeError("新标签和依赖图的题号不一致。")
    targets = []
    for process, nodes in GRAPH.items():
        for blank_id, node in sorted(nodes.items(), key=lambda item: int(item[0])):
            label, subtype = TARGETS[blank_id]
            targets.append({
                "node_id": f"ID{blank_id}",
                "role": "target",
                "legacy_label": node.label,
                "new_label": label,
                "subtype": subtype,
                "process": process,
                "parent_ids": [f"ID{parent}" for parent in node.depends_on],
                "graph_role": node.role,
            })
    probes = []
    for probe in PROBES:
        probes.append({
            "probe_id": probe["probe_id"],
            "role": "probe",
            "new_label": probe["new_label"],
            "subtype": probe["subtype"],
            "target_id": f"ID{probe['target_id']}",
            "context_depth": probe["context_depth"],
            "writes_back": False,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "targets": targets,
        "probes": probes,
        "note": "路径标签只标会改变后续推理路径的空。C-PATH-ANCHOR 只用于中间状态起点：ID23、ID31、ID40 是姿态旋转所在坐标系，ID129 是安装角校正的绝对方向。ID97 仍是矩阵执行，不标成路径边。",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_labels() -> None:
    from domains.wind_tunnel.contract_probes import PROBES as live

    export_document(Path(__file__).resolve().parents[2] / "data" / "wind_tunnel" / "contract_labels.json")
    live_ids = {item["probe_id"] for item in live}
    labeled = {item["probe_id"] for item in PROBES}
    if live_ids != labeled:
        raise RuntimeError(f"探针标注和当前探针不一致：{sorted(live_ids ^ labeled)}")
    for item in live:
        spec = next(probe for probe in PROBES if probe["probe_id"] == item["probe_id"])
        if item["new_label"] != spec["new_label"] or item["target"] != spec["target_id"]:
            raise RuntimeError(f"{item['probe_id']} 的标注和探针定义不一致。")
        if item.get("context_depth") != "target_local" or item.get("writes_back") is not False:
            raise RuntimeError(f"{item['probe_id']} 不是不写回的目标局部探针。")
    print(f"label check ok: {len(TARGETS)} targets, {len(PROBES)} probes")


if __name__ == "__main__":
    check_labels()
