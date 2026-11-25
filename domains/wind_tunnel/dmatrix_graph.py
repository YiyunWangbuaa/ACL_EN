# -*- coding: utf-8 -*-
"""图可达生成的审计 D 矩阵（支持层），不做故障注入。

D[f, m] = 1 当且仅当 m ∈ f.potential（注入点 + 图后代的可达测点列）。
模糊组 = potential 完全相同的故障。组再分成两类：

- 类标签重复：组员写入的空集合相同，只是 DIR/RSN/EXE 标签不同。
- 真模糊：组员写入的空不同，后代集却相同。
"""
from __future__ import annotations

from domains.fault_class import DIR, EXE, RSN

from collections import defaultdict

from domains.wind_tunnel.fault_instance import instances
from domains.wind_tunnel.measurement import catalog, node_id_of

_LABELS = frozenset({DIR, RSN, EXE})


def _blank_key(blank: str) -> tuple:
    digits = "".join(ch for ch in str(blank) if ch.isdigit())
    prefix = "" if str(blank)[:1].isdigit() else str(blank)[:1]
    return (prefix, int(digits) if digits else 0, str(blank))


def injected_blanks(item) -> frozenset[str]:
    """The blanks this row writes. The written text is not part of the support."""
    return frozenset(blank for blank, _value in item.injector)


def support(sample_id: str) -> dict[str, frozenset[str]]:
    """故障 id -> 可达测点列（图可达，无注入）。"""
    return {item.id: item.potential for item in instances(sample_id)}


def ambiguity_groups(sample_id: str) -> tuple[tuple[str, ...], ...]:
    """按 potential 归并的多成员模糊组，组内故障按 id 排序。"""
    groups: dict[frozenset[str], list[str]] = defaultdict(list)
    for fault_id, reach in support(sample_id).items():
        groups[reach].append(fault_id)
    return tuple(sorted(
        tuple(sorted(members))
        for members in groups.values()
        if len(members) > 1
    ))


def multi_class_blanks(rows) -> dict[str, dict[str, tuple[str, ...]]]:
    """Blanks written by more than one of DIR, RSN, EXE."""
    found: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for item in rows:
        if item.klass not in _LABELS:
            continue
        for blank in injected_blanks(item):
            found[blank][item.klass].append(item.id)
    return {
        blank: {klass: tuple(ids) for klass, ids in classes.items()}
        for blank, classes in sorted(found.items(), key=lambda pair: _blank_key(pair[0]))
        if len(classes) > 1
    }


def classify_groups(rows) -> tuple[dict, ...]:
    """Split same-potential groups into label copies and real ambiguity."""
    grouped: dict[frozenset[str], list] = defaultdict(list)
    for item in rows:
        grouped[item.potential].append(item)
    found = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        members = sorted(members, key=lambda item: item.id)
        sets = [injected_blanks(item) for item in members]
        classes = tuple(sorted({item.klass for item in members}))
        shared = sets[0].intersection(*sets[1:]) if sets else frozenset()
        same = len(set(sets)) == 1
        if shared and len(classes) > 1:
            kind = "类标签重复"
        elif not shared:
            kind = "真模糊"
        elif same:
            kind = "同注入同标签"
        else:
            kind = "同标签部分重叠"
        found.append({
            "kind": kind,
            "members": tuple(item.id for item in members),
            "classes": classes,
            "blanks": tuple(sorted(sets[0], key=_blank_key)) if same else (),
            "shared": tuple(sorted(shared, key=_blank_key)),
            "blank_sets": tuple(tuple(sorted(item, key=_blank_key)) for item in sets),
        })
    found.sort(key=lambda group: (-len(group["members"]), group["kind"], group["members"]))
    return tuple(found)


def render(sample_id: str) -> dict:
    """审计视图：行=故障，列=测点（节点级），以及两类模糊组。"""
    library = instances(sample_id)
    columns = [m.id for m in catalog() if m.kind == "node"]
    rows = []
    for item in sorted(library, key=lambda row: row.id):
        nodes = sorted({node_id_of(col) for col in item.potential if node_id_of(col)}, key=_blank_key)
        rows.append({
            "fault": item.id,
            "klass": item.klass,
            "blanks": list(sorted(injected_blanks(item), key=_blank_key)),
            "reach_nodes": nodes,
            "reach_count": len(nodes),
        })
    groups = classify_groups(library)
    overlap = multi_class_blanks(library)
    return {
        "sample": sample_id,
        "columns": columns,
        "rows": rows,
        "ambiguity_groups": tuple(group["members"] for group in groups),
        "groups": groups,
        "n_ambiguity": len(groups),
        "max_group": max((len(group["members"]) for group in groups), default=0),
        "n_label_repeat": sum(group["kind"] == "类标签重复" for group in groups),
        "n_same_label": sum(group["kind"] == "同注入同标签" for group in groups),
        "n_partial": sum(group["kind"] == "同标签部分重叠" for group in groups),
        "n_true": sum(group["kind"] == "真模糊" for group in groups),
        "multi_class_blanks": overlap,
    }


if __name__ == "__main__":
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "body-rule"
    view = render(sample)
    print(f"=== {sample}：图可达 D 矩阵审计 ===")
    print(f"故障行 {len(view['rows'])}，测点列（节点） {len(view['columns'])}")
    print(
        f"多成员模糊组 {view['n_ambiguity']} 个，最大组 {view['max_group']} 人；"
        f"类标签重复 {view['n_label_repeat']}，同注入同标签 {view['n_same_label']}，"
        f"同标签部分重叠 {view['n_partial']}，真模糊 {view['n_true']}"
    )
    print(f"DIR/RSN/EXE 共写的空 {len(view['multi_class_blanks'])} 个")
    for group in view["groups"]:
        blanks = ",".join(group["blanks"] or group["shared"]) or "无共有空"
        print(f"  [{group['kind']}] {len(group['members'])} 人 空 {blanks}: {', '.join(group['members'])}")
