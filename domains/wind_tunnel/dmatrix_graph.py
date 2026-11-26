# -*- coding: utf-8 -*-
"""Audit D matrix from graph reachability (the support layer). It does not inject faults.

D[f, m] = 1 exactly when m is in f.potential (the injection point plus the reachable probe columns of the graph descendants).
An ambiguity group is the faults whose potential is identical. Groups fall into two kinds:

- label repeat: the members write the same set of blanks and differ only in the DIR/RSN/EXE label.
- true ambiguity: the members write different blanks, but the descendant sets are the same.
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
    """Fault id to the reachable probe columns (graph reachability, no injection)."""
    return {item.id: item.potential for item in instances(sample_id)}


def ambiguity_groups(sample_id: str) -> tuple[tuple[str, ...], ...]:
    """Multi-member ambiguity groups merged by potential, with faults sorted by id."""
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
            kind = "label repeat"
        elif not shared:
            kind = "true ambiguity"
        elif same:
            kind = "same injection and label"
        else:
            kind = "partial overlap of one label"
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
    """Audit view: rows are faults, columns are node-level probes, plus the two kinds of ambiguity groups."""
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
        "n_label_repeat": sum(group["kind"] == "label repeat" for group in groups),
        "n_same_label": sum(group["kind"] == "same injection and label" for group in groups),
        "n_partial": sum(group["kind"] == "partial overlap of one label" for group in groups),
        "n_true": sum(group["kind"] == "true ambiguity" for group in groups),
        "multi_class_blanks": overlap,
    }


if __name__ == "__main__":
    import sys

    sample = sys.argv[1] if len(sys.argv) > 1 else "body-rule"
    view = render(sample)
    print(f"=== {sample}: graph-reachability D-matrix audit ===")
    print(f"fault rows {len(view['rows'])}, probe columns (nodes) {len(view['columns'])}")
    print(
        f"multi-member ambiguity groups {view['n_ambiguity']}, largest group {view['max_group']}; "
        f"label repeat {view['n_label_repeat']}, same injection and label {view['n_same_label']}, "
        f"partial overlap of one label {view['n_partial']}, true ambiguity {view['n_true']}"
    )
    print(f"blanks written by more than one of DIR/RSN/EXE: {len(view['multi_class_blanks'])}")
    for group in view["groups"]:
        blanks = ",".join(group["blanks"] or group["shared"]) or "no shared blanks"
        print(f"  [{group['kind']}] {len(group['members'])} members blanks {blanks}: {', '.join(group['members'])}")
