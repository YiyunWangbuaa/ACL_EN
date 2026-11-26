# -*- coding: utf-8 -*-
"""Graph-reachability D matrix. The support layer reads the dependency graph only. It does not inject a wrong text and does not recompute."""
from __future__ import annotations

from domains.fault_class import DIR, EXE, RSN

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.dmatrix_graph import classify_groups, multi_class_blanks, render  # noqa: E402


def _line(title: str) -> None:
    print(f"\n{title}")


def _report_rows(title: str, rows) -> None:
    groups = classify_groups(rows)
    overlap = multi_class_blanks(rows)
    pairs = {}
    for classes in overlap.values():
        key = "+".join(sorted(classes))
        pairs[key] = pairs.get(key, 0) + 1
    true = [group for group in groups if group["kind"] == "true-ambiguity"]
    copied = [group for group in groups if group["kind"] == "label-repeat"]
    same = [group for group in groups if group["kind"] == "same-injection-same-label"]
    partial = [group for group in groups if group["kind"] == "same-label-partial-overlap"]
    _line(title)
    print(
        f"rows {len(rows)}, multi-member groups {len(groups)}"
        f" (label-repeat {len(copied)}, same-injection-same-label {len(same)}, "
        f"same-label-partial-overlap {len(partial)}, true-ambiguity {len(true)})"
        f", largest group {max((len(group['members']) for group in groups), default=0)}"
    )
    print(
        f"blanks written by both DIR/RSN/EXE {len(overlap)}"
        + (f": {pairs}" if pairs else "")
    )
    _print_groups("true-ambiguity", true)
    _print_groups("same-label-partial-overlap", partial)


def _print_groups(title: str, groups) -> None:
    if not groups:
        print(f"{title}: 0")
        return
    print(f"{title}:")
    for group in groups:
        print(f"  {len(group['members'])} members, shared blanks {','.join(group['shared']) or 'none'}")
        for blanks, member in zip(group["blank_sets"], group["members"]):
            print(f"    {member}  blanks {','.join(blanks)}")


def _wind(sample_id: str) -> None:
    view = render(sample_id)
    _line(sample_id)
    print(
        f"rows {len(view['rows'])}, probe columns {len(view['columns'])}, "
        f"multi-member groups {view['n_ambiguity']}"
        f" (label-repeat {view['n_label_repeat']}, same-injection-same-label {view['n_same_label']}, "
        f"same-label-partial-overlap {view['n_partial']}, true-ambiguity {view['n_true']}), largest group {view['max_group']}"
    )
    overlap = view["multi_class_blanks"]
    pairs = {}
    for classes in overlap.values():
        key = "+".join(sorted(classes))
        pairs[key] = pairs.get(key, 0) + 1
    print(f"blanks written by both DIR/RSN/EXE {len(overlap)}" + (f": {pairs}" if pairs else ""))
    _print_groups("true-ambiguity", [group for group in view["groups"] if group["kind"] == "true-ambiguity"])
    _print_groups("same-label-partial-overlap", [group for group in view["groups"] if group["kind"] == "same-label-partial-overlap"])


def main() -> None:
    from domains.aerodynamics.aligned import view as flight
    from domains.computer_vision.aligned import view as vision
    from domains.computer_vision.scene_aligned import view as scene
    from domains.mbd_kernel import instances
    from domains.robotics.kin_aligned import view as kin

    for title, build in (
        ("flight mechanics", flight),
        ("robot kinematics", kin),
    ):
        _report_rows(title, instances(build()))
    _line("computer vision")
    for title, build in (
        ("vision basics", vision),
        ("Object and stereo", scene),
    ):
        _report_rows(f"computer vision · {title}", instances(build()))
    for sample_id in (
        "standard-default",
        "standard-order",
        "alpha-rule",
        "roll-rule",
        "order-rule",
        "mount-rule",
        "body-rule",
    ):
        _wind(sample_id)
    print("\nGraph-reachability D matrix finished (no injection)")


if __name__ == "__main__":
    main()
