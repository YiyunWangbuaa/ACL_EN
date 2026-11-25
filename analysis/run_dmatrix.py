# -*- coding: utf-8 -*-
"""图可达 D 矩阵。支持层只读依赖图，不注入错文、不重算。"""
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
    true = [group for group in groups if group["kind"] == "真模糊"]
    copied = [group for group in groups if group["kind"] == "类标签重复"]
    same = [group for group in groups if group["kind"] == "同注入同标签"]
    partial = [group for group in groups if group["kind"] == "同标签部分重叠"]
    _line(title)
    print(
        f"行 {len(rows)}，多成员组 {len(groups)}"
        f"（类标签重复 {len(copied)}，同注入同标签 {len(same)}，"
        f"同标签部分重叠 {len(partial)}，真模糊 {len(true)}）"
        f"，最大组 {max((len(group['members']) for group in groups), default=0)}"
    )
    print(
        f"DIR/RSN/EXE 共写的空 {len(overlap)}"
        + (f"：{pairs}" if pairs else "")
    )
    _print_groups("真模糊", true)
    _print_groups("同标签部分重叠", partial)


def _print_groups(title: str, groups) -> None:
    if not groups:
        print(f"{title}：0")
        return
    print(f"{title}：")
    for group in groups:
        print(f"  {len(group['members'])} 人 共有空 {','.join(group['shared']) or '无'}")
        for blanks, member in zip(group["blank_sets"], group["members"]):
            print(f"    {member}  空 {','.join(blanks)}")


def _wind(sample_id: str) -> None:
    view = render(sample_id)
    _line(sample_id)
    print(
        f"行 {len(view['rows'])}，测点列 {len(view['columns'])}，"
        f"多成员组 {view['n_ambiguity']}"
        f"（类标签重复 {view['n_label_repeat']}，同注入同标签 {view['n_same_label']}，"
        f"同标签部分重叠 {view['n_partial']}，真模糊 {view['n_true']}），最大组 {view['max_group']}"
    )
    overlap = view["multi_class_blanks"]
    pairs = {}
    for classes in overlap.values():
        key = "+".join(sorted(classes))
        pairs[key] = pairs.get(key, 0) + 1
    print(f"DIR/RSN/EXE 共写的空 {len(overlap)}" + (f"：{pairs}" if pairs else ""))
    _print_groups("真模糊", [group for group in view["groups"] if group["kind"] == "真模糊"])
    _print_groups("同标签部分重叠", [group for group in view["groups"] if group["kind"] == "同标签部分重叠"])


def main() -> None:
    from domains.aerodynamics.aligned import view as flight
    from domains.computer_vision.aligned import view as vision
    from domains.computer_vision.scene_aligned import view as scene
    from domains.mbd_kernel import instances
    from domains.robotics.kin_aligned import view as kin

    for title, build in (
        ("飞行力学", flight),
        ("机器人学运动学", kin),
    ):
        _report_rows(title, instances(build()))
    _line("机器视觉")
    for title, build in (
        ("基础理论环节", vision),
        ("物体与双目环节", scene),
    ):
        _report_rows(f"机器视觉 · {title}", instances(build()))
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
    print("\n图可达 D 矩阵完成（未注入）")


if __name__ == "__main__":
    main()
