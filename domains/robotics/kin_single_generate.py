# -*- coding: utf-8 -*-
"""Main-experiment kinematics set: one standard file and nine single factors.

Joint-sign files are recomputed by kin_variant_generate. Task-angle signs,
sweep, dihedral and the wing-root coordinate are rewritten from sample 01 and
checked against the unflipped pose or the published expressions. Samples that
flip two joints stay in the kinematics review directory and are not copied here.

Run::

    python -m domains.robotics.kin_single_generate
"""
from __future__ import annotations

import json
from pathlib import Path

import sympy as sp

from domain_layout import DATA
from domains.robotics.kin_variant_generate import (
    SOURCE,
    _angle_text,
    _apply_slots,
    _cascade,
    _pose,
    _replace_once,
    _rotations,
    _same,
    _task_angles,
    build_text,
)

OUT = DATA / "robotics" / "single"


def _mark(text: str, sample_id: str, title: str, factor: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# 数据"):
        raise RuntimeError("稿面缺少数据标题。")
    lines[0] = f"# 数据 {sample_id} {title}"
    lines.insert(1, "")
    lines.insert(2, f"单定义扰动：{factor}")
    return "\n".join(lines).rstrip() + "\n"


def _count(text: str, old: str, expected: int) -> str:
    found = text.count(old)
    if found != expected:
        raise RuntimeError(f"文句出现 {found} 次，应为 {expected}：{old}")
    return old


def _swap(text: str, old: str, new: str, expected: int) -> str:
    _count(text, old, expected)
    return text.replace(old, new)


def _task_pose():
    pose = _pose(*_rotations(_cascade(())))
    alpha, beta, gamma = _task_angles(pose)
    return pose, alpha, beta, gamma


def _rebuild(alpha, beta, gamma, kind: str) -> sp.Matrix:
    bv_sign = 1 if kind == "alpha" else -1
    c_sign = -1 if kind == "beta" else 1
    d_sign = 1 if kind == "gamma" else -1
    row_sign = 1 if kind == "gamma" else -1
    a = sp.cos(alpha) * sp.cos(beta)
    bv = bv_sign * sp.sin(alpha) * sp.cos(beta)
    c = c_sign * sp.sin(beta)
    d = bv * sp.cos(gamma) + d_sign * c * sp.sin(gamma)
    norm = sp.sqrt(a**2 + d**2)
    row1 = sp.Matrix([a, bv, c])
    row2 = sp.Matrix([
        -d / norm,
        a * sp.cos(gamma) / norm,
        row_sign * a * sp.sin(gamma) / norm,
    ])
    row3 = sp.simplify(row1.cross(row2))
    return sp.simplify(sp.Matrix.vstack(row1.T, row2.T, row3.T))


def _task_text(kind: str) -> str:
    pose, alpha, beta, gamma = _task_pose()
    if kind == "alpha":
        alpha = -alpha
    elif kind == "beta":
        beta = -beta
    elif kind == "gamma":
        gamma = -gamma
    else:
        raise RuntimeError(kind)
    rebuilt = _rebuild(alpha, beta, gamma, kind)
    if not _same(rebuilt, pose):
        raise RuntimeError(f"{kind} 的任务角不能还原姿态矩阵")
    text = SOURCE.read_text(encoding="utf-8")
    angles = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    if kind == "alpha":
        text = _swap(
            text,
            "工具俯仰任务角在任务方向指向工具下方时为正，并等于 atan2(-d_y, d_x)。",
            "工具俯仰任务角在任务方向指向工具上方时为正，并等于 atan2(d_y, d_x)。",
            1,
        )
        text = _swap(
            text,
            "任务方向指向工具下方时为正。α_T=atan2(-d_y, d_x)。",
            "任务方向指向工具上方时为正。α_T=atan2(d_y, d_x)。",
            2,
        )
        text = _swap(
            text,
            "工具俯仰任务角在任务方向指向工具下方时为正。",
            "工具俯仰任务角在任务方向指向工具上方时为正。",
            1,
        )
        text = _swap(text, "bv=-sin(α_T)*cos(β_T)", "bv=sin(α_T)*cos(β_T)", 1)
        updates = {"F54": "上方", "F59": "负", "F65": angles}
    elif kind == "beta":
        text = _swap(
            text,
            "工具侧向任务角在任务方向指向工具右侧时为正，并等于 atan2(d_z, sqrt(d_x^2 + d_y^2))。",
            "工具侧向任务角在任务方向指向工具左侧时为正，并等于 atan2(-d_z, sqrt(d_x^2 + d_y^2))。",
            1,
        )
        text = _swap(
            text,
            "任务方向指向工具右侧时为正。β_T=atan2(d_z, sqrt(d_x^2 + d_y^2))。",
            "任务方向指向工具左侧时为正。β_T=atan2(-d_z, sqrt(d_x^2 + d_y^2))。",
            2,
        )
        text = _swap(
            text,
            "工具侧向任务角在任务方向指向工具右侧时为正。",
            "工具侧向任务角在任务方向指向工具左侧时为正。",
            1,
        )
        text = _swap(text, "c=sin(β_T)", "c=-sin(β_T)", 1)
        updates = {"F55": "左侧", "F60": "负", "F65": angles}
    else:
        text = _swap(text, "γ_T=atan2(n·r, n·u)", "γ_T=atan2(-(n·r), n·u)", 2)
        text = _swap(
            text,
            "正方向与绕工具前方的右手定则相同，反方向为负。",
            "正方向与绕工具前方的右手定则相反，反方向为负。",
            3,
        )
        text = _swap(
            text,
            "正方向与绕工具前方轴的右手定则相同，反方向为负。",
            "正方向与绕工具前方轴的右手定则相反，反方向为负。",
            1,
        )
        text = _swap(
            text,
            "d=bv*cos(γ_T)-c*sin(γ_T)",
            "d=bv*cos(γ_T)+c*sin(γ_T)",
            1,
        )
        text = _replace_once(
            text,
            "本例工具滚转任务角写成 arcsin(1/2) 或 atan2(1, sqrt(3)) 都对。",
            f"本例工具滚转任务角写成 {_angle_text(gamma)} 或与之等价的反三角写法都对。",
        )
        updates = {
            "F56": "相同",
            "F64": "负",
            "F65": angles,
            "I8": "[-d/N, a*cos(γ_T)/N, a*sin(γ_T)/N]",
        }
    return _slots(text, updates)


def _slots(text: str, updates: dict[str, str]) -> str:
    forward, inverse = text.split("# 逆运动学样本", 1)
    inverse, tail = inverse.split("# 层间对应", 1)
    forward = _apply_slots(forward, "F", updates)
    inverse = _apply_slots(inverse, "I", updates)
    return forward + "# 逆运动学样本" + inverse + "# 层间对应" + tail


def _geometry(kind: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    if kind == "sweep":
        text = _swap(
            text,
            "工具右侧翼尖前缘位于工具翼根前缘后方时为正。前后偏移等于该侧向坐标乘 tan(Λ)，再取负号。",
            "工具右侧翼尖前缘位于工具翼根前缘前方时为正。前后偏移等于该侧向坐标乘 tan(Λ)。",
            1,
        )
        text = _swap(text, "前缘后掠使工具翼尖相对工具翼根靠后。", "前缘后掠使工具翼尖相对工具翼根靠前。", 1)
        text = _swap(text, "x_r - (b/2)*tan(Λ)", "x_r + (b/2)*tan(Λ)", 8)
        updates = {"F71": "前方", "F75": "(b/2)*tan(Λ)"}
    elif kind == "dihedral":
        text = _swap(
            text,
            "工具右侧翼尖高于工具翼根前缘时为正。",
            "工具右侧翼尖低于工具翼根前缘时为正。",
            1,
        )
        text = _swap(text, "上反使工具翼尖相对工具翼根升高。", "上反使工具翼尖相对工具翼根降低。", 1)
        text = _swap(text, ", (b/2)*tan(Γ),", ", -(b/2)*tan(Γ),", 6)
        updates = {"F83": "下方", "F84": "-(b/2)*tan(Γ)"}
    elif kind == "root":
        text = _swap(text, "从工具顶点量起，指向工具前方为正。", "从工具顶点量起，指向工具后方为正。", 1)
        text = _swap(text, "x_r - (b/2)*tan(Λ)", "-x_r - (b/2)*tan(Λ)", 8)
        text = _swap(text, "L + L_mount + -x_r", "L + L_mount - x_r", 4)
        updates = {}
    else:
        raise RuntimeError(kind)
    if updates:
        text = _slots(text, updates)
    return text


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError(f"缺少 {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    standard = _mark(SOURCE.read_text(encoding="utf-8"), "01", "球腕正装", "无")
    (OUT / "sample_01.md").write_text(standard, encoding="utf-8")
    written = [("01", "无", "球腕正装")]
    joints = (
        ("02", "q1 正方向相反", "关节转角 q1", ("q1",)),
        ("03", "q2 正方向相反", "关节转角 q2", ("q2",)),
        ("04", "q3 正方向相反", "关节转角 q3", ("q3",)),
    )
    for sample_id, title, factor, flips in joints:
        text = _mark(build_text(sample_id, title, flips), sample_id, title, factor)
        (OUT / f"sample_{sample_id}.md").write_text(text, encoding="utf-8")
        written.append((sample_id, factor, title))
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    tasks = (
        ("05", "工具俯仰任务角取反", "工具俯仰任务角", "alpha"),
        ("06", "工具侧向任务角取反", "工具侧向任务角", "beta"),
        ("07", "工具滚转任务角取反", "工具滚转任务角", "gamma"),
        ("08", "前缘后掠角取反", "前缘后掠角", "sweep"),
        ("09", "上反角取反", "上反角", "dihedral"),
        ("10", "翼根前缘坐标取反", "翼根前缘坐标", "root"),
    )
    for sample_id, title, factor, kind in tasks:
        body = _task_text(kind) if kind in {"alpha", "beta", "gamma"} else _geometry(kind)
        text = _mark(body, sample_id, title, factor)
        if f"单定义扰动：{factor}" not in text:
            raise RuntimeError(sample_id)
        (OUT / f"sample_{sample_id}.md").write_text(text, encoding="utf-8")
        written.append((sample_id, factor, title))
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    factors = [item[1] for item in written]
    if factors[0] != "无" or len(factors) != 10 or len(set(factors)) != 10:
        raise RuntimeError("机器人学主实验必须是 1 条标准定义和 9 条单定义扰动。")
    lines = [
        "# 机器人学主实验",
        "",
        "1 条球腕正装和 9 条单定义扰动。每条只改一项。正运动学和逆运动学仍分开注入。",
        "同时翻转两个或三个关节的稿留在 data/robotics/review/kinematics，不进入这组。",
        "",
    ]
    for sample_id, factor, title in written:
        lines.append(f"- [数据 {sample_id} {title}](sample_{sample_id}.md)：单定义扰动：{factor}。")
    lines.append("")
    (OUT / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "factors.json").write_text(
        json.dumps(
            [{"id": sample_id, "factor": factor, "title": title} for sample_id, factor, title in written],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
