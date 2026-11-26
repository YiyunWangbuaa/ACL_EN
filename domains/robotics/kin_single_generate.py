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
from domains.robotics.en_surface import english_text
from domains.robotics.kin_variant_generate import (
    SOURCE,
    _ROLL_DEFAULT,
    _angle_text,
    _apply_slots,
    _cascade,
    _pose,
    _replace_once,
    _roll_clause,
    _rotations,
    _same,
    _task_angles,
    build_text,
)

OUT = DATA / "robotics" / "single"


def _mark(text: str, sample_id: str, title: str, factor: str) -> str:
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# Data"):
        raise RuntimeError("The manuscript is missing a data title.")
    lines[0] = f"# Data {sample_id} {title}"
    lines.insert(1, "")
    lines.insert(2, f"Single-definition perturbation: {factor}")
    return "\n".join(lines).rstrip() + "\n"


def _count(text: str, old: str, expected: int) -> str:
    found = text.count(old)
    if found != expected:
        raise RuntimeError(f"phrase appears {found} times, expected {expected}: {old}")
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
        raise RuntimeError(f"{kind} task angles do not recover the attitude matrix")
    text = SOURCE.read_text(encoding="utf-8")
    angles = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    if kind == "alpha":
        text = _swap(
            text,
            "Tool pitch is positive when the task direction points downward, and it equals atan2(-d_y, d_x).",
            "Tool pitch is positive when the task direction points upward, and it equals atan2(d_y, d_x).",
            1,
        )
        text = _swap(
            text,
            "The angle is positive when the task direction points downward. α_T=atan2(-d_y, d_x).",
            "The angle is positive when the task direction points upward. α_T=atan2(d_y, d_x).",
            2,
        )
        text = _swap(
            text,
            "Tool pitch is positive when the task direction points downward.",
            "Tool pitch is positive when the task direction points upward.",
            1,
        )
        text = _swap(text, "bv=-sin(α_T)*cos(β_T)", "bv=sin(α_T)*cos(β_T)", 1)
        updates = {"F54": "upward", "F59": "negative", "F65": angles}
    elif kind == "beta":
        text = _swap(
            text,
            "The tool lateral angle is positive when the task direction points to the right, and it equals atan2(d_z, sqrt(d_x^2 + d_y^2)).",
            "The tool lateral angle is positive when the task direction points to the left, and it equals atan2(-d_z, sqrt(d_x^2 + d_y^2)).",
            1,
        )
        text = _swap(
            text,
            "The angle is positive when the task direction points to the right. β_T=atan2(d_z, sqrt(d_x^2 + d_y^2)).",
            "The angle is positive when the task direction points to the left. β_T=atan2(-d_z, sqrt(d_x^2 + d_y^2)).",
            2,
        )
        text = _swap(
            text,
            "The tool lateral angle is positive when the task direction points to the right.",
            "The tool lateral angle is positive when the task direction points to the left.",
            1,
        )
        text = _swap(text, "c=sin(β_T)", "c=-sin(β_T)", 1)
        updates = {"F55": "left", "F60": "negative", "F65": angles}
    else:
        text = _swap(text, "γ_T=atan2(n·r, n·u)", "γ_T=atan2(-(n·r), n·u)", 2)
        text = _swap(
            text,
            "The positive sense is the same as the right-hand rule about tool forward, and the opposite sense is negative.",
            "The positive sense is opposite the right-hand rule about tool forward, and the opposite sense is negative.",
            2,
        )
        text = _swap(
            text,
            "Its positive sense is the same as the right-hand rule about tool forward, and the opposite sense is negative.",
            "Its positive sense is opposite as the right-hand rule about tool forward, and the opposite sense is negative.",
            1,
        )
        text = _swap(
            text,
            "Its positive sense is the same as the right-hand rule about the tool forward axis, and the opposite sense is negative.",
            "Its positive sense is opposite the right-hand rule about the tool forward axis, and the opposite sense is negative.",
            1,
        )
        text = _swap(
            text,
            "d=bv*cos(γ_T)-c*sin(γ_T)",
            "d=bv*cos(γ_T)+c*sin(γ_T)",
            1,
        )
        clause = _roll_clause(_angle_text(gamma))
        if clause is not None:
            text = _replace_once(text, _ROLL_DEFAULT, clause)
        updates = {
            "F56": "same",
            "F64": "negative",
            "F65": angles,
            "I8": "[-d/N, a*cos(γ_T)/N, a*sin(γ_T)/N]",
        }
    return _slots(text, updates)


def _slots(text: str, updates: dict[str, str]) -> str:
    forward, inverse = text.split("# Inverse kinematics", 1)
    inverse, tail = inverse.split("# Cross-layer correspondence", 1)
    forward = _apply_slots(forward, "F", updates)
    inverse = _apply_slots(inverse, "I", updates)
    return forward + "# Inverse kinematics" + inverse + "# Cross-layer correspondence" + tail


def _geometry(kind: str) -> str:
    text = SOURCE.read_text(encoding="utf-8")
    if kind == "sweep":
        text = _swap(
            text,
            "The angle is positive when the right-tip leading edge lies behind the wing-root leading edge. The fore-aft offset equals that lateral coordinate times tan(Λ), and then changes sign.",
            "The angle is positive when the right-tip leading edge lies ahead of the wing-root leading edge. The fore-aft offset equals that lateral coordinate times tan(Λ).",
            1,
        )
        text = _swap(text, "Leading-edge sweep places the tip behind the root.", "Leading-edge sweep places the tip ahead of the root.", 1)
        text = _swap(text, "x_r - (b/2)*tan(Λ)", "x_r + (b/2)*tan(Λ)", 8)
        updates = {"F71": "forward", "F75": "(b/2)*tan(Λ)"}
    elif kind == "dihedral":
        text = _swap(
            text,
            "The angle is positive when the right tip is above the wing-root leading edge.",
            "The angle is positive when the right tip is below the wing-root leading edge.",
            1,
        )
        text = _swap(text, "Dihedral makes the tip higher than the root.", "Dihedral makes the tip lower than the root.", 1)
        text = _swap(text, ", (b/2)*tan(Γ),", ", -(b/2)*tan(Γ),", 6)
        updates = {"F83": "downward", "F84": "-(b/2)*tan(Γ)"}
    elif kind == "root":
        text = _swap(text, "It is positive forward.", "It is positive rearward.", 1)
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
        raise RuntimeError(f"missing {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    standard = _mark(SOURCE.read_text(encoding="utf-8"), "01", "upright wrist", "none")
    (OUT / "sample_01.md").write_text(english_text(standard), encoding="utf-8")
    written = [("01", "none", "upright wrist")]
    joints = (
        ("02", "q1 sense reversed", "joint angle q1", ("q1",)),
        ("03", "q2 sense reversed", "joint angle q2", ("q2",)),
        ("04", "q3 sense reversed", "joint angle q3", ("q3",)),
    )
    for sample_id, title, factor, flips in joints:
        text = _mark(build_text(sample_id, title, flips), sample_id, title, factor)
        (OUT / f"sample_{sample_id}.md").write_text(english_text(text), encoding="utf-8")
        written.append((sample_id, factor, title))
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    tasks = (
        ("05", "tool pitch reversed", "tool pitch", "alpha"),
        ("06", "tool lateral angle reversed", "tool lateral angle", "beta"),
        ("07", "tool roll reversed", "tool roll", "gamma"),
        ("08", "leading-edge sweep reversed", "leading-edge sweep", "sweep"),
        ("09", "dihedral reversed", "dihedral", "dihedral"),
        ("10", "wing-root coordinate reversed", "wing-root coordinate", "root"),
    )
    for sample_id, title, factor, kind in tasks:
        body = _task_text(kind) if kind in {"alpha", "beta", "gamma"} else _geometry(kind)
        text = _mark(body, sample_id, title, factor)
        if f"Single-definition perturbation: {factor}" not in text:
            raise RuntimeError(sample_id)
        (OUT / f"sample_{sample_id}.md").write_text(english_text(text), encoding="utf-8")
        written.append((sample_id, factor, title))
        print(f"wrote sample_{sample_id}.md {factor}", flush=True)
    factors = [item[1] for item in written]
    if factors[0] != "none" or len(factors) != 10 or len(set(factors)) != 10:
        raise RuntimeError(
            "The robotics main experiment must be 1 standard definition and 9 single-definition perturbations."
        )
    lines = [
        "# Robotics main experiment",
        "",
        "One upright-wrist standard and nine single-definition perturbations. Each item changes one definition. Forward and inverse kinematics are still injected separately.",
        "Samples that flip two or three joints stay in data/robotics/review/kinematics and are not part of this set.",
        "",
    ]
    for sample_id, factor, title in written:
        lines.append(
            f"- [Data {sample_id} {title}](sample_{sample_id}.md): "
            f"single-definition perturbation: {factor}."
        )
    lines.append("")
    (OUT / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
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
