# -*- coding: utf-8 -*-
"""Generate cloze answers for moving body-frame moments to a reference point.

The questions follow Ref_point.md. Which body slot is axial, and therefore
which engineering coordinate flips, comes from BodyFrameModel. The lengths are
the measured scalars in the manuscript: model length L, sting-base gap L_t,
and balance-center-to-sting-base distance L_r.

The balance center is the body origin. The nose lies a distance Xr ahead of it.
Engineering [Xg, Yg, Zg] measures the axial slot rearward from the nose and
keeps the other two slots in the body positive directions. Force at the origin
then adds F cross r / L_RT to the moment at the reference point.

Each extra moment uses the template C·L_c/L_RT. C is replaced by the force
coefficient CA, CZ, or CN. L_c is replaced by the moment arm written from
Xg, Yg, Zg, and Xr, including the cross-product sign. The letters C and L_c
are not left in the answer.

Run::

    python -m domains.wind_tunnel.ref_point_generate
    python -m domains.wind_tunnel.ref_point_generate --spec declaration.json --out ref_point_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from body_frame_model import (
    AXIAL,
    BODY_AXES,
    DefinitionError,
    FRONT,
    LOAD_COEFFICIENTS,
    BodyFrameModel,
    _VECTOR,
    _dot,
)


ENGINEERING = {"XTZ": "Xg", "YTZ": "Yg", "ZTZ": "Zg"}
MOMENT_CODE = {"axial": "CA", "normal": "CN", "lateral": "CZ"}
FORCE_OF_ROLE = {item.role: item.symbol for item in LOAD_COEFFICIENTS if item.kind == "force"}
ORIGINS = ["天平校心", "机体质心", "风洞原点"]
SLOTS = ["Xg", "Yg", "Zg"]
PLACES = ["原点", "力矩参考点"]
SIGNS = ["正", "负"]
MOMENT_KEYS = ("MCN_CA", "MCZ_CA", "MCA_CZ", "MCN_CZ", "MCA_CN", "MCZ_CN")

XG_SYM, YG_SYM, ZG_SYM, L_RT_SYM = sp.symbols("Xg Yg Zg L_RT", real=True)
ARM = sp.symbols("arm")
ENGINEERING_SYMBOL = {"XTZ": XG_SYM, "YTZ": YG_SYM, "ZTZ": ZG_SYM}
ARM_TEXT = {"XTZ": "(Xr - Xg)", "YTZ": "(Xr - Yg)", "ZTZ": "(Xr - Zg)"}
FORCE_SYMBOL = {symbol: sp.symbols(symbol) for symbol in ("CN", "CZ", "CA")}

STANDARD_SPEC = {
    "placement": "upright",
    "body_frame": {
        "chirality": "right",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    },
}


def _text(expr, axial_name: str = "XTZ") -> str:
    rendered = str(sp.simplify(expr))
    return rendered.replace("arm", ARM_TEXT[axial_name])


def _role(absolute_axis: str) -> str:
    return absolute_axis.split("_", 1)[1]


def _embed(coords, model: BodyFrameModel):
    total = [sp.Integer(0), sp.Integer(0), sp.Integer(0)]
    for coord, name in zip(coords, BODY_AXES):
        vector = model.body_frame[name].vector
        for index in range(3):
            total[index] += coord * vector[index]
    return total


def _dot_expr(left, right) -> sp.Expr:
    return sp.simplify(sum(left[index] * right[index] for index in range(3)))


def generate(spec: dict) -> dict:
    """Build the cloze key for one declaration."""
    model = BodyFrameModel().declare(spec)
    if any(not model.body_frame[name].points_to for name in BODY_AXES):
        raise DefinitionError("body_frame must be complete.")
    axial_name = next(
        name for name in BODY_AXES if model.body_frame[name].absolute_axis == AXIAL
    )
    # Engineering axial is measured rearward from the nose. Flip it when the body slot points forward.
    forward_sign = _dot(model.body_frame[axial_name].vector, _VECTOR[FRONT])
    if forward_sign not in (1, -1):
        raise DefinitionError("The body axial axis must point forward or rearward.")
    math_coords = []
    for name in BODY_AXES:
        if name == axial_name:
            math_coords.append(sp.Integer(forward_sign) * ARM)
        else:
            math_coords.append(ENGINEERING_SYMBOL[name])

    contributions = {}
    for force_role, force_symbol in FORCE_OF_ROLE.items():
        force_coords = []
        for name in BODY_AXES:
            role = _role(model.body_frame[name].absolute_axis)
            force_coords.append(FORCE_SYMBOL[force_symbol] if role == force_role else 0)
        moment = _cross_expr(_embed(force_coords, model), _embed(math_coords, model))
        for name in BODY_AXES:
            moment_role = _role(model.body_frame[name].absolute_axis)
            key = f"M{force_symbol}_{MOMENT_CODE[moment_role]}"
            contributions[key] = sp.simplify(
                _dot_expr(moment, model.body_frame[name].vector) / L_RT_SYM
            )

    items: list[dict] = []

    def add(step: int, label: str, prompt: str, answer, options=None) -> None:
        items.append({
            "id": len(items) + 1,
            "manuscript_id": str(len(items) + 1),
            "step": step,
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer if isinstance(answer, str) else _text(answer, axial_name),
        })

    add(1, "C2", "体轴系原点所在位置", "天平校心", ORIGINS)
    add(2, "A1", "天平校心到模型顶点的距离 Xr", "L + L_t - L_r")
    add(3, "C1", "与体轴数学坐标正方向定义不同的工程分量", ENGINEERING[axial_name], SLOTS)
    add(3, "A1", "力矩参考点的体轴数学坐标", _vector_text(math_coords, axial_name))
    add(4, "C2", "力矩等效的起点", "原点", PLACES)
    add(4, "C2", "力矩等效的终点", "力矩参考点", PLACES)
    add(4, "C2", "产生额外力矩的力所在位置", "原点", PLACES)
    add(4, "C2", "额外力矩的作用位置", "力矩参考点", PLACES)
    rendered = {key: _template(contributions[key], axial_name) for key in MOMENT_KEYS}
    for key in MOMENT_KEYS:
        add(
            5,
            "A1",
            f"等效力矩系数 {key} 的符号。假设 Xg、Yg、Zg、Xr 都为正",
            "负" if rendered[key].startswith("-") else "正",
            SIGNS,
        )
    for key in MOMENT_KEYS:
        add(
            5,
            "A2",
            f"等效力矩系数 {key}。只允许使用模板 C·L_c/L_RT："
            "C 填入 CA、CZ、CN 中对应该力的系数，"
            "L_c 填入由 Xg、Yg、Zg、Xr 组成的力臂。"
            "假设 Xg、Yg、Zg、Xr 都为正。"
            "负号写在整个式子最外侧，力臂内部必须为正，表示额外力矩的绝对值。"
            "不要保留字母 C 或 L_c",
            rendered[key],
        )

    return {
        "task": "body_frame_moment_reference_point",
        "declaration": spec,
        "fixed": {
            "origin": "balance_center",
            "xr": "L + L_t - L_r",
            "moment_template": "C·L_c/L_RT",
            "C": "CA, CZ, or CN",
            "L_c": "moment arm written from Xg, Yg, Zg, and Xr",
        },
        "items": items,
    }


def _cross_expr(left, right):
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _positive_arm(arm_expr):
    """Move one overall minus outside. The remaining arm is the positive magnitude."""
    expanded = sp.expand(arm_expr)
    polynomial = sp.Poly(expanded, ARM, XG_SYM, YG_SYM, ZG_SYM)
    leading = polynomial.LC()
    if leading not in (1, -1):
        raise DefinitionError(f"Moment arm {arm_expr} does not have a unit leading coefficient.")
    if leading < 0:
        return "-", sp.expand(-expanded)
    return "", expanded


def _template(expr, axial_name: str) -> str:
    """Fill C·L_c/L_RT. The sign stands outside a positive arm."""
    numer, denom = sp.fraction(sp.together(sp.simplify(expr)))
    if sp.simplify(denom - L_RT_SYM) != 0:
        raise DefinitionError(f"Moment term {expr} is not divided by L_RT.")
    force_name = next(
        (name for name, symbol in FORCE_SYMBOL.items() if symbol in numer.free_symbols),
        None,
    )
    if force_name is None:
        raise DefinitionError(f"Moment term {expr} has no force coefficient CA, CZ, or CN.")
    outer, positive = _positive_arm(sp.simplify(numer / FORCE_SYMBOL[force_name]))
    arm = _text(positive, axial_name)
    if arm.startswith("(") and arm.endswith(")"):
        pass
    elif any(operator in arm for operator in ("+", "-")):
        arm = f"({arm})"
    return f"{outer}{force_name}·{arm}/L_RT"


def _vector_text(values, axial_name: str) -> str:
    cells = []
    for value in values:
        rendered = _text(value, axial_name)
        if rendered.startswith("(") and rendered.endswith(")"):
            rendered = rendered[1:-1]
        cells.append(rendered)
    return "[" + ", ".join(cells) + "]"


def _render(payload: dict) -> str:
    lines = ["# 力矩参考点评测答案", ""]
    step = None
    for item in payload["items"]:
        if item["step"] != step:
            step = item["step"]
            lines.append(f"## 步骤 {step}")
            lines.append("")
        options = ""
        if item["options"]:
            options = "（" + "/".join(item["options"]) + "）"
        lines.append(
            f"- ID {item['manuscript_id']} [{item['label']}] {item['prompt']}{options}：{item['answer']}"
        )
    lines.append("")
    return "\n".join(lines)


def _check_standard() -> None:
    payload = generate(STANDARD_SPEC)
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    assert len(payload["items"]) == 20
    assert by_id["1"] == "天平校心"
    assert by_id["2"] == "L + L_t - L_r"
    assert by_id["3"] == "Xg"
    assert by_id["4"] == "[Xr - Xg, Yg, Zg]"
    assert by_id["5"] == "原点"
    assert by_id["6"] == "力矩参考点"
    assert by_id["7"] == "原点"
    assert by_id["8"] == "力矩参考点"
    assert by_id["9"] == "正"
    assert by_id["10"] == "负"
    assert by_id["11"] == "正"
    assert by_id["12"] == "负"
    assert by_id["13"] == "负"
    assert by_id["14"] == "正"
    assert by_id["15"] == "CN·Zg/L_RT"
    assert by_id["16"] == "-CZ·Yg/L_RT"
    assert by_id["17"] == "CA·Yg/L_RT"
    assert by_id["18"] == "-CN·(Xr - Xg)/L_RT"
    assert by_id["19"] == "-CA·Zg/L_RT"
    assert by_id["20"] == "CZ·(Xr - Xg)/L_RT"
    for offset, key in enumerate(MOMENT_KEYS):
        sign = by_id[str(9 + offset)]
        formula = by_id[str(15 + offset)]
        if (sign == "负") != formula.startswith("-"):
            raise AssertionError(f"{key} 的符号 {sign} 与式子 {formula} 不一致。")

    rearward = json.loads(json.dumps(STANDARD_SPEC))
    rearward["body_frame"]["XTZ"] = {"along": "body_axial", "points": "body_rearward"}
    turned = {item["manuscript_id"]: item["answer"] for item in generate(rearward)["items"]}
    assert turned["3"] == "Xg"
    assert turned["4"] == "[-(Xr - Xg), Yg, Zg]"
    for offset, key in enumerate(MOMENT_KEYS):
        sign = turned[str(9 + offset)]
        formula = turned[str(15 + offset)]
        if sign not in SIGNS or (sign == "负") != formula.startswith("-"):
            raise AssertionError(f"轴向朝后时 {key} 的符号 {sign} 与式子 {formula} 不一致。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the moment-reference-point cloze answer key.")
    parser.add_argument("--spec", type=Path, help="JSON declaration. Defaults to the standard upright frame.")
    parser.add_argument("--out", type=Path, default=Path("ref_point_eval.json"))
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8")) if args.spec else STANDARD_SPEC
    if args.spec is None:
        _check_standard()
    payload = generate(spec)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = args.out.with_suffix(".md")
    markdown.write_text(_render(payload), encoding="utf-8")
    counts = {}
    for item in payload["items"]:
        counts[item["label"]] = counts.get(item["label"], 0) + 1
    print(f"Wrote {len(payload['items'])} blanks to {args.out}")
    print("Labels: " + ", ".join(f"{label} {counts[label]}" for label in ("C1", "C2", "A1", "A2")))


if __name__ == "__main__":
    main()
