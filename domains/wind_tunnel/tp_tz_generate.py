# -*- coding: utf-8 -*-
"""Generate cloze answers for the balance-frame to body-frame load evaluation.

The questions follow doc/TP_TZ.md. Geometry, signs, and matrices come from
BodyFrameModel under the given declaration. Passive rotation, intrinsic
moving-axis rotation, and the force/moment axis associations are fixed.

The installation roll γ_M−γ_tp is applied first, while the balance axes are
still the originally defined ones, so the roll matrix belongs to the balance
axial coordinate. RTP_TZ1 then matches those coordinates with each frame's
absolute directions, and RTP_TZ2 applies upright or inverted mounting. The
passive product is

    [XTZ, YTZ, ZTZ]^T = R1*R2*R3*[XTP, YTP, ZYP]^T

with R3 the balance roll, R2 = RTP_TZ1, and R1 = RTP_TZ2.

Run::

    python -m domains.wind_tunnel.tp_tz_generate
    python -m domains.wind_tunnel.tp_tz_generate --spec declaration.json --out tp_tz_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sympy as sp

from body_frame_model import (
    AXIAL,
    BALANCE_AXIAL,
    BALANCE_VOCAB,
    BODY_AXES,
    DefinitionError,
    GAMMA_M,
    INTRINSIC,
    LOAD_COEFFICIENTS,
    NEGATIVE_ROTATION,
    PLACE_INVERTED,
    PLACE_UPRIGHT,
    POSITIVE_ROTATION,
    ROTATION_ANGLE,
    SENSE_OPPOSITE,
    SENSE_SAME,
    BodyFrameModel,
    roll_about_nose_sense,
    _VECTOR,
    _body_direction_in_tunnel,
    _dot,
)


FRAMES = ["body-axis frame", "balance frame", "wind frame"]
CONVERSION_PAIRS = [f"{source} to {target}" for source in FRAMES for target in FRAMES]
AXES = ["XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP"]
DIRECTED = [
    "XFD", "YFD", "ZFD", "XTZ", "YTZ", "ZTZ", "XTP", "YTP", "ZYP",
    "-XFD", "-YFD", "-ZFD", "-XTZ", "-YTZ", "-ZTZ", "-XTP", "-YTP", "-ZYP",
]
ROLE_VECTOR = {"normal": _VECTOR["body_up"], "axial": _VECTOR["body_forward"], "lateral": _VECTOR["body_right"]}
MOUNTS = ["upright", "inverted"]
HAND_SIGNS = ["positive", "negative"]
TURN_SIGNS = ["positive", "reverse"]
BODY_PAIRS = ["CN_TZ,MN_TZ", "CA_TZ,MA_TZ", "CZ_TZ,MZ_TZ"]
BALANCE_PAIRS = ["CN_TP,MN_TP", "CA_TP,MA_TP", "CZ_TP,MZ_TP"]

GAMMA_M_SYM, GAMMA_TP_SYM = sp.symbols("gamma_M gamma_tp", real=True)
DISPLAY = {"gamma_M": "γ_M", "gamma_tp": "γ_tp"}
FORCE_OF_ROLE = {item.role: item.symbol for item in LOAD_COEFFICIENTS if item.kind == "force"}
MOMENT_OF_ROLE = {item.role: item.symbol for item in LOAD_COEFFICIENTS if item.kind == "moment"}

STANDARD_SPEC = {
    "alpha": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma": {"right_wing_toward": "body_down", "sign": "positive"},
    "alpha_m": {"airspeed_toward": "body_down", "sign": "positive"},
    "beta_m": {"airspeed_toward": "body_right", "sign": "positive"},
    "gamma_m": {"right_wing_toward": "body_down", "sign": "positive"},
    "placement": "upright",
    "body_frame": {
        "chirality": "right",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    },
    "wind_frame": {
        "chirality": "right",
        "XFD": {"along": "tunnel_axial", "points": "tunnel_forward"},
        "YFD": {"along": "tunnel_normal", "points": "tunnel_up"},
    },
    "gamma_tp": {"sense": "same"},
    "balance_frame": {
        "chirality": "right",
        "XTP": {"along": "balance_axial", "points": "balance_forward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    },
}


def _text(expr) -> str:
    rendered = str(sp.simplify(expr))
    for source, target in DISPLAY.items():
        rendered = rendered.replace(source, target)
    return rendered


def _matrix_text(matrix: sp.Matrix) -> str:
    rows = []
    for row in range(matrix.rows):
        cells = ", ".join(_text(matrix[row, column]) for column in range(matrix.cols))
        rows.append(f"[{cells}]")
    return "[" + ", ".join(rows) + "]"


def _directed(frame: dict, target) -> str:
    for name, axis in frame.items():
        alignment = _dot(axis.vector, target)
        if alignment == 1:
            return name
        if alignment == -1:
            return f"-{name}"
    raise DefinitionError("No coordinate axis lies on the requested absolute direction.")


def _role(absolute_axis: str) -> str:
    return absolute_axis.split("_", 1)[1]


def _nose_hand_sign(right_wing_toward: str) -> str:
    """Sign of a roll about the body axial positive direction, not the coordinate axis."""
    sense = roll_about_nose_sense(right_wing_toward)
    if sense == SENSE_SAME:
        return "positive"
    if sense == SENSE_OPPOSITE:
        return "negative"
    raise DefinitionError("Body roll has no right-hand sense about the axial positive direction.")


def _hand_sign(rotation: str) -> str:
    if rotation == POSITIVE_ROTATION:
        return "positive"
    if rotation == NEGATIVE_ROTATION:
        return "negative"
    raise DefinitionError("A roll or installation angle has no right-hand sense.")


def _sense_number(rotation: str) -> int:
    return 1 if _hand_sign(rotation) == "positive" else -1


def _alignment(model: BodyFrameModel) -> sp.Matrix:
    """Balance-to-body matrix at the upright embedding, before mounting."""
    return sp.Matrix([
        [
            _dot(_VECTOR[model.body_frame[body_name].points_to], model.balance_frame[balance_name].vector)
            for balance_name in BALANCE_VOCAB.coordinate_axes
        ]
        for body_name in BODY_AXES
    ])


def _placed(model: BodyFrameModel) -> sp.Matrix:
    return sp.Matrix([
        [
            _dot(
                _body_direction_in_tunnel(model.body_frame[body_name].points_to, model.placement),
                model.balance_frame[balance_name].vector,
            )
            for balance_name in BALANCE_VOCAB.coordinate_axes
        ]
        for body_name in BODY_AXES
    ])


def _mount(model: BodyFrameModel, alignment: sp.Matrix) -> sp.Matrix:
    placed = _placed(model)
    if alignment.det() == 0:
        raise DefinitionError("RTP_TZ1 is singular, so the mounting matrix cannot be split off.")
    mount = sp.simplify(placed * alignment.inv())
    if not (mount * alignment).equals(placed):
        raise DefinitionError("RTP_TZ2*RTP_TZ1 does not recover the placed balance-to-body matrix.")
    return mount


def _pair(role: str, suffix: str) -> str:
    return f"{FORCE_OF_ROLE[role]}_{suffix},{MOMENT_OF_ROLE[role]}_{suffix}"


def generate(spec: dict) -> dict:
    """Build the cloze key for one declaration."""
    model = BodyFrameModel().declare(spec)
    if model.aircraft.rotation_type != INTRINSIC:
        raise DefinitionError("Aircraft rotation must stay intrinsic.")
    if model.placement not in (PLACE_UPRIGHT, PLACE_INVERTED):
        raise DefinitionError("placement must be upright or inverted.")
    if any(not model.body_frame[name].points_to for name in BODY_AXES):
        raise DefinitionError("body_frame is incomplete.")
    if any(not model.balance_frame[name].points_to for name in BALANCE_VOCAB.coordinate_axes):
        raise DefinitionError("balance_frame is incomplete.")
    roll = model.mechanism_angles[GAMMA_M]
    install = model.installation
    if not roll.associated_axis or roll.rotation_when_positive not in (POSITIVE_ROTATION, NEGATIVE_ROTATION):
        raise DefinitionError("gamma_m must declare the positive right-wing direction.")
    if not install.associated_axis or install.rotation_when_positive not in (POSITIVE_ROTATION, NEGATIVE_ROTATION):
        raise DefinitionError("gamma_tp must declare its sense.")

    alignment = _alignment(model)
    mount = _mount(model, alignment)
    body_axial = model.body_frame[roll.associated_axis]
    balance_axial = model.balance_frame[install.associated_axis]
    if body_axial.absolute_axis != AXIAL or balance_axial.absolute_axis != BALANCE_AXIAL:
        raise DefinitionError("The roll and the installation must both lie on the axial axis.")
    same_direction = _dot(body_axial.vector, balance_axial.vector)
    if same_direction not in (1, -1):
        raise DefinitionError("The body and balance axial axes must be parallel.")

    body_sign = _sense_number(roll.rotation_when_positive)
    # The declared sense is about absolute forward. The passive matrix turns about
    # the coordinate axis, so a rearward axis reverses the installation angle.
    balance_sign = _sense_number(balance_axial.installation_rotation_when_positive)
    # Express the same physical roll in the original balance coordinate.
    # same_direction converts the body-coordinate sense onto the balance axial slot.
    theta = sp.simplify(
        same_direction * body_sign * GAMMA_M_SYM - balance_sign * GAMMA_TP_SYM
    )
    gamma_coeff = sp.simplify(theta.coeff(GAMMA_M_SYM))
    if gamma_coeff == 1:
        turn = "positive"
    elif gamma_coeff == -1:
        turn = "reverse"
    else:
        raise DefinitionError(f"Roll angle must be linear in gamma_M, got {theta}.")
    roll_matrix = sp.simplify(balance_axial.rotation_matrix.subs(ROTATION_ANGLE, theta))
    total = sp.simplify(mount * alignment * roll_matrix)

    balance_force = []
    balance_moment = []
    for name in BALANCE_VOCAB.coordinate_axes:
        role = _role(model.balance_frame[name].absolute_axis)
        balance_force.append(sp.symbols(f"{FORCE_OF_ROLE[role]}_TP"))
        balance_moment.append(sp.symbols(f"{MOMENT_OF_ROLE[role]}_TP"))
    body_force = sp.simplify(total * sp.Matrix(balance_force))
    body_moment = sp.simplify(total * sp.Matrix(balance_moment))
    expressed = {}
    for index, name in enumerate(BODY_AXES):
        role = _role(model.body_frame[name].absolute_axis)
        expressed[FORCE_OF_ROLE[role]] = body_force[index]
        expressed[MOMENT_OF_ROLE[role]] = body_moment[index]

    items: list[dict] = []

    def add(manuscript_id: str, step: int, label: str, prompt: str, answer, options=None) -> None:
        number = int(manuscript_id)
        if number >= 2:
            number -= 1
        items.append({
            "id": len(items) + 1,
            "manuscript_id": str(number),
            "step": step,
            "label": label,
            "prompt": prompt,
            "options": options,
            "answer": answer if isinstance(answer, str) else _text(answer),
        })

    add("1", 1, "C2", "Forces and moments measured in the balance frame must be converted to the body-axis frame, so the transform is", "balance frame to body-axis frame", CONVERSION_PAIRS)

    add(
        "3", 2, "C1",
        "The balance installation angle is positive when it is this relation to the right-hand rule about the positive absolute balance axis (forward)",
        _hand_sign(install.rotation_when_positive),
        HAND_SIGNS,
    )
    add(
        "4", 2, "C1",
        "The airframe roll is positive when it is this relation to the right-hand rule about the positive absolute airframe axis (forward)",
        _nose_hand_sign(roll.right_wing_toward),
        HAND_SIGNS,
    )
    add("5", 2, "A1", "With the balance axis, the airframe axis, and both positive directions fixed, the roll between the balance frame and the airframe", _text(theta))
    add("6", 2, "A1", "That turn is about the coordinate axis that corresponds to the absolute balance axis", balance_axial.name, AXES)
    add("7", 2, "A1", "The sense about that axis", turn, TURN_SIGNS)
    add("8", 2, "A2", "the rotation matrix is R_(γ_M-γ_tp)", _matrix_text(roll_matrix))

    for manuscript_id, role_name in (("9", "normal"), ("10", "axial"), ("11", "lateral")):
        add(
            manuscript_id, 3, "C1", f"the coordinate axis for the positive absolute {role_name} of the balance frame",
            _directed(model.balance_frame, ROLE_VECTOR[role_name]), DIRECTED,
        )
    for manuscript_id, role_name in (("12", "normal"), ("13", "axial"), ("14", "lateral")):
        add(
            manuscript_id, 3, "C1", f"the coordinate axis for the positive absolute {role_name} of the airframe",
            _directed(model.body_frame, ROLE_VECTOR[role_name]), DIRECTED,
        )
    add("15", 3, "A2", "The pre-matrix from the balance frame to the body-axis frame is RTP_TZ1", _matrix_text(alignment))
    add("16", 4, "C1", "Under this declaration the model is", "upright" if model.placement == PLACE_UPRIGHT else "inverted", MOUNTS)
    add("17", 4, "A2", "The pre-matrix under this declaration is RTP_TZ2", _matrix_text(mount))

    factor_options = ["R_(γ_M-γ_tp)", "RTP_TZ1", "RTP_TZ2"]
    add("18", 5, "A2", "R1 in the product", "RTP_TZ2", factor_options)
    add("19", 5, "A2", "R2 in the product", "RTP_TZ1", factor_options)
    add("20", 5, "A2", "R3 in the product", "R_(γ_M-γ_tp)", factor_options)
    add("24", 6, "A2", "the symbolically simplified product R = R1*R2*R3", _matrix_text(total))

    for manuscript_id, axis_name in zip(("25", "26", "27"), BODY_AXES):
        role = _role(model.body_frame[axis_name].absolute_axis)
        add(manuscript_id, 7, "C1", f"the body-axis force and moment associated with {axis_name}", _pair(role, "TZ"), BODY_PAIRS)
    for manuscript_id, axis_name in zip(("28", "29", "30"), BALANCE_VOCAB.coordinate_axes):
        role = _role(model.balance_frame[axis_name].absolute_axis)
        add(manuscript_id, 7, "C1", f"the balance force and moment associated with {axis_name}", _pair(role, "TP"), BALANCE_PAIRS)

    for manuscript_id, symbol in zip(("31", "32", "33", "34", "35", "36"), ("CN", "CA", "CZ", "MN", "MA", "MZ")):
        add(manuscript_id, 8, "A2", f"the body-axis component {symbol}_TZ", _text(expressed[symbol]))

    return {
        "task": "balance_frame_to_body_frame",
        "declaration": spec,
        "fixed": {
            "transform": "passive",
            "rotation_type": INTRINSIC,
            "product": "[XTZ, YTZ, ZTZ]^T = R1*R2*R3*[XTP, YTP, ZYP]^T",
            "loads": {item.symbol: item.role for item in LOAD_COEFFICIENTS},
        },
        "items": items,
    }


def _render(payload: dict) -> str:
    lines = ["# balance frame to body-axis frame answers", ""]
    step = None
    for item in payload["items"]:
        if item["step"] != step:
            step = item["step"]
            lines.append(f"## Step {step}")
            lines.append("")
        options = ""
        if item["options"]:
            options = "（" + "/".join(item["options"]) + "）"
        lines.append(
            f"- ID {item['manuscript_id']} [{item['label']}] {item['prompt']}{options}: {item['answer']}"
        )
    lines.append("")
    return "\n".join(lines)


def _check_standard() -> None:
    payload = generate(STANDARD_SPEC)
    by_id = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    identity = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert by_id["1"] == "balance frame to body-axis frame"
    assert by_id["2"] == "positive"
    assert by_id["3"] == "positive"
    assert by_id["4"] == "γ_M - γ_tp"
    assert by_id["5"] == "XTP"
    assert by_id["6"] == "positive"
    assert by_id["8"] == "YTP"
    assert by_id["9"] == "XTP"
    assert by_id["10"] == "ZYP"
    assert by_id["11"] == "YTZ"
    assert by_id["12"] == "XTZ"
    assert by_id["13"] == "ZTZ"
    assert by_id["14"] == identity
    assert by_id["15"] == "upright"
    assert by_id["16"] == identity
    assert [by_id["17"], by_id["18"], by_id["19"]] == ["RTP_TZ2", "RTP_TZ1", "R_(γ_M-γ_tp)"]
    assert "20" not in by_id and "21" not in by_id and "22" not in by_id
    assert by_id["23"] == by_id["7"]
    assert by_id["24"] == "CA_TZ,MA_TZ"
    assert by_id["25"] == "CN_TZ,MN_TZ"
    assert by_id["26"] == "CZ_TZ,MZ_TZ"
    assert by_id["27"] == "CA_TP,MA_TP"
    assert by_id["28"] == "CN_TP,MN_TP"
    assert by_id["29"] == "CZ_TP,MZ_TP"
    assert by_id["31"] == "CA_TP"
    assert "CN_TP" in by_id["30"]
    assert "CZ_TP" in by_id["32"]
    assert by_id["34"] == "MA_TP"
    assert len(payload["items"]) == 32

    level = {item["manuscript_id"]: item["answer"] for item in payload["items"]}
    zeros = {sp.Symbol("gamma_M"): 0, sp.Symbol("gamma_tp"): 0}
    for manuscript_id, symbol in (("30", "CN_TP"), ("31", "CA_TP"), ("32", "CZ_TP")):
        expr = sp.sympify(level[manuscript_id].replace("γ_M", "gamma_M").replace("γ_tp", "gamma_tp"))
        assert sp.simplify(expr.subs(zeros)) == sp.Symbol(symbol)

    inverted = json.loads(json.dumps(STANDARD_SPEC))
    inverted["placement"] = PLACE_INVERTED
    inverted_answers = {item["manuscript_id"]: item["answer"] for item in generate(inverted)["items"]}
    assert inverted_answers["14"] == "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    assert inverted_answers["15"] == "inverted"
    assert inverted_answers["16"] == "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]"
    assert [inverted_answers["17"], inverted_answers["18"], inverted_answers["19"]] == [
        "RTP_TZ2", "RTP_TZ1", "R_(γ_M-γ_tp)",
    ]
    assert inverted_answers["4"] == "γ_M - γ_tp"

    opposite = json.loads(json.dumps(STANDARD_SPEC))
    opposite["gamma_tp"] = {"sense": SENSE_OPPOSITE}
    opposite_answers = {item["manuscript_id"]: item["answer"] for item in generate(opposite)["items"]}
    assert opposite_answers["2"] == "negative"
    assert opposite_answers["4"] == "γ_M + γ_tp"

    nose_up = json.loads(json.dumps(STANDARD_SPEC))
    nose_up["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    nose_answers = {item["manuscript_id"]: item["answer"] for item in generate(nose_up)["items"]}
    assert nose_answers["3"] == "negative"
    assert nose_answers["6"] == "reverse"
    assert nose_answers["4"] == "-γ_M - γ_tp"

    rearward_body = json.loads(json.dumps(STANDARD_SPEC))
    rearward_body["body_frame"] = {
        "chirality": "left",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_rearward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    }
    rearward_body_answers = {item["manuscript_id"]: item["answer"] for item in generate(rearward_body)["items"]}
    assert rearward_body_answers["12"] == "-XTZ"
    assert rearward_body_answers["3"] == "positive"
    assert rearward_body_answers["4"] == "γ_M - γ_tp"
    assert rearward_body_answers["5"] == "XTP"
    assert rearward_body_answers["6"] == "positive"

    rearward_balance = json.loads(json.dumps(STANDARD_SPEC))
    rearward_balance["balance_frame"] = {
        "chirality": "left",
        "XTP": {"along": "balance_axial", "points": "balance_rearward"},
        "YTP": {"along": "balance_normal", "points": "balance_up"},
    }
    rearward_balance_answers = {
        item["manuscript_id"]: item["answer"] for item in generate(rearward_balance)["items"]
    }
    assert rearward_balance_answers["2"] == "positive"
    assert rearward_balance_answers["4"] == "-γ_M + γ_tp"
    assert rearward_balance_answers["9"] == "-XTP"
    assert rearward_balance_answers["5"] == "XTP"
    assert rearward_balance_answers["6"] == "reverse"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the balance-to-body cloze answer key.")
    parser.add_argument("--spec", type=Path, help="JSON declaration. Defaults to the standard upright frame.")
    parser.add_argument("--out", type=Path, default=Path("tp_tz_eval.json"))
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
