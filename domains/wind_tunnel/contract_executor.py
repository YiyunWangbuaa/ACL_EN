# -*- coding: utf-8 -*-
"""Derive the target a declared contract chain requires.

Handedness comes from the triad the model wrote (body ID5/6/7, wind ID2/3/4,
balance ID89/90/91), not from the gold sample's ``model.chirality``. A triad
that is missing or is not one signed X, Y and Z is a contradiction. The
passive rotation convention stays fixed. The declared answers still supply
the frame, axis, and turn.
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from body_frame_model import BodyFrameModel
from domains.wind_tunnel.tp_tz_generate import (
    GAMMA_M_SYM,
    GAMMA_TP_SYM,
)
from domains.wind_tunnel.tp_tz_generate import (
    _matrix_text as _roll_matrix_text,
)
from domains.wind_tunnel.tz_fd_generate import (
    ALPHA_SYM,
    BETA_SYM,
    GAMMA_SYM,
)
from domains.wind_tunnel.tz_fd_generate import (
    _matrix_text as _attitude_matrix_text,
)

_BODY = {"XTZ", "YTZ", "ZTZ"}
_WIND = {"XFD", "YFD", "ZFD"}
_BALANCE = {"XTP", "YTP", "ZYP"}
_FRAME_AXES = {
    "intermediate body-axis frame": _BODY,
    "body-axis frame": _BODY,
    "body-axis frame": _BODY,
    "wind frame": _WIND,
    "balance frame": _BALANCE,
}
# Positive normal, axial, and lateral directions: upward, forward, and right. forward × right = down.
_ROLE_VECTOR = {"normal": (0, 0, -1), "axial": (1, 0, 0), "lateral": (0, 1, 0)}
# Three blanks: normal, axial, and lateral. The answer is the directed coordinate axis that points along that absolute positive direction.
_TRIADS = {
    "TZ": ("5", "6", "7", _BODY),
    "FD": ("2", "3", "4", _WIND),
    "TP": ("89", "90", "91", _BALANCE),
}
TRIAD_SLOTS = tuple(slot for family in ("FD", "TZ", "TP") for slot in _TRIADS[family][:3])


@dataclass(frozen=True)
class Execution:
    status: str
    matrix: str = ""
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "derived"


def _signed_angle(symbol, turn: str, negative: str):
    if turn == "positive":
        return symbol
    if turn == negative:
        return -symbol
    return None


def _passive_rotation(axis_name: str, angle, left_handed: bool = False) -> sp.Matrix:
    """Passive rotation about the named coordinate axis.

    The matrix depends on the letter, the angle, and the frame's declared
    chirality. It does not read which way that letter points in the sample.
    """
    cosine = sp.cos(angle)
    sine = -sp.sin(angle) if left_handed else sp.sin(angle)
    letter = str(axis_name).strip()[:1]
    if letter == "X":
        return sp.Matrix([[1, 0, 0], [0, cosine, sine], [0, -sine, cosine]])
    if letter == "Y":
        return sp.Matrix([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    if letter == "Z":
        return sp.Matrix([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    raise ValueError(axis_name)


def _family_of_axis(axis_name: str) -> str | None:
    if axis_name in _BODY:
        return "TZ"
    if axis_name in _WIND:
        return "FD"
    if axis_name in _BALANCE:
        return "TP"
    return None


def _signed_axis(text: str, allowed: set[str]) -> tuple[str, int]:
    raw = str(text).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    if name not in allowed:
        raise ValueError(f"{raw!r} is not a coordinate axis of this frame.")
    return name, -1 if negative else 1


def _cross(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: tuple[int, int, int], right: tuple[int, int, int]) -> int:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _left_from_roles(normal: str, axial: str, lateral: str, allowed: set[str]) -> bool:
    """Left-handed when the stated X, Y, Z are not a right-handed basis."""
    vectors: dict[str, tuple[int, int, int]] = {}
    for text, role in ((normal, "normal"), (axial, "axial"), (lateral, "lateral")):
        name, sign = _signed_axis(text, allowed)
        letter = name[:1]
        if letter in vectors:
            raise ValueError("The three coordinate axes must be exactly one X, one Y, and one Z.")
        base = _ROLE_VECTOR[role]
        vectors[letter] = tuple(sign * item for item in base)
    if set(vectors) != {"X", "Y", "Z"}:
        raise ValueError("The three coordinate axes must be exactly one X, one Y, and one Z.")
    triple = _dot(vectors["X"], _cross(vectors["Y"], vectors["Z"]))
    if triple == 1:
        return False
    if triple == -1:
        return True
    raise ValueError("The triple is neither left-handed nor right-handed.")


def recognized_left_handed(texts: dict[str, str], family: str) -> bool:
    """Handedness implied by the model's triad for one frame family."""
    normal_id, axial_id, lateral_id, allowed = _TRIADS[family]
    missing = [item for item in (normal_id, axial_id, lateral_id) if not str(texts.get(item, "")).strip()]
    if missing:
        raise ValueError("The triple is incomplete, so the handedness cannot be determined.")
    return _left_from_roles(texts[normal_id], texts[axial_id], texts[lateral_id], allowed)


def _resolved_left_handed(model: BodyFrameModel, axis_name: str, texts: dict[str, str] | None):
    """Handedness of the triad the answers actually wrote. Missing answers are not the gold frame."""
    del model
    family = _family_of_axis(axis_name)
    if family is None:
        return False
    if texts is None:
        return Execution("contradictory", reason="The model did not write the triple, so the handedness cannot be determined.")
    try:
        return recognized_left_handed(texts, family)
    except ValueError as exc:
        return Execution("contradictory", reason=str(exc))


def attitude_target(
    model: BodyFrameModel,
    symbol,
    frame: str,
    axis_name: str,
    turn: str,
    texts: dict[str, str] | None = None,
) -> Execution:
    """Passive matrix for one declared attitude rotation."""
    family = _FRAME_AXES.get(str(frame).strip())
    axis_name = str(axis_name).strip()
    if family is None:
        return Execution("contradictory", reason=f"Frame {frame!r} is not the intermediate body-axis frame, the wind frame, or the balance frame.")
    if axis_name not in family:
        return Execution("contradictory", reason=f"Coordinate axis {axis_name} does not belong to the declared {frame}.")
    angle = _signed_angle(symbol, str(turn).strip(), "negative")
    if angle is None:
        return Execution("contradictory", reason=f"Turn {turn!r} does not determine the sign of the rotation angle.")
    left = _resolved_left_handed(model, axis_name, texts)
    if isinstance(left, Execution):
        return left
    matrix = sp.simplify(_passive_rotation(axis_name, angle, left))
    return Execution("derived", _attitude_matrix_text(matrix))


def _balance_sign(text: str) -> tuple[str, int] | None:
    raw = str(text).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    if name not in _BALANCE:
        return None
    return name, -1 if negative else 1


def roll_target(
    model: BodyFrameModel,
    hand_balance: str,
    hand_body: str,
    axis_name: str,
    turn: str,
    normal: str = "",
    axial: str = "",
    lateral: str = "",
) -> Execution:
    """Roll matrix from the stated hands, the balance triad, the axis, and the turn."""
    hand_balance = str(hand_balance).strip()
    hand_body = str(hand_body).strip()
    axis_name = str(axis_name).strip()
    turn = str(turn).strip()
    if hand_balance not in {"positive", "negative"} or hand_body not in {"positive", "negative"}:
        return Execution("contradictory", reason="The positive direction of the installation angle or the airframe roll is neither positive nor negative.")
    if turn not in {"positive", "reverse"}:
        return Execution("contradictory", reason=f"Turn {turn!r} is neither positive nor reverse.")
    if axis_name not in _BALANCE:
        return Execution("contradictory", reason=f"Roll axis {axis_name} is not a balance coordinate axis.")
    parsed = [_balance_sign(item) for item in (normal, axial, lateral)]
    if any(item is None for item in parsed):
        return Execution("contradictory", reason="The balance-axis binding is not a coordinate axis of this frame.")
    letters = [item[0][0] for item in parsed]
    if set(letters) != {"X", "Y", "Z"}:
        return Execution("contradictory", reason="The three balance axes must be exactly one X, one Y, and one Z.")
    declared, pointing = parsed[1]
    if axis_name != declared:
        return Execution("contradictory", reason=f"Roll holds only on the declared axis {declared!r}; the text says {axis_name!r}.")
    body = 1 if hand_body == "positive" else -1
    installation = 1 if hand_balance == "positive" else -1
    implied = "positive" if pointing * body == 1 else "reverse"
    if turn != implied:
        return Execution(
            "contradictory",
            reason=f"Handedness and the axial direction imply turn {implied}, which contradicts the declared {turn}.",
        )
    theta = sp.simplify(pointing * (body * GAMMA_M_SYM - installation * GAMMA_TP_SYM))
    try:
        left = _left_from_roles(normal, axial, lateral, _BALANCE)
    except ValueError as exc:
        return Execution("contradictory", reason=str(exc))
    matrix = sp.simplify(_passive_rotation(axis_name, theta, left))
    del model
    return Execution("derived", _roll_matrix_text(matrix))


def _model(spec: dict) -> BodyFrameModel:
    return BodyFrameModel().declare(spec)


def generated_answers(spec: dict) -> dict[str, str]:
    """Every public answer on the full question set."""
    from domains.wind_tunnel.dataset_generate import _public_id
    from domains.wind_tunnel.ref_point_generate import generate as reference_generate
    from domains.wind_tunnel.tp_fd_generate import generate as balance_wind_generate
    from domains.wind_tunnel.tp_tz_generate import generate as balance_generate
    from domains.wind_tunnel.tz_fd_generate import generate as wind_generate

    found = {}
    for process, payload in (
        ("wind frame to body-axis frame", wind_generate(spec)),
        ("balance frame to body-axis frame", balance_generate(spec)),
        ("balance frame to wind frame", balance_wind_generate(spec)),
        ("moment reference point", reference_generate(spec)),
    ):
        for item in payload["items"]:
            found[_public_id(process, str(item["manuscript_id"]))] = str(item["answer"])
    return found


def _toggle_sign(text: str) -> str:
    raw = str(text).strip()
    return raw[1:] if raw.startswith("-") else f"-{raw}"


def _check_recognized_chirality(model: BodyFrameModel, gold: dict[str, str]) -> None:
    """A consistent wrong triad is the contract. A broken triad has no matrix."""
    from cloze_score import sympy_equal

    frame, axis, turn = gold["23"], gold["24"], gold["25"]
    faithful = dict(gold)
    faithful["6"] = _toggle_sign(faithful["6"])
    if recognized_left_handed(faithful, "TZ") == (model.chirality == "left"):
        raise RuntimeError("Flipping one sign of the body-axis triple left the handedness equal to the gold.")
    gold_matrix = attitude_target(model, ALPHA_SYM, frame, axis, turn, gold)
    model_matrix = attitude_target(model, ALPHA_SYM, frame, axis, turn, faithful)
    if not gold_matrix.ok or not model_matrix.ok or sympy_equal(gold_matrix.matrix, model_matrix.matrix):
        raise RuntimeError("The angle-of-attack matrix recomputed from the model triple did not leave the gold handedness.")
    broken = dict(gold)
    broken["5"] = broken["6"]
    rejected = attitude_target(model, ALPHA_SYM, frame, axis, turn, broken)
    if rejected.ok:
        raise RuntimeError("Repeated body-axis coordinates still produced a matrix.")
    missing = dict(gold)
    missing["5"] = ""
    absent = attitude_target(model, ALPHA_SYM, frame, axis, turn, missing)
    if absent.ok:
        raise RuntimeError("A missing body-axis triple blank still produced a matrix.")


def check_executor(specs: dict[str, dict], answers: dict[str, dict[str, str]]) -> None:
    """Gold declarations must reproduce the generator matrix. Contradictions stay excluded."""
    from cloze_score import sympy_equal

    standard = _model(specs["standard-default"])
    cases = (
        ("standard-default", standard, "31", "32", "33", BETA_SYM, "34"),
        ("standard-default", standard, "40", "41", "42", GAMMA_SYM, "43"),
        ("standard-default", standard, "23", "24", "25", ALPHA_SYM, "26"),
    )
    for sample_id, model, frame_id, axis_id, turn_id, symbol, target_id in cases:
        gold = answers[sample_id]
        derived = attitude_target(model, symbol, gold[frame_id], gold[axis_id], gold[turn_id], gold)
        if not derived.ok or not sympy_equal(derived.matrix, gold[target_id]):
            raise RuntimeError(f"The gold contract of {sample_id} ID{target_id} did not produce the gold matrix: {derived}")
    body = _model(specs["body-rule"])
    for frame_id, axis_id, turn_id, symbol, target_id in (
        ("23", "24", "25", ALPHA_SYM, "26"),
        ("31", "32", "33", BETA_SYM, "34"),
        ("40", "41", "42", GAMMA_SYM, "43"),
    ):
        gold = answers["body-rule"]
        derived = attitude_target(body, symbol, gold[frame_id], gold[axis_id], gold[turn_id], gold)
        if not derived.ok or not sympy_equal(derived.matrix, gold[target_id]):
            raise RuntimeError(f"The gold contract of body-rule ID{target_id} did not produce the gold matrix.")
    standard_texts = answers["standard-default"]
    for family, attr in (("TZ", "chirality"), ("FD", "wind_chirality"), ("TP", "balance_chirality")):
        got = recognized_left_handed(standard_texts, family)
        want = getattr(standard, attr) == "left"
        if got != want:
            raise RuntimeError(f"The triple handedness of standard manuscript {family} disagrees with the manuscript attribute {attr}.")
    wind_path = attitude_target(standard, ALPHA_SYM, "wind frame", "ZFD", "positive", standard_texts)
    if not wind_path.ok:
        raise RuntimeError("The wind-frame path should produce a matrix.")
    balance_path = attitude_target(standard, BETA_SYM, "balance frame", "YTP", "positive", standard_texts)
    if not balance_path.ok:
        raise RuntimeError("The balance-frame path should produce a matrix.")
    clash = attitude_target(standard, BETA_SYM, "intermediate body-axis frame", "YFD", "positive", standard_texts)
    if clash.ok:
        raise RuntimeError("An intermediate body-axis frame paired with a wind-frame axis should be contradictory.")
    unsure = attitude_target(standard, BETA_SYM, "intermediate body-axis frame", "YTZ", "indeterminate", standard_texts)
    if unsure.ok:
        raise RuntimeError("An indeterminate turn should be contradictory.")
    _check_recognized_chirality(standard, standard_texts)
    for sample_id in ("standard-default", "roll-rule"):
        gold = answers[sample_id]
        derived = roll_target(
            _model(specs[sample_id]),
            gold["83"], gold["84"], gold["86"], gold["87"],
            gold["89"], gold["90"], gold["91"],
        )
        if not derived.ok or not sympy_equal(derived.matrix, gold["88"]):
            raise RuntimeError(f"The roll contract of {sample_id} did not produce the gold matrix: {derived}")
    standard_gold = answers["standard-default"]
    opposed = roll_target(
        standard, "positive", "negative", standard_gold["86"], "positive",
        standard_gold["89"], standard_gold["90"], standard_gold["91"],
    )
    if opposed.ok:
        raise RuntimeError("A negative airframe roll whose turn is still declared positive should be contradictory.")
    print("executor check ok")
