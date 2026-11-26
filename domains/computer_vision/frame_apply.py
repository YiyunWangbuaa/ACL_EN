# -*- coding: utf-8 -*-
"""Application blanks whose answers are fixed by an earlier recognition."""
from __future__ import annotations

from domains.computer_vision import cv_model as cm

_WORDS = {
    "right": "right", "left": "left",
    "up": "up", "upward": "up",
    "down": "down", "downward": "down",
    "forward": "forward", "front": "forward",
    "rearward": "rearward", "rear": "rearward",
}
_ROLES = (("normal", "normal"), ("lateral", "lateral"), ("axial", "axial"))
_ANGLES = {"azimuth": "ψ", "pitch": "θ", "roll": "φ"}


def _bind(answer: str, direction):
    raw = str(answer).strip()
    negative = raw.startswith("-")
    name = raw[1:] if negative else raw
    return name, (cm._neg(direction) if negative else direction)


def frame_from(texts: dict[str, str], slots: tuple[str, str, str]) -> dict:
    """slots are axial, normal, lateral, each naming the coordinate on that direction."""
    found = {}
    for slot, direction in zip(slots, (cm.FORWARD, cm.UP, cm.RIGHT)):
        name, vector = _bind(texts[slot], direction)
        if name in found:
            raise ValueError(f"ID{slot} uses the same coordinate name as another axis.")
        found[name] = vector
    if len(found) != 3:
        raise ValueError("The three axes do not each match one absolute direction.")
    return found


def _role(text: str) -> str:
    for word, role in _ROLES:
        if word in str(text):
            return role
    raise ValueError(f"absolute direction is not axial, normal, or lateral: {text}")


def positive_axis(pointing: str) -> str:
    raw = str(pointing).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    if not raw:
        raise ValueError("coordinate axis name is empty.")
    return raw


def rotation_sign(texts: dict[str, str], triad: tuple[str, str, str], role_id: str, axis_id: str, direction_id: str, moving_role: str) -> str:
    """Right-hand sign of the stated axis, given where the moving direction goes."""
    frame = frame_from(texts, triad)
    axis = positive_axis(texts[axis_id])
    if axis not in frame:
        raise ValueError(f"rotation axis {axis} is not one of the three declared axes.")
    direction = _WORDS.get(str(texts[direction_id]).strip())
    if direction is None:
        raise ValueError(f"deflection direction is not recognized: {texts[direction_id]}")
    moving = cm.axis_on(frame, moving_role)[1]
    handed = cm.sense(frame[axis], moving, cm.DIRECTIONS[direction])
    stated = _role(texts[role_id])
    actual = cm.role_of(frame[axis])
    if stated != actual:
        raise ValueError(f"the axis used is {actual}, the stated absolute direction is {stated}.")
    return "positive" if handed > 0 else "negative"


def product_names(texts: dict[str, str], order: tuple[str, str, str], suffix: str, sense_id: str, kind_id: str, spin_id: str, tail: str) -> tuple[str, str, str, str]:
    """Passive intrinsic keeps the last Euler angle on the left. Active reverses the product."""
    sense = str(texts[sense_id]).strip()
    kind = str(texts[kind_id]).strip()
    spin = str(texts[spin_id]).strip()
    paired = {("moving axes", "intrinsic"): False, ("fixed axes", "extrinsic"): True}
    if (kind, spin) not in paired:
        raise ValueError("moving axes must be paired with intrinsic, and fixed axes with extrinsic.")
    if sense not in {"active", "passive"}:
        raise ValueError("the coordinate transform is not active or passive.")
    factors = []
    for node in order:
        angle = str(texts[node]).strip()
        if angle not in _ANGLES:
            raise ValueError(f"Euler order is not azimuth, pitch, or roll: {angle}")
        factors.append(f"R_{_ANGLES[angle]}{suffix}")
    extrinsic = paired[(kind, spin)]
    slots = tuple(factors) if extrinsic else (factors[2], factors[1], factors[0])
    if sense == "active":
        slots = tuple(reversed(slots))
    return slots + (tail,)


def mount_agreement(placement: str) -> str:
    text = str(placement).strip()
    if text in {"inverted", "reversed"}:
        return "opposite"
    if text == "upright":
        return "same"
    raise ValueError(f"installation does not decide whether the absolute directions agree: {text}")
