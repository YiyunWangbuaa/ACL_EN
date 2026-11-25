# -*- coding: utf-8 -*-
"""Application blanks whose answers are fixed by an earlier recognition."""
from __future__ import annotations

from domains.computer_vision import cv_model as cm

_WORDS = {
    "右侧": "right", "左侧": "left", "上方": "up", "下方": "down",
    "前方": "forward", "后方": "rearward",
}
_ROLES = (("法向", "normal"), ("侧向", "lateral"), ("轴向", "axial"))
_ANGLES = {"方位角": "ψ", "俯仰角": "θ", "滚转角": "φ"}


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
            raise ValueError(f"ID{slot} 与另一根轴使用了同一个坐标名。")
        found[name] = vector
    if len(found) != 3:
        raise ValueError("三根坐标轴没有各对应一个绝对方向。")
    return found


def _role(text: str) -> str:
    for word, role in _ROLES:
        if word in str(text):
            return role
    raise ValueError(f"绝对方向不是轴向、法向或侧向：{text}")


def positive_axis(pointing: str) -> str:
    raw = str(pointing).strip()
    if raw.startswith("-"):
        raw = raw[1:]
    if not raw:
        raise ValueError("坐标轴名为空。")
    return raw


def rotation_sign(texts: dict[str, str], triad: tuple[str, str, str], role_id: str, axis_id: str, direction_id: str, moving_role: str) -> str:
    """Right-hand sign of the stated axis, given where the moving direction goes."""
    frame = frame_from(texts, triad)
    axis = positive_axis(texts[axis_id])
    if axis not in frame:
        raise ValueError(f"旋转轴 {axis} 不在已声明的三根坐标轴里。")
    direction = _WORDS.get(str(texts[direction_id]).strip())
    if direction is None:
        raise ValueError(f"偏转方向无法识别：{texts[direction_id]}")
    moving = cm.axis_on(frame, moving_role)[1]
    handed = cm.sense(frame[axis], moving, cm.DIRECTIONS[direction])
    stated = _role(texts[role_id])
    actual = cm.role_of(frame[axis])
    if stated != actual:
        raise ValueError(f"所绕的轴是{actual}，写的绝对方向是{stated}。")
    return "正" if handed > 0 else "负"


def product_names(texts: dict[str, str], order: tuple[str, str, str], suffix: str, sense_id: str, kind_id: str, spin_id: str, tail: str) -> tuple[str, str, str, str]:
    """Passive intrinsic keeps the last Euler angle on the left. Active reverses the product."""
    sense = str(texts[sense_id]).strip()
    kind = str(texts[kind_id]).strip()
    spin = str(texts[spin_id]).strip()
    paired = {("动轴", "内旋"): False, ("定轴", "外旋"): True}
    if (kind, spin) not in paired:
        raise ValueError("动轴必须配内旋，定轴必须配外旋。")
    if sense not in {"主动", "被动"}:
        raise ValueError("坐标系转换不是主动或被动。")
    factors = []
    for node in order:
        angle = str(texts[node]).strip()
        if angle not in _ANGLES:
            raise ValueError(f"欧拉角顺序不是方位、俯仰、滚转：{angle}")
        factors.append(f"R_{_ANGLES[angle]}{suffix}")
    extrinsic = paired[(kind, spin)]
    slots = tuple(factors) if extrinsic else (factors[2], factors[1], factors[0])
    if sense == "主动":
        slots = tuple(reversed(slots))
    return slots + (tail,)


def mount_agreement(placement: str) -> str:
    text = str(placement).strip()
    if "反" in text or "倒" in text:
        return "相反"
    if "正" in text:
        return "相同"
    raise ValueError(f"安装方式无法判断绝对方向是否一致：{text}")
