# -*- coding: utf-8 -*-
"""Injection module for the computer-vision stereo-schlieren task.

One subprocess ("stereo pose") with 165 local slots. The
reconstruction process uses the same final-question format. Intermediate
slots stay self-contained. Slots that are also final-e2e answers share that
question text and are preceded by the earlier steps filled with gold.
"""
from __future__ import annotations

import json

PROCESS = "stereo pose"
# The reconstruction draft changes wording only. Final items share the cross-draft output format.
RECON_PROCESS = "pure stereo pose"
_VISION_PROCESSES = {PROCESS, RECON_PROCESS}

PRIOR = (
    "Pinhole imaging is x = (K·S)·[R|t]·X. An image line is l = (K·S)^(-T)·n, where n is the normal of the plane spanned by the optical center and the spatial line.\n"
    "The epipolar constraint is x2^T·F·x1 = 0, with F = (K2·S2)^(-T)·E·(K1·S1)^(-1) and E = [t21]_×·R21.\n"
    "Rotation matrices use a passive transform and an intrinsic moving-axis sequence. Write every matrix element; do not expand a product."
)

N_BLANKS = 165


def _span(start: int, stop: int) -> list[str]:
    return [str(index) for index in range(start, stop)]


# Final blanks that share one final-e2e call. The key is the group's first local id.
_FINAL_GROUP = {
    "41": ["41"],
    "84": ["84"],
    "138": ["138"],
    "139": ["139"],
    "146": ["146"],
    "147": ["147"],
    "148": ["148", "149", "150"],
    "153": ["153", "154"],
    "158": ["158", "159"],
}
# Steps already completed before the final item. Factor expansions belong in that product's prefix.
_CHAIN = {
    "41": _span(1, 41),
    "84": _span(44, 84),
    "138": [],
    "139": [],
    "146": _span(140, 146),
    "147": _span(140, 146),
    "148": _span(87, 136),
    "153": ["41", "84", "43", "86", "151", "152"],
    "158": [],
}
_FINAL_QUESTION = {
    "41": (
        "Camera 1 has azimuth, pitch, and roll ψ1, θ1, φ1. "
        "From these angles and the current axis definitions, derive the rotation from the world frame to camera 1. "
        "Give R_cw1."
    ),
    "84": (
        "Camera 2 has azimuth, pitch, and roll ψ2, θ2, φ2. "
        "From these angles and the current axis definitions, derive the rotation from the world frame to camera 2. "
        "Give R_cw2."
    ),
    "138": "Expand (K1·S1)^(-T) in the line formula by itself. Do not multiply it by [P_c1]_×, R_cw1, or d_w.",
    "139": "Expand (K2·S2)^(-T) in the line formula by itself. Do not multiply it by [P_c2]_×, R_cw2, or d_w.",
    "146": "Expand [P_c1]_× in the line formula by itself. Do not multiply it by (K1·S1)^(-T), R_cw1, or d_w.",
    "147": "Expand [P_c2]_× in the line formula by itself. Do not multiply it by (K2·S2)^(-T), R_cw2, or d_w.",
    "148": (
        "The reference direction in the world frame is d_wV. "
        "Its image line in camera 1 is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV, "
        "and in camera 2 is l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV. "
        "The unit vector of the object longitudinal axis in the object frame is e. Its direction in the world frame is d_wL. "
        "Its image line in camera 1 is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL, "
        "and in camera 2 is l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL. "
        "The unit vector of the object normal in the object frame is e. Its direction in the world frame is d_wN. "
        "Its image line in camera 1 is l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN, "
        "and in camera 2 is l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN. "
        "Write d_wV, d_wL, and d_wN."
    ),
    "153": (
        "The optical centers of camera 1 and camera 2 in the world frame are C1=(x1,y1,z1) and C2=(x2,y2,z2), "
        "with attitude angles ψ1, θ1, φ1 and ψ2, θ2, φ2. "
        "Derive the rotation R21 and the translation t21 from camera 1 to camera 2. "
        "Give R21 and t21."
    ),
    "158": (
        "The homogeneous coordinates of the reference point in the camera-1 and camera-2 images are [u1,v1,1] and [u2,v2,1]. "
        "The ray direction in camera 1 is d1=(K1·S1)^(-1)·[u1,v1,1]^T, "
        "and in camera 2 is d2=(K2·S2)^(-1)·[u2,v2,1]^T. "
        "Expand both ray directions as vectors. "
        "Give d1 and d2."
    ),
}


def _sentence_groups() -> list[list[str]]:
    """One slot per sentence, except final-e2e bundles that are one question."""
    merged = {group[0]: group for group in _FINAL_GROUP.values() if len(group) > 1}
    skip = {local_id for group in merged.values() for local_id in group[1:]}
    groups = []
    for index in range(1, N_BLANKS + 1):
        local_id = str(index)
        if local_id in skip:
            continue
        groups.append(merged.get(local_id, [local_id]))
    return groups


def _group_of(target: str) -> list[str]:
    for group in _sentence_groups():
        if target in group:
            return group
    raise KeyError(target)


def _render_prompt(item: dict) -> str:
    text = item["prompt"]
    options = item.get("options") or []
    if options:
        text += "\nOptions: " + " ".join(f"{chr(65 + i)}.{opt}" for i, opt in enumerate(options))
    return text


def _filled(item: dict, memory: dict | None) -> str:
    if memory is not None:
        return str(memory.get(str(item["id"]), ""))
    options = [str(option) for option in (item.get("options") or [])]
    answer = str(item.get("answer") or "")
    if options and len(answer) == 1 and answer.isalpha():
        index = ord(answer) - 65
        if 0 <= index < len(options):
            return options[index]
    return answer


def _chain_text(by_id: dict, local_ids: list[str], memory: dict | None) -> str:
    parts: list[str] = []
    previous = None
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            parts.append("".join(buffer).strip())
            buffer.clear()

    for local_id in local_ids:
        item = by_id[local_id]
        key = (item.get("segment"), item.get("paragraph"))
        if previous is not None and key != previous:
            flush()
        previous = key
        lead = item.get("lead") or item.get("prompt") or ""
        tail = item.get("tail") or ("" if item.get("lead") else ".")
        buffer.append(f"{lead}{_filled(item, memory)}{tail}")
    flush()
    return "\n".join(part for part in parts if part)


# Value characters match the gold: integers and symbols already in the item, not LaTeX.
_PLAIN_VALUE = (
    "Use only the notation of the gold data: integers and symbols that already appear in this item. "
    "Write multiplication as * and division as /. sin, cos, and tan must take parentheses, as in sin(θ1). "
    "Do not use LaTeX, backslashes, \\frac, subscript braces, or $. "
    "Do not write an equals sign."
)


def _value_shape(code: str) -> str:
    if code == "choice":
        return "The value is only the option letter, without quotes and without the option text."
    if code == "matrix":
        return (
            "The value is one 3×3 matrix [[a, b, c], [d, e, f], [g, h, i]], wrapped in English double quotes, with every element written. "
            "Each element is an integer or an expression in the given quantities. "
            "Do not leave P_c1x, P_c1y, P_c1z, or an unsubstituted braced subscript."
        )
    if code == "vector":
        return "The value is one row of three components [a, b, c], wrapped in English double quotes, not written as a matrix."
    return "The value is one expression, wrapped in English double quotes."


def _json_ask(items: list[dict]) -> str:
    fields = []
    for item in items:
        if item["format_code"] == "choice":
            fields.append(f"{item['id']}:letter")
        else:
            fields.append(f'{item["id"]}:"answer"')
    return "{" + ", ".join(fields) + "}"


def _requirement(items: list[dict], keep_steps: bool = False) -> str:
    shapes: list[str] = []
    seen: set[str] = set()
    for item in items:
        code = item["format_code"]
        if code not in seen:
            seen.add(code)
            shapes.append(_value_shape(code))
    text = (
        f"Output only one JSON object {_json_ask(items)}. "
        "Every key must be one of the given blank ids, and each blank id has one value. "
        "When this sentence has several blanks, do not merge them into one expression without blank ids. "
        + "".join(shapes)
        + _PLAIN_VALUE
    )
    if keep_steps:
        text += "Do not repeat the completed steps."
    return text


def _final_question(process: str, head: str) -> str:
    text = _FINAL_QUESTION[head]
    if process == RECON_PROCESS:
        from domains.computer_vision.cv_vision_context import rewrite

        return rewrite(text)
    return text


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix=False) -> str:
    del keep_suffix
    item = by_id[target]
    process = item.get("process")
    final_ids = {local_id for group in _FINAL_GROUP.values() for local_id in group}
    if process not in _VISION_PROCESSES or str(target) not in final_ids:
        prompt = _render_prompt(item)
        return f"{declaration.strip()}\n\n[Question]{prompt}\n[Requirement] {_requirement([item])}"
    group = _group_of(target)
    head = group[0]
    chain = _chain_text(by_id, _CHAIN[head], memory)
    asked = [by_id[local_id] for local_id in group]
    blocks = [declaration.strip(), ""]
    if chain:
        blocks.extend([
            "[Completed steps]",
            "The steps below are already filled with the gold answers for this final item. Do not rewrite them and do not fill them again.",
            chain,
            "",
        ])
    blocks.extend([
        "[Question]",
        _final_question(process, head),
        "[Requirement]",
        _requirement(asked, keep_steps=True),
    ])
    return "\n".join(blocks)


def _render_process(by_id: dict, targets: list[str], keep_suffix=False) -> str:
    return "\n".join(_render_prompt(by_id[target]) for target in targets)


def _json_object(items: list[dict], example: bool = False) -> str:
    if example:
        return json.dumps({item["id"]: "example" for item in items}, ensure_ascii=False)
    return json.dumps({item["id"]: "?" for item in items}, ensure_ascii=False)


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [str(index) for index in range(1, N_BLANKS + 1)]
    items = [by_id[lid] for lid in local_ids]
    lead = (
        "This is an end-to-end question. Fill every blank in this subprocess, and call the model only once. "
        "Output only one JSON object. Each key is a blank id and each value is the answer. "
        "A choice value is a letter, without quotes. Matrix and vector values are strings in English double quotes. "
        "Write one value for each blank id. Do not merge several blanks into one expression without blank ids. "
        "Do not output the original sentence and do not explain. "
        + _PLAIN_VALUE
    )
    output = "\n".join([
        "[Output]",
        "Put every answer in one JSON object. Each key is a blank id and each value is the answer.",
        "A choice value is only a letter, without quotes. Matrix and vector values must be strings in English double quotes.",
        "Write one value for each blank id. Do not merge several blanks into one expression without blank ids.",
        _PLAIN_VALUE,
        f"Output only one line {_json_object(items)}.",
        f"Format example (not the answer to this item): {_json_object(items, example=True)}",
    ])
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        chain["declaration"].strip(),
        "",
        "[Background]",
        PRIOR,
        "",
        "[Subprocess]",
        PROCESS,
        "",
        "[Process]",
        _render_process(by_id, targets=local_ids, keep_suffix=True),
        "",
        output,
    ])
