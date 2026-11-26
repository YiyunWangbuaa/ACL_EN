# -*- coding: utf-8 -*-
"""Injection module for the CV-CTRL machine-vision control (control group)."""
from __future__ import annotations

import json

from domains.computer_vision.cv_dataset import _PLAIN_VALUE, _requirement

PROCESS = "world to camera"

PRIOR = (
    "Pinhole imaging is x = (K·S)·[R|t]·X. An image line is l = (K·S)^(-T)·n, where n is the normal of the plane spanned by the optical center and the spatial line. "
    "The epipolar constraint is x2^T·F·x1 = 0, with F = (K2·S2)^(-T)·E·(K1·S1)^(-1) and E = [t21]_×·R21. "
    "Rotation matrices use a passive transform and an intrinsic moving-axis sequence. Write every matrix element; do not expand a product."
)

N_BLANKS = 66


def _sentence_groups() -> list[list[str]]:
    return [[str(i)] for i in range(1, N_BLANKS + 1)]


def _render_prompt(item: dict) -> str:
    text = item["prompt"]
    options = item.get("options") or []
    if options:
        text += "\nOptions: " + " ".join(f"{chr(65 + i)}.{opt}" for i, opt in enumerate(options))
    return text


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix=False) -> str:
    del memory, keep_suffix
    item = by_id[target]
    prompt = _render_prompt(item)
    return f"{declaration.strip()}\n\n[Question]{prompt}\n[Requirement] {_requirement([item])}"


def _render_process(by_id: dict, targets: list[str], keep_suffix=False) -> str:
    return "\n".join(_render_prompt(by_id[target]) for target in targets)


def _json_object(items: list[dict], example: bool = False) -> str:
    if example:
        return json.dumps({item["id"]: "example" for item in items}, ensure_ascii=False)
    return json.dumps({item["id"]: "?" for item in items}, ensure_ascii=False)


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [str(i) for i in range(1, N_BLANKS + 1)]
    items = [by_id[lid] for lid in local_ids]
    lead = (
        "This is an end-to-end question. Fill every blank in this subprocess, and call the model only once. "
        "Output only one JSON object. Each key is a blank id and each value is the answer. "
        "A choice value is a letter, without quotes. A matrix value is a string in English double quotes. "
        "Write one value for each blank id. Do not merge several blanks into one expression without blank ids. "
        "Do not output the original sentence and do not explain. "
        + _PLAIN_VALUE
    )
    output = "\n".join([
        "[Output]",
        "Put every answer in one JSON object. Each key is a blank id and each value is the answer.",
        "A choice value is only a letter, without quotes. A matrix value must be a string in English double quotes.",
        "Write one value for each blank id. Do not merge several blanks into one expression without blank ids.",
        _PLAIN_VALUE,
        f"Output only one line {_json_object(items)}.",
        f"Format example (not the answer to this item): {_json_object(items, example=True)}",
    ])
    return "\n".join([
        lead, "", "[Declaration]", chain["declaration"].strip(), "", "[Background]",
        PRIOR, "", "[Subprocess]", PROCESS, "", "[Process]",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
