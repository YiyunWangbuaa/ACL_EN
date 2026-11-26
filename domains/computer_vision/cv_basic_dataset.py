# -*- coding: utf-8 -*-
"""Injection for the basic-theory vision set.

One subprocess, 62 self-contained blanks (first-camera-as-world). Prompts use the
same slot format as the computer-vision sets.
"""
from __future__ import annotations

import json

from domains.computer_vision.cv_dataset import PRIOR, _PLAIN_VALUE, _injection, _render_process

PROCESS = "vision basics"
N_BLANKS = 66


def _sentence_groups() -> list[list[str]]:
    return [[str(index)] for index in range(1, N_BLANKS + 1)]


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
        lead, "", "[Declaration]", chain["declaration"].strip(), "", "[Background]",
        PRIOR, "", "[Subprocess]", PROCESS, "", "[Process]",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
