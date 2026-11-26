# -*- coding: utf-8 -*-
"""Injection for the flight-mechanics control. The cloze is one manuscript, like the wind tunnel."""
from __future__ import annotations

import re

from cloze_score import format_choices, resolve_choice
from domains.wind_tunnel.dataset_generate import FORMAT_EXAMPLE, FORMAT_TEXT

PROCESS = "body relative to ground"
PRIOR = ""
N_BLANKS = 81
SLOT_MARK = re.compile(r"@@(\d+)@@")

# Parallel to wind-tunnel "wind frame to body-axis frame" blanks 1-81.
# The wind frame is written as the ground frame, the body-axis frame as the body frame,
# and the balance frame as the airflow frame.
# Blanks separated by a sentence period are answered in separate steps.
# Blanks in the same sentence, with no period between them, are answered together.
PROCESS_TEXT = """1. In the initial state the body axes stay level. To compute the effect of the mechanism angles α_M, β_M, and γ_M on the theoretical angle of attack α, sideslip β, and roll, the required transform is @@1@@.

2. The positive absolute normal (upward), positive absolute axial (forward), and positive absolute lateral (right) of the ground frame correspond to @@2@@, @@3@@, and @@4@@. The same three directions of the body frame correspond to @@5@@, @@6@@, and @@7@@. The pre-transform from the ground frame to the body frame at the level pose, RP1, is @@8@@. That matrix makes the relation between the axes of @@9@@ and its own absolute directions the same as the relation between the axes of @@10@@ and its own absolute directions. The model installation is @@11@@. At this installation the positive absolute normal (upward) and positive absolute lateral (right) of the model are @@12@@ the positive absolute normal and positive absolute lateral of the ground. Another pre-transform, RP2, is therefore @@13@@. That matrix makes the axis directions of @@14@@ the same as the coordinate directions of @@15@@.

3. From the mechanism layout, the first rotation angle is @@16@@. The second rotation angle is @@17@@. The third rotation angle is @@18@@.

4. When α_M > 0 the model uses @@19@@. When α_M > 0 the airspeed points @@20@@ on the body. When the theoretical angle of attack is positive the airspeed points @@21@@ on the body. When α_M > 0 the sign of the theoretical angle of attack is @@22@@. This is the level pose of @@23@@, rotating about @@24@@. The sense about that axis is @@25@@, through the angle α_M, and the rotation matrix R_α_M is @@26@@.

5. When β_M > 0 the model uses @@27@@. When β_M > 0 the airspeed points @@28@@ on the body. When the theoretical sideslip is positive the airspeed points @@29@@ on the body. When β_M > 0 the sign of the theoretical sideslip is @@30@@. This is the level pose of @@31@@, rotating about @@32@@. The sense about that axis is @@33@@, through the angle β_M, and the rotation matrix R_β_M is @@34@@.

6. When γ_M > 0 the rotation axis is the positive absolute model axis (forward) and the sense is @@35@@ the right-hand rule. The right wing then points @@36@@ on the body. When the theoretical roll is positive the rotation axis is the positive absolute body axis (forward) and the sense is @@37@@ the right-hand rule. The right wing then points @@38@@ on the body. When γ_M > 0 the sign of the theoretical roll is @@39@@. This is the level pose of @@40@@, rotating about the coordinate axis @@41@@. The sense about that axis is @@42@@, through the angle γ_M, and the rotation matrix R_γ_M is @@43@@.

7. This step changes coordinates, so the coordinate transform is @@44@@. The body frame is @@45@@, so the rotation uses @@46@@. The axis correspondence is transformed before the attitude. Write [xb,yb,zb]^T=R1*R2*R3*R4*R5*[Xg,Yg,Zg]^T. Choose R1 through R5 from the matrix names already obtained. In the formula, R1 is @@47@@. R2 is @@48@@. R3 is @@49@@. R4 is @@50@@. R5 is @@51@@.

8. Before the total matrix, write each factor in the symbols just chosen. The expansion of R1 is @@52@@. The expansion of R2 is @@53@@. The expansion of R3 is @@54@@. The symbolically simplified total matrix R_t = R1*R2*R3 is @@55@@.

9. The airspeed coordinates in the level body frame do not depend on the ground-frame coordinates, so RP2*RP1 times the ground-frame airspeed unit vector is @@56@@. It is therefore enough to map the airspeed coordinates in @@57@@ into the body frame after the attitude change.

10. Compute [xb,yb,zb]^T=R_t*[X_V,Y_V,Z_V]^T. The airspeed unit vector [X_V,Y_V,Z_V] in the level frame is @@58@@. Substitution gives xb = @@59@@. It gives yb = @@60@@. It gives zb = @@61@@.

11. When only α_M > 0, xb is @@62@@. yb is @@63@@. zb is @@64@@.

12. When only β_M > 0, xb is @@65@@. yb is @@66@@. zb is @@67@@.

13. The template for α is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The angle sign of α is @@68@@. The inverse function of α is @@69@@. The numerator sign of α is @@70@@. The numerator component of α is @@71@@. The denominator sign of α is @@72@@. The denominator component of α is @@73@@.

14. The template for β is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The angle sign of β is @@74@@. The inverse function of β is @@75@@. The numerator sign of β is @@76@@. The numerator component of β is @@77@@. The denominator sign of β is @@78@@. The denominator component of β is @@79@@.

15. In the initial state the body axes stay level, and the freestream is aligned with the ground-frame axis and blows toward the nose. After the mechanism changes α_M, β_M, and γ_M, the body is in a new attitude. From the known α_M, β_M, and γ_M, derive the theoretical flow angles α and β at that attitude. The template given here is: the template for the theoretical angle of attack α is [angle sign][inverse trigonometric function]([numerator]/[denominator]), and the template for β is [angle sign][inverse trigonometric function]([numerator]/[denominator]). The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle; the sign is judged separately and may be written only outside the inverse function. Do not put the sign inside the inverse function in place of the outer sign. The theoretical angle of attack is α = @@80@@. The theoretical sideslip is β = @@81@@."""


def _period_index(tail: str) -> int:
    """Index of the sentence-ending period, or -1.

    An ASCII period followed by whitespace or the end of the tail closes a
    sentence. A period inside a token does not.
    """
    for index, char in enumerate(tail):
        if char != ".":
            continue
        nxt = tail[index + 1] if index + 1 < len(tail) else ""
        if nxt in ("", " ", "\n", "\t"):
            return index
    return -1


def _sentence_break(chunk: str) -> bool:
    return _period_index(chunk) != -1


def _sentence_groups() -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    cursor = 0
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        if _sentence_break(PROCESS_TEXT[cursor:match.start()]) and current:
            groups.append(current)
            current = []
        current.append(match.group(1))
        cursor = match.end()
    if current:
        groups.append(current)
    return groups


def _options_text(item: dict) -> str:
    options = [str(option) for option in (item.get("options") or [])]
    return format_choices(options)


def _gold_text(item: dict) -> str:
    options = [str(option) for option in (item.get("options") or []) if option]
    if not options:
        return str(item["answer"])
    _letter, text = resolve_choice(str(item["answer"]), options)
    return text


def _review_slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, options:{_options_text(item)}, format:{item['format_code']}, "
        f"label:{item['label']}, answer:{_gold_text(item)}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, options:{_options_text(item)}, format:{item['format_code']}}}"


def _json_field(item: dict, example: bool, letter: str) -> str:
    blank_id = item["id"]
    if item.get("options"):
        return f"{blank_id}:{letter if example else 'letter'}"
    if example:
        return f'{blank_id}:"{FORMAT_EXAMPLE[item["format_code"]]}"'
    return f'{blank_id}:"answer"'


def _json_object(items: list[dict], example: bool = False) -> str:
    fields = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_index = 0
    for item in items:
        letter = letters[choice_index % len(letters)]
        if item.get("options"):
            choice_index += 1
        fields.append(_json_field(item, example, letter))
    return "{" + ", ".join(fields) + "}"


def _output_preamble() -> list[str]:
    return [
        "[Output]",
        "Put every answer in one JSON object. The key is the blank id and the value is the answer, for example {1:A}.",
        "This is a step-by-step question: one sentence produces one JSON object, and every blank in that sentence is written in that object.",
        "Only an end-to-end question puts every answer of the whole subprocess in one JSON object. Here, write only the current sentence.",
        "For a multiple-choice blank, write only the letter, with no quotation marks. A formula, matrix, vector, expression, or distance must be a string in English double quotes.",
    ]


def _angle_symbol_note() -> str:
    return (
        "Angle symbols in a formula are the symbols of this item: α_M, β_M, and γ_M. "
        "The symbol θ appears only in format examples; do not write θ in an answer."
    )


def _output_block(item: dict) -> str:
    lines = _output_preamble()
    lines.append(f"Write only one line {_json_object([item])}.")
    if item.get("options"):
        lines.append("Do not write the option text, a period, an explanation, Markdown, or LaTeX.")
    else:
        lines.extend([
            "The double quotes only mark a JSON string; they are not part of the answer. Do not add an explanation, Markdown, or LaTeX outside the quotes.",
            _angle_symbol_note(),
            FORMAT_TEXT[item["format_code"]],
        ])
    lines.append(f"Format example (not the answer to this blank): {_json_object([item], example=True)}")
    return "\n".join(lines)


def _output_block_many(by_id: dict, blank_ids: list[str]) -> str:
    items = [by_id[blank_id] for blank_id in blank_ids]
    lines = _output_preamble()
    lines.extend([
        f"Write only one line {_json_object(items)}. Keep the blanks in id order inside the same object, and do not split them into several JSON objects.",
        "Do not write the sentence text.",
        _angle_symbol_note(),
    ])
    for item in items:
        if item.get("options"):
            lines.append(f"The value of ID {item['id']} is an option letter.")
        else:
            code = item["format_code"]
            lines.append(
                f"The value of ID {item['id']} is a {code} and must be a string in English double quotes. {FORMAT_TEXT[code]}"
            )
    lines.append(f"Format example (not the answer to these blanks): {_json_object(items, example=True)}")
    return "\n".join(lines)


def _render_process(by_id: dict, targets: list[str] | None = None, keep_suffix: bool = False, memory=None) -> str:
    open_ids = list(targets or [])
    open_set = set(open_ids)
    last_open = open_ids[-1] if open_ids else None
    pieces: list[str] = []
    cursor = 0
    seen: list[str] = []
    seen_open = False
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        blank_id = match.group(1)
        if blank_id not in by_id:
            raise KeyError(f"body relative to ground has no answer for ID {blank_id}.")
        seen.append(blank_id)
        pieces.append(PROCESS_TEXT[cursor:match.start()])
        item = by_id[blank_id]
        if not open_set:
            pieces.append(_review_slot(item))
        elif blank_id in open_set:
            seen_open = True
            pieces.append(_open_slot(item))
            if blank_id == last_open and not keep_suffix:
                tail = PROCESS_TEXT[match.end():]
                period = _period_index(tail)
                if period == -1:
                    raise RuntimeError(f"body relative to ground ID {blank_id} has no sentence period.")
                if SLOT_MARK.search(tail[:period]):
                    raise RuntimeError(
                        f"body relative to ground ID {blank_id} still has a blank before the sentence period."
                    )
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            if not keep_suffix:
                raise RuntimeError(
                    f"body relative to ground ID {blank_id} appears after the blanks of the current sentence."
                )
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_text(item) if memory is None else str(memory.get(item["id"], "")))
        cursor = match.end()
    if open_ids and not seen_open:
        raise KeyError(f"body relative to ground has no sentence for {open_ids}.")
    pieces.append(PROCESS_TEXT[cursor:])
    if not open_ids and seen != [str(number) for number in range(1, N_BLANKS + 1)]:
        raise RuntimeError(f"The process blanks are in the order {seen}.")
    return "".join(pieces).strip()


def _injection(declaration: str, by_id: dict, target: str, memory=None, keep_suffix: bool = False) -> str:
    group = next(item for item in _sentence_groups() if target in item)
    items = [by_id[blank_id] for blank_id in group]
    if len(group) == 1:
        lead = (
            f"Fill only the current sentence. This is a step-by-step question. Write one JSON object {_json_object(items)}. "
            "Do not explain, and do not fill any other blank."
        )
        output = _output_block(items[0])
    else:
        listed = ", ".join(group)
        lead = (
            f"Fill every blank in this sentence. The ids are {listed}. "
            "These blanks belong to one sentence, so answer the whole sentence at once. "
            f"This is a step-by-step question. Write one JSON object {_json_object(items)}. "
            "Put every blank of this sentence in that one object. Do not split them into several objects. "
            "A multiple-choice value is a letter, with no quotation marks. A formula, matrix, vector, expression, or distance is a string in English double quotes. "
            "Do not write the sentence text, and do not explain."
        )
        output = _output_block_many(by_id, group)
    if keep_suffix:
        lead += " Text after the period is context only. Do not fill those blanks."
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        declaration.strip(),
        "",
        "[Subprocess]",
        PROCESS,
        "",
        "[Process]",
        _render_process(by_id, targets=group, memory=memory, keep_suffix=keep_suffix),
        "",
        output,
    ])


def _cv_end_to_end(chain: dict) -> str:
    by_id = chain["by_id"]
    local_ids = [blank_id for group in _sentence_groups() for blank_id in group]
    lead = (
        "This is an end-to-end question. Fill every blank in this subprocess, in one call. "
        "Write one JSON object. The key is the blank id and the value is the answer. "
        "The object must contain every id of this subprocess. Do not split it into several objects, and do not answer sentence by sentence. "
        "A multiple-choice value is a letter, with no quotation marks. A formula, matrix, vector, expression, or distance is a string in English double quotes. "
        "Do not write the sentence text, and do not explain."
    )
    output = "\n".join([
        "[Output]",
        "Put every answer in the same JSON object. The key is the blank id and the value is the answer.",
        'For example {1:A, 2:C, 11:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}. The example is not the answer to this item.',
        "For a multiple-choice blank, write only the letter, with no quotation marks. A formula, matrix, vector, expression, or distance must be a string in English double quotes.",
        "Write only that one line of JSON. Do not add an explanation, Markdown, or LaTeX.",
    ])
    return "\n".join([
        lead, "", "[Declaration]", chain["declaration"].strip(), "",
        "[Subprocess]", PROCESS, "", "[Process]",
        _render_process(by_id, targets=local_ids, keep_suffix=True), "", output,
    ])
