# -*- coding: utf-8 -*-
"""Review samples for ground frame to track frame and airflow frame.

Storage stays under data/flight/basic/ground_to_track_air.

Run::

    python -m domains.aerodynamics.ground_track_air_dataset
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from cloze_score import answer_letter, format_choices, resolve_choice
from domain_layout import process_dirs
from domains.aerodynamics.ground_track_air_generate import BLANK_COUNT, PROCESS, generate
from domains.aerodynamics.numbering import TOTAL_BLANKS, public_id, public_span
from flight_frame_model import (
    ANGLE_NAMES,
    COORDINATE_AXES,
    DECLARED,
    FIXED,
    GAMMA,
    INFERRED,
    MU,
    OWNERS,
    STANDARD_SPEC,
    VARPHI,
    FlightFrameModel,
)


OUT_DIR = process_dirs("aerodynamics", "ground_to_track_air")["review"]
SLOT_MARK = re.compile(r"@@(\d+)@@")
MATRIX_FORMAT = (
    "one 3×3 matrix, as [[a, b, c], [d, e, f], [g, h, i]]. "
    "Put exactly one space after each comma. Use * for multiplication."
)
FORMAT_TEXT = {"matrix": MATRIX_FORMAT}
FORMAT_EXAMPLE = {"matrix": "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}
FORMAT_CODE = {MATRIX_FORMAT: "matrix"}
HAND_EN = {"right": "right-handed", "left": "left-handed"}
SOURCE_EN = {DECLARED: "declared", INFERRED: "inferred from chirality", FIXED: "fixed"}
ROLE_EN = {"axial": "axial", "normal": "normal", "lateral": "lateral"}
OWNER_EN = {
    "ground": "ground",
    "body": "airframe",
    "airflow": "airflow",
    "stability": "stability",
    "track": "track",
}
FRAME_TITLE = {
    "ground": "Ground frame",
    "body": "Airframe",
    "airflow": "Airflow frame",
    "stability": "Stability frame",
    "track": "Track frame",
}
POINT_EN = {
    "forward": "forward",
    "rearward": "rearward",
    "up": "upward",
    "down": "downward",
    "left": "left",
    "right": "right",
}
ANGLE_HEADING = {
    "ψ": "Yaw",
    "θ": "Pitch",
    "Φ": "Roll",
    "φ": "Track azimuth",
    "μ": "Track inclination",
    "γ": "Track roll",
    "α": "Angle of attack",
    "β": "Sideslip",
}
BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
TRACK_SHOW = {"Xk": "Xp", "Yk": "Yp", "Zk": "Zp"}
PRIOR = (
    "The positive axial direction is forward, the positive normal direction is down, "
    "and the positive lateral direction is right. "
    "The positive axial direction of the ground, the airframe, the airflow, the stability frame, "
    "and the track is this absolute forward, not the positive direction of a coordinate axis. "
    "The positive lateral direction is the normal crossed into the axial direction. "
    "When yaw is transformed alone, the rotation axis is the absolute normal of the ground. "
    "When pitch is transformed alone, the rotation axis is the absolute lateral axis of the ground. "
    "When roll is transformed alone, the rotation axis is the absolute axial axis of the ground. "
    "When track azimuth is transformed alone, the rotation axis is the absolute normal of the ground. "
    "When track inclination is transformed alone, the rotation axis is the absolute lateral axis of the ground. "
    "When track roll is transformed alone, the rotation axis is the absolute axial axis of the track. "
    "When angle of attack is transformed alone, the rotation axis is the absolute lateral axis of the airframe. "
    "When sideslip is transformed alone, the rotation axis is the absolute normal of the stability frame. "
    "The positive sense of each angle is the sense in the declaration. "
    "Coordinate transforms are passive. "
    "Attitude angles, track angles, and airflow angles use an intrinsic moving-axis sequence."
)
PROCESS_TEXT = """1. From the definitions, @@1@@ is the intermediate frame in the transform from the ground frame to the airflow frame. The Euler angles between the ground frame and the track frame are @@2@@. The Euler angles between the ground frame and the airflow frame are @@3@@. The Euler angle between the track frame and the airflow frame is @@4@@.

2. The transform from the ground frame to the track frame is considered next. Every transform is @@5@@, and the track angles use @@6@@. The matrix of the angle rotated first is @@7@@. The order that leaves axes already turned undisturbed is @@8@@. Start with positive Euler angles between the ground frame and the track frame, and convert step by step from the ground frame toward the track frame. After each step the ground axes move with the rotation. Intermediate frames are not given new symbols; the ground-frame symbols are kept.

3. The positive ground normal is @@9@@, the positive ground lateral direction is @@10@@, and the positive ground axial direction is @@11@@. The positive track normal is @@12@@, the positive track lateral direction is @@13@@, and the positive track axial direction is @@14@@.

4. When track azimuth is positive, the horizontal airspeed projection lies on the @@15@@ side of Xg. Take @@16@@ as the rotation axis. Rotate through φ in the @@17@@ sense. The passive rotation matrix of φ is @@18@@.

5. When track inclination is positive, the aircraft moves @@19@@. Take @@20@@ as the rotation axis and rotate through μ in the @@21@@ sense. The passive rotation matrix of μ is @@22@@.

6. After the Euler angles between the ground frame and the track frame are removed, the absolute axes of the ground frame coincide with the track frame, but the coordinate-axis directions do not, so a matrix RP_gp is needed to align them. RP_gp=@@23@@.

7. In [xp, yp, zp]=R1*R2*R3[xg, yg, zg], choose R1 through R3 from the matrix codes already obtained. R1=@@24@@. R2=@@25@@. R3=@@26@@.

Before the total matrix, expand each chosen factor. The expansion of R1 is @@27@@. The expansion of R2 is @@28@@. The expansion of R3 is @@29@@.

8. Set Sμφ=R1*R2*R3 and compute Sμφ=@@30@@.

9. The transform from the track frame to the airflow frame is derived next. The positive airflow normal is @@31@@, the positive airflow lateral direction is @@32@@, and the positive airflow axial direction is @@33@@. Start with a positive Euler angle between the track frame and the airflow frame, and convert from the track frame toward the airflow frame. After each step the track axes move with the rotation. Intermediate frames are not given new symbols; the track-frame symbols are kept.

10. When track roll is positive, the aircraft rolls to its @@34@@. Take @@35@@ as the rotation axis. Rotate through γ in the @@36@@ sense. The passive rotation matrix of γ is @@37@@.

11. After the Euler angle between the track frame and the airflow frame is removed, the absolute axes of the two frames coincide, but the coordinate directions may not, so a matrix RP_pa is needed to align them. RP_pa=@@38@@.

12. In [xa, ya, za]=R1*R2[xp, yp, zp], choose R1 and R2 from the matrix codes already obtained. R1=@@39@@. R2=@@40@@.

Before the total matrix, expand each chosen factor. The expansion of R1 is @@41@@. The expansion of R2 is @@42@@.

13. Set Sγ=R1*R2 and compute Sγ=@@43@@.

14. The two transforms above give the matrix Sμφγ from the ground frame to the airflow frame directly. Sμφγ=R1*R2, and R1 and R2 are chosen from the matrix codes already obtained. R1=@@44@@. R2=@@45@@. Before the total matrix, expand each chosen factor. The expansion of R1 is @@46@@. The expansion of R2 is @@47@@. Compute Sμφγ=@@48@@."""
_DECLARATION_REQUIRED = (
    "Scenario",
    "Track azimuth φ:",
    "Track inclination μ:",
    "Track roll γ:",
    "Ground frame (",
    "- Xg:",
    "- Yg:",
    "- Zg:",
    "Airframe (",
    "- xb:",
    "- yb:",
    "- zb:",
    "Airflow frame (",
    "- Xa:",
    "- Ya:",
    "- Za:",
    "Stability frame (",
    "Track frame (",
    "- Xp:",
    "- Yp:",
    "- Zp:",
)
_DECLARATION_FORBIDDEN = (
    "definition change", "keeps the common default", "not counted", "is changed to",
    "not a variable definition", "Euler order", "track order", "airflow order",
)


def _samples() -> list[dict]:
    standard = deepcopy(STANDARD_SPEC)
    azimuth_negative = deepcopy(STANDARD_SPEC)
    azimuth_negative[VARPHI] = {"sign": "negative"}
    inclination_negative = deepcopy(STANDARD_SPEC)
    inclination_negative[MU] = {"sign": "negative"}
    bank_negative = deepcopy(STANDARD_SPEC)
    bank_negative[GAMMA] = {"sign": "negative"}
    track_left = deepcopy(STANDARD_SPEC)
    track_left["track_frame"] = {
        "chirality": "left",
        "Xk": {"along": "track_axial", "points": "track_forward"},
        "Yk": {"along": "track_lateral", "points": "track_right"},
    }
    airflow_lateral = deepcopy(STANDARD_SPEC)
    airflow_lateral["airflow_frame"] = {
        "chirality": "right",
        "Xa": {"along": "airflow_axial", "points": "airflow_forward"},
        "Ya": {"along": "airflow_lateral", "points": "airflow_left"},
    }
    return [
        {"id": "01", "title": "common default", "change": "All five frames and all eight angles use the standard definition.", "spec": standard},
        {"id": "02", "title": "track azimuth reversed", "change": "Track azimuth is negative when the horizontal airspeed projection lies to the right of Xg.", "spec": azimuth_negative},
        {"id": "03", "title": "track inclination reversed", "change": "Track inclination is negative when the aircraft climbs.", "spec": inclination_negative},
        {"id": "04", "title": "track roll reversed", "change": "Track roll is negative when the aircraft rolls to its right.", "spec": bank_negative},
        {"id": "05", "title": "track left-handed", "change": "The track frame is left-handed, Xp lies along the track axial direction and points forward, and Yp lies along the track lateral direction and points right.", "spec": track_left},
        {"id": "06", "title": "airflow lateral reversed", "change": "The airflow frame is right-handed, Xa lies along the airflow axial direction and points forward, and Ya lies along the airflow lateral direction and points left.", "spec": airflow_lateral},
    ]


def _direction(points_to: str) -> str:
    owner, point = points_to.split("_", 1)
    return f"{OWNER_EN[owner]} {POINT_EN[point]}"


def _axis_name(owner: str, name: str) -> str:
    if owner == "body":
        return BODY_SHOW[name]
    if owner == "track":
        return TRACK_SHOW[name]
    return name


def _frame_lines(model: FlightFrameModel, owner: str) -> list[str]:
    title = FRAME_TITLE[owner]
    lines = [f"{title} ({HAND_EN[model.chirality[owner]]})"]
    for name in COORDINATE_AXES[owner]:
        axis = model.frames[owner][name]
        role = axis.absolute_axis.removeprefix(owner + "_")
        lines.append(
            f"- {_axis_name(owner, name)}: along the {OWNER_EN[owner]} {ROLE_EN[role]} direction, "
            f"pointing {_direction(axis.points_to)} ({SOURCE_EN[axis.source]})."
        )
    return lines


def _declaration(model: FlightFrameModel) -> str:
    lines = [
        "Fixed conventions",
        "- Coordinate transforms are passive. Attitude angles, track angles, and airflow angles use an intrinsic moving-axis sequence. The matrix of the angle rotated first is the rightmost factor.",
        "- Ground axis Xg lies along the ground axial direction and points ground forward.",
        "",
        "Absolute directions",
        "The ground frame, the airframe, the airflow frame, the stability frame, and the track frame each have three absolute axes. The positive axial direction is forward, the positive lateral direction is right, and the positive normal direction is down. Forward × right = down. The six absolute directions are the two senses of these three axes.",
        "Wherever the text names the positive axial direction, the positive normal direction, or the positive lateral direction, and wherever it names the positive direction of the ground axial axis, the airframe axial axis, the airflow axial axis, the stability axial axis, or the track axial axis, it means these absolute axes, not the positive direction of the coordinate axes Xg, Yg, Zg, xb, yb, zb, Xa, Ya, Za, Xs, Ys, Zs, Xp, Yp, Zp. A coordinate-axis declaration only assigns each coordinate axis to an absolute axis and a side. That side may agree with the absolute positive direction or oppose it.",
        "The five frames share one pointing relation: normal × axial = lateral. Looking along the positive axial direction, right lies on the right-hand side. Forward × right = down. Looking forward, geometric right lies on the right-hand side, and down is vertically down. This relation does not follow a sample. The right of all five frames points geometric right.",
        "The three axes and six directions of the ground frame, relative to the ground:",
        "- Axial: the fore-aft axis in the ground plane. The positive direction is forward, and it is the positive direction of Xg. Rearward is the opposite direction.",
        "- Normal: the vertical axis of the ground. The positive direction is down, vertically down. Upward is the opposite direction, vertically up.",
        "- Lateral: the horizontal cross axis of the ground. The positive direction is the normal crossed into the axial direction, called right on the ground. Left is the opposite direction.",
        "The three axes and six directions of the airframe, relative to the aircraft:",
        "- Axial: from the tail to the nose. The positive direction is forward, toward the nose. Rearward points toward the tail.",
        "- Normal: the up-down axis in the plane of symmetry, perpendicular to the reference longitudinal axis. The positive direction is down, toward the bottom of the airframe. Upward is the opposite direction, toward the top of the airframe.",
        "- Lateral: the left-right axis perpendicular to the plane of symmetry. The positive direction is the normal crossed into the axial direction, called right on the airframe. Left is the opposite direction.",
        "The three axes and six directions of the airflow frame, relative to the airspeed:",
        "- Axial: along the airspeed. The positive direction is forward, along the airspeed. Rearward is the opposite direction.",
        "- Normal: the up-down axis in the plane of symmetry, perpendicular to the airspeed. The positive direction is down. Upward is the opposite direction, in the plane of symmetry and perpendicular to the airspeed.",
        "- Lateral: the normal crossed into the axial direction, called right in the airflow frame. Left is the opposite direction.",
        "The three axes and six directions of the stability frame, relative to the airspeed projection in the plane of symmetry:",
        "- Axial: the forward projection of the airspeed onto the plane of symmetry. The positive direction is forward, toward the nose. Rearward is the opposite direction.",
        "- Normal: the up-down axis in the plane of symmetry, perpendicular to the axial direction. The positive direction is down, toward the bottom of the airframe. Upward is the opposite direction, toward the top of the airframe.",
        "- Lateral: perpendicular to the plane of symmetry. The positive direction is the normal crossed into the axial direction, called right in the stability frame. Left is the opposite direction.",
        "The three axes and six directions of the track frame, relative to the airspeed:",
        "- Axial: along the airspeed. The positive direction is forward, along the airspeed. Rearward is the opposite direction.",
        "- Normal: the up-down axis in the vertical plane through the airspeed, perpendicular to the airspeed. The positive direction is down, toward the ground. Upward is the opposite direction, in that vertical plane, perpendicular to the airspeed, and away from the ground.",
        "- Lateral: the normal crossed into the axial direction, called right in the track frame. Left is the opposite direction.",
        "",
        "Scenario",
        "This scenario examines the relations among the aircraft, the ground, and the airflow in flight mechanics. The airframe and the ground frame describe the attitude of the aircraft: yaw, pitch, and roll. The airframe and the airflow frame describe the airflow relative to the aircraft, through the stability frame, and measure angle of attack and sideslip. The track frame and the ground frame describe the flight direction relative to the ground: track azimuth and track inclination. Track roll also lies between the track frame and the airflow frame.",
        "One special task starts with the absolute directions of the airframe and the ground frame coincident, and with the airflow along the ground axial direction, blowing toward the nose. The tail is fixed to a mechanism that carries the airframe into a special attitude by a new angle sequence. That sequence differs from the ordinary transform sequence. The mechanism angles must be understood first, and then the correct angle sequence inferred. Their definitions are given with the angles below. These angles carry the airframe, initially coincident with the ground frame, to an airframe position distinct from the ordinary attitude. At that position the theoretical airflow angles are computed again from the definitions of angle of attack and sideslip.",
        "The task has these limits:",
        "- The coordinate-axis correspondence is transformed first, and the attitude afterward. Write [xb, yb, zb]^T=R1*R2*R3*R4*R5*[Xg, Yg, Zg]^T. The product is written from right to left, the alignment matrix first and then the mechanism-angle matrices. The matrix of the mechanism angle rotated first sits next to the alignment matrix.",
        "- RP1 is the alignment matrix from the ground frame to the airframe at the level attitude. It makes the relation of the airframe axes to their own absolute directions the same as the relation of the ground axes to their own absolute directions.",
        "- Only an upright installation is considered. For an upright installation, the top of the airframe agrees with upward on the ground when the aircraft is level. The positive absolute normal of the airframe (upward) and the positive absolute lateral direction (right) then agree with the positive absolute normal and the positive absolute lateral direction of the ground. An inverted installation would put the top of the airframe opposite upward on the ground, so a second alignment matrix RP2 is also required. That matrix makes the airframe axis directions the same as the ground-frame axis directions. For an upright installation, RP2 is the identity.",
        "",
        "Angles",
    ]
    for name in ANGLE_NAMES:
        record = model.binding(name)
        lines.append(f"{ANGLE_HEADING[record.symbol]} {record.symbol}: {record.definition}.")
    lines.extend([
        "Mechanism angles",
        "Mechanism angle of attack α_M: the support rod moves in the vertical plane, which is a rotation about the lateral axis of the ground frame, and the angle is the angle between the rod (the airframe axis) and the horizontal plane. The angle is positive when the aircraft pitches above the ground.",
        "Mechanism sideslip β_M: the sideslip rod moves in the plane that is perpendicular to the vertical plane and passes through the angle-of-attack rod, so the sideslip mechanism acts on the angle-of-attack mechanism. β_M is the angle between the sideslip rod and the vertical plane. The angle is positive when the aircraft points to the right of the ground.",
        "Mechanism roll γ_M: the angle through which the aircraft rolls about the sideslip rod. When the angle is positive, the rotation about the absolute axial direction (forward) follows the right-hand rule.",
        "",
        "Frames",
    ])
    for owner in OWNERS:
        lines.extend(_frame_lines(model, owner))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _check_declaration(declaration: str) -> None:
    missing = [item for item in _DECLARATION_REQUIRED if item not in declaration]
    if missing:
        raise RuntimeError(f"The declaration is missing: {', '.join(missing)}")
    present = [item for item in _DECLARATION_FORBIDDEN if item in declaration]
    if present:
        raise RuntimeError(f"The declaration still compares changes: {', '.join(present)}")


def _question(item: dict) -> dict:
    output_format = None if item["options"] else MATRIX_FORMAT
    if item["options"] and item["answer"] not in item["options"]:
        raise RuntimeError(f"The answer of ID {item['manuscript_id']} is not in the options.")
    if output_format and not str(item["answer"]).startswith("[["):
        raise RuntimeError(f"ID {item['manuscript_id']} is not a matrix.")
    return {
        "process": PROCESS,
        "id": public_id(PROCESS, item["manuscript_id"]),
        "local_id": item["manuscript_id"],
        "prompt": item["prompt"],
        "options": item["options"],
        "output_format": output_format,
        "format_code": "choice" if output_format is None else FORMAT_CODE[output_format],
        "label": item["label"],
        "answer": answer_letter(item["answer"], item.get("options") or []),
    }


def _options_text(item: dict) -> str:
    return format_choices([str(option) for option in (item.get("options") or [])])


def _gold_phrase(item: dict) -> str:
    answer = str(item["answer"])
    options = [str(option) for option in (item.get("options") or []) if option]
    if not options:
        return answer
    _letter, text = resolve_choice(answer, options)
    return text


def _review_slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, options:{_options_text(item)}, format:{item['format_code']}, "
        f"label:{item['label']}, answer:{item['answer']}}}"
    )


def _open_slot(item: dict) -> str:
    return f"{{ID:{item['id']}, options:{_options_text(item)}, format:{item['format_code']}}}"


def _sentence_groups() -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    cursor = 0
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        if "." in PROCESS_TEXT[cursor:match.start()] and current:
            groups.append(current)
            current = []
        current.append(match.group(1))
        cursor = match.end()
    if current:
        groups.append(current)
    return groups


def _render_process(
    by_id: dict[str, dict],
    target: str | None = None,
    targets: list[str] | None = None,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
) -> str:
    if targets is None and target is not None:
        targets = next(group for group in _sentence_groups() if target in group)
    open_ids = list(targets or [])
    open_set = set(open_ids)
    last_open = open_ids[-1] if open_ids else None
    pieces: list[str] = []
    cursor = 0
    seen: list[str] = []
    seen_open = False
    for match in SLOT_MARK.finditer(PROCESS_TEXT):
        blank_id = match.group(1)
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
                period = tail.find(".")
                if period == -1:
                    raise RuntimeError(f"ID {blank_id} has no sentence period.")
                pieces.append(tail[: period + 1])
                return "".join(pieces).strip()
        elif seen_open:
            pieces.append(_open_slot(item))
        else:
            pieces.append(_gold_phrase(item) if memory is None else memory.get(item["id"], ""))
        cursor = match.end()
    pieces.append(PROCESS_TEXT[cursor:])
    if open_ids and not seen_open:
        raise KeyError(f"No sentence contains {target}.")
    if not open_ids and seen != [str(number) for number in range(1, BLANK_COUNT + 1)]:
        raise RuntimeError(f"Blank order {seen} does not match 1 through {BLANK_COUNT}.")
    return "".join(pieces).strip()


def _json_object(items: list[dict], example: bool = False) -> str:
    fields = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    choice_index = 0
    for item in items:
        if item.get("options"):
            letter = letters[choice_index % len(letters)]
            choice_index += 1
            fields.append(f"{item['id']}:{letter if example else 'letter'}")
        elif example:
            fields.append(f'{item["id"]}:"{FORMAT_EXAMPLE[item["format_code"]]}"')
        else:
            fields.append(f'{item["id"]}:"answer"')
    return "{" + ", ".join(fields) + "}"


def _injection(
    declaration: str,
    by_id: dict[str, dict],
    target: str,
    memory: dict[str, str] | None = None,
    keep_suffix: bool = False,
) -> str:
    group = next(item for item in _sentence_groups() if target in item)
    items = [by_id[blank_id] for blank_id in group]
    if len(group) == 1:
        lead = f"Fill only the current sentence. This is a step-by-step question. Output one JSON object {_json_object(items)}. Do not explain, and do not fill any other blank."
    else:
        listed = ", ".join(item["id"] for item in items)
        lead = (
            f"Fill every open slot in this sentence. The IDs are {listed}. "
            f"This is a step-by-step question. Output one JSON object {_json_object(items)}. "
            "Do not split it into several objects. A choice value is a letter, without quotes. "
            "A matrix value is a string in English double quotes."
        )
    if keep_suffix:
        lead += " Text after the period is only for reference. Do not fill those blanks."
    output = [
        "[Output]",
        "Put every answer in one JSON object. The key is the blank number, and the value is the answer.",
        "A choice value is only the letter, without quotes. A matrix must be a string in English double quotes.",
        f"Output one line {_json_object(items)}.",
        f"Format example (not the answer to this item): {_json_object(items, example=True)}",
    ]
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        declaration.strip(),
        "",
        "[Background]",
        PRIOR,
        "",
        "[Subprocess]",
        PROCESS,
        "",
        "[Process]",
        _render_process(by_id, targets=group, memory=memory, keep_suffix=keep_suffix),
        "",
        "\n".join(output),
    ])


def _render_markdown(record: dict) -> str:
    lines = [
        f"# Data {record['id']} {record['title']}",
        "",
        f"This file is a flight-mechanics review manuscript for the ground to track and airflow subprocess. Blanks of the three subprocesses are numbered continuously from 1 through {TOTAL_BLANKS}. This subprocess is {public_span(PROCESS)[0]} through {public_span(PROCESS)[1]}. One sentence is answered at a time. Blanks before this sentence are filled with the answer text. Blanks in this sentence stay open.",
        "",
        "## Declaration",
        "",
        record["declaration"].strip(),
        "",
        "## Background",
        "",
        PRIOR,
        "",
        "## Full process",
        "",
        "Each slot below carries its label and the answer, for review. These are review fields. The model does not see them on an open blank.",
        "",
        f"### {PROCESS}",
        "",
        record["process_text"],
        "",
    ]
    return "\n".join(lines)


def build_record(sample: dict) -> dict:
    model = FlightFrameModel().declare(deepcopy(sample["spec"]))
    payload = generate(deepcopy(sample["spec"]))
    questions = [_question(item) for item in payload["items"]]
    if len(questions) != BLANK_COUNT:
        raise RuntimeError(f"{sample['id']} produced {len(questions)} questions.")
    by_id = {item["local_id"]: item for item in questions}
    declaration = _declaration(model)
    _check_declaration(declaration)
    return {
        "id": sample["id"],
        "title": sample["title"],
        "change": sample["change"],
        "spec": sample["spec"],
        "declaration": declaration,
        "prior": PRIOR,
        "process": PROCESS,
        "process_text": _render_process(by_id),
        "questions": questions,
        "domain": "flight mechanics",
    }


def _check_sentences(record: dict) -> None:
    by_id = {item["local_id"]: item for item in record["questions"]}
    covered = [blank_id for sentence in _sentence_groups() for blank_id in sentence]
    if covered != [str(number) for number in range(1, BLANK_COUNT + 1)]:
        raise RuntimeError(f"Sentence groups do not cover 1 through {BLANK_COUNT}: {covered}")
    for sentence in _sentence_groups():
        kinds = [by_id[blank_id]["label"] for blank_id in sentence]
        families = {kind[0] for kind in kinds}
        if "A" in families and "C" in families:
            raise RuntimeError(f"A and C share one sentence: {list(zip(sentence, kinds))}")
        if sentence == ["20", "21"]:
            continue
        if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
            raise RuntimeError(f"An A blank is not alone in its sentence: {list(zip(sentence, kinds))}")
    opened = _injection(record["declaration"], by_id, "16").split("[Process]\n", 1)[1]
    process = opened.split("[Output]", 1)[0].rstrip()
    if not process.endswith("rotation axis.") or "{ID:48, options:" not in process or "through φ" in process:
        raise RuntimeError("ID 16 does not end at its sentence period.")
    earlier = _injection(record["declaration"], by_id, "16").split("[Process]\n", 1)[1]
    if "on the right side of Xg" not in earlier or "on the A side of Xg" in earlier:
        raise RuntimeError("The answer before the blank was not written as the option text.")
    matrix = _injection(record["declaration"], by_id, "18")
    if '{50:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}' not in matrix:
        raise RuntimeError("The matrix was not placed in JSON in double quotes.")
    shared = _injection(record["declaration"], by_id, "9")
    if "{41:letter, 42:letter, 43:letter}" not in shared:
        raise RuntimeError("The three axes of one sentence were not placed in one JSON object.")
    remembered = _injection(record["declaration"], by_id, "16", memory={"47": "MEMTEST"})
    if "MEMTEST" not in remembered:
        raise RuntimeError("An earlier blank was not replaced by the model's own output.")


def _format_document(example: str) -> str:
    return "\n".join([
        "# Flight-mechanics injection format",
        "",
        f"The domain is flight mechanics. Blanks of the three subprocesses are numbered continuously from 1 through {TOTAL_BLANKS}. The current subprocess is ground to track and airflow, numbers {public_span(PROCESS)[0]} through {public_span(PROCESS)[1]}.",
        "",
        "One call answers one sentence. The input order is the answering instruction, [Declaration], [Background], [Subprocess], [Process], [Output]. A choice value is a letter. A matrix is a string in English double quotes.",
        "",
        "An A blank stands alone in its sentence and does not share a sentence with a C blank. A: the answer text is written before the blank, and the text is cut at the period. B: the blank is replaced by the model's own output from the previous sentence. C′: the answer text is written before the blank, and the text after the period is kept. C″: the model's own output is written before the blank, and the following text is kept.",
        "",
        "## Background",
        "",
        PRIOR,
        "",
        f"- matrix: {MATRIX_FORMAT} Example: {FORMAT_EXAMPLE['matrix']}",
        "",
        "## One real injection",
        "",
        "Below is the full input for data 01, ground to track and airflow, ID 16.",
        "",
        "```text",
        example,
        "```",
        "",
    ])


def _index(samples: list[dict]) -> str:
    lines = [
        "# Flight-mechanics review samples",
        "",
        f"Subprocess: ground to track and airflow. Each sample has {BLANK_COUNT} blanks, public numbers {public_span(PROCESS)[0]} through {public_span(PROCESS)[1]}. The data are in data/flight/basic/ground_to_track_air/review.",
        "",
        "| id | title | note |",
        "| --- | --- | --- |",
    ]
    for sample in samples:
        lines.append(f"| {sample['id']} | {sample['title']} | {sample['change']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    records = [build_record(sample) for sample in samples]
    _check_sentences(records[0])
    for sample, record in zip(samples, records):
        (OUT_DIR / f"sample_{sample['id']}.md").write_text(_render_markdown(record), encoding="utf-8")
    (OUT_DIR / "samples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    example = _injection(
        records[0]["declaration"],
        {item["local_id"]: item for item in records[0]["questions"]},
        "16",
    )
    (OUT_DIR / "00_injection_format.md").write_text(_format_document(example), encoding="utf-8")
    (OUT_DIR / "00_index.md").write_text(_index(samples), encoding="utf-8")
    print(f"Wrote {len(records)} aerodynamics review samples, {BLANK_COUNT} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
