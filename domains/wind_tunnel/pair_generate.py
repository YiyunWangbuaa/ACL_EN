# -*- coding: utf-8 -*-
"""Single-factor contract pairs.

Five abnormal definitions, each written once: the declaration states the
contract, and every definition's rule question stays on the page. Two
standards use the same full question set. Manuscripts that drop those
questions, or that add an extra warning in front of a changed rule, are not
generated.

The spherical-wrist order is the standard side of the order pair. The current
angle-rod description is the abnormal side. The other four abnormal
definitions are paired with the unified default.

Run::

    python -m domains.wind_tunnel.pair_generate
"""

from __future__ import annotations

import json
from copy import deepcopy

from body_frame_model import BodyFrameModel
from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    PRIOR,
    PROCESSES,
    SLOT_MARK,
    _check_declaration,
    _declaration,
    _mechanism_phrases,
    _process_template,
    _question,
    _render_markdown,
    _render_process,
    _sentence_groups,
    build_record,
)
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC


OUT_DIR = DOMAINS["wind_tunnel"]["pair"]
TZ = "wind frame to body-axis frame"
_CACHE: dict[str, dict[str, list[dict]]] = {}
RULE_IDS = (
    "standard-default",
    "standard-order",
    "alpha-rule",
    "roll-rule",
    "order-rule",
    "mount-rule",
    "body-rule",
)


def _alpha(spec: dict) -> None:
    spec["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}


def _roll(spec: dict) -> None:
    spec["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}


def _mount(spec: dict) -> None:
    spec["placement"] = "inverted"


def _body(spec: dict) -> None:
    spec["body_frame"] = {
        "chirality": "left",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    }


def _wrist(spec: dict) -> None:
    frame = deepcopy(spec["body_frame"])
    frame["euler_order"] = ["beta", "alpha", "gamma"]
    spec["body_frame"] = frame


FACTORS = (
    {
        "key": "alpha",
        "title": "theoretical angle of attack sign",
        "change": "Only the theoretical angle of attack is changed: it is positive when the airspeed projection lies upward on the airframe axis. The mechanism angle of attack, the axes, the installation, and the rotation order stay put.",
        "apply": _alpha,
        "standard_id": "standard-default",
    },
    {
        "key": "roll",
        "title": "mechanism roll direction",
        "change": "Only the mechanism roll γ_M is changed: it is opposite the right-hand rule about the positive absolute model axis (forward). The theoretical roll stays the same as the right-hand rule.",
        "apply": _roll,
        "standard_id": "standard-default",
    },
    {
        "key": "order",
        "title": "mechanism-angle order",
        "change": "The abnormal side keeps the current angle-rod mechanism: the order is fixed by the sting layout as angle of attack, sideslip, roll. The standard side is changed to a three-degree-of-freedom moving-axis spherical wrist, and the order is yaw, pitch, roll.",
        "apply": None,
        "standard_id": "standard-order",
    },
    {
        "key": "mount",
        "title": "upright and inverted",
        "change": "Only the model installation is changed to inverted. Frame chirality, angle positive directions, and the rotation order stay put.",
        "apply": _mount,
        "standard_id": "standard-default",
    },
    {
        "key": "body",
        "title": "body-axis chirality",
        "change": "Only the body-axis frame is changed to left-handed. XTZ still points forward along the airframe axial axis, YTZ still points upward along the airframe normal axis, and ZTZ is inferred from the chirality.",
        "apply": _body,
        "standard_id": "standard-default",
    },
)


def _payloads(spec: dict) -> dict[str, list[dict]]:
    key = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    if key not in _CACHE:
        found = {}
        for process, generate in PROCESSES:
            found[process] = generate(deepcopy(spec))["items"]
        _CACHE[key] = found
    return _CACHE[key]


def _raw(spec: dict, process: str, local_id: str) -> str:
    for item in _payloads(spec)[process]:
        if item["manuscript_id"] == local_id:
            return item["answer"]
    raise KeyError(f"{process} has no manuscript id {local_id}")


def _public_template(process: str, template: str) -> str:
    from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET

    offset = PROCESS_OFFSET[process]
    return SLOT_MARK.sub(lambda match: f"@@{int(match.group(1)) + offset}@@", template)


def _check_sentences(process: str, template: str, phrases: dict[str, str], by_local: dict[str, dict]) -> None:
    for sentence in _sentence_groups(process, phrases, template):
        kinds = [by_local[local_id]["label"] for local_id in sentence]
        families = {kind[0] for kind in kinds}
        if "A" in families and "C" in families:
            raise RuntimeError(f"{process} puts A and C in the same sentence: {list(zip(sentence, kinds))}")
        if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
            raise RuntimeError(f"An A blank of {process} is not in a sentence by itself: {list(zip(sentence, kinds))}")


def _note(title: str, condition: str) -> str:
    if condition == "standard":
        return (
            f"Review note: {title}. This sentence and the file title are not model input. "
            "The model sees only the declaration, the background, and the process below. This row is the standard side of a pair, and every rule question is kept."
        )
    if condition != "rule":
        raise RuntimeError(f"Only a complete normal-contract manuscript is generated, so {condition} cannot be written.")
    return (
        f"Review note: {title}, condition ask-the-rule-first. This sentence and the file title are not model input. "
        "The model sees only the declaration, the background, and the process below. This row keeps every rule question, and the declaration has no extra warning."
    )


def _make(
    spec: dict,
    sample_id: str,
    title: str,
    change: str,
    factor: str,
    condition: str,
    standard_id: str,
) -> dict:
    model = BodyFrameModel().declare(deepcopy(spec))
    phrases = _mechanism_phrases(model)
    declaration = _declaration(model)
    if "This item specially stipulates" in declaration:
        raise RuntimeError(f"The declaration of {sample_id} carries an extra warning.")
    _check_declaration(declaration)
    questions = []
    processes = {}
    process_text = {}
    for process, _generate in PROCESSES:
        template = _process_template(model, process)
        items = [_question(process, item) for item in _payloads(spec)[process]]
        slots = SLOT_MARK.findall(template)
        if sorted((item["local_id"] for item in items), key=int) != sorted(slots, key=int):
            raise RuntimeError(f"The questions of {sample_id} {process} do not match the template slots.")
        by_local = {item["local_id"]: item for item in items}
        _check_sentences(process, template, phrases, by_local)
        processes[process] = _render_process(process, by_local, phrases=phrases, template=template)
        process_text[process] = _public_template(process, template)
        questions.extend(items)
    record = {
        "id": sample_id,
        "title": title,
        "factor": factor,
        "condition": condition,
        "paired_standard_id": standard_id,
        "change": change,
        "spec": spec,
        "declaration": declaration,
        "prior": PRIOR,
        "process_text": process_text,
        "questions": questions,
        "excluded_ids": [],
        "tier": 0,
        "group": "pair",
    }
    record["processes"] = processes
    record["review_note"] = _note(title, condition)
    return record


def _answers(record: dict) -> dict[str, str]:
    return {item["id"]: item["answer"] for item in record["questions"]}


def _diff_ids(left: dict, right: dict) -> list[str]:
    shared = set(left) & set(right)
    return sorted((item_id for item_id in shared if left[item_id] != right[item_id]), key=int)


def _spec_keys(left: dict, right: dict) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _check_physics(default: dict, specs: dict[str, dict]) -> None:
    def raw(name: str, process: str, local_id: str) -> str:
        return _raw(specs[name], process, local_id)

    if raw("alpha", TZ, "20") != "downward" or raw("default", TZ, "20") != "downward":
        raise RuntimeError("The theoretical-angle-of-attack pair changed the airspeed direction under the mechanism angle of attack.")
    if raw("alpha", TZ, "21") != "upward" or raw("alpha", TZ, "22") != "negative":
        raise RuntimeError("The sign after the abnormal theoretical angle of attack is not negative.")
    if raw("default", TZ, "22") != "positive":
        raise RuntimeError("The default theoretical angle-of-attack sign is not positive.")
    if raw("alpha", TZ, "26") != raw("default", TZ, "26"):
        raise RuntimeError("The theoretical-angle-of-attack pair changed the mechanism angle-of-attack matrix.")
    if not raw("alpha", TZ, "80").startswith("-arctan"):
        raise RuntimeError("After the abnormal theoretical angle of attack, α_t does not put the minus outside.")

    if raw("roll", TZ, "35") != "opposite" or raw("roll", TZ, "36") != "upward":
        raise RuntimeError("The abnormal mechanism roll was not changed to opposite the right-hand rule with the right wing upward.")
    if raw("roll", TZ, "37") != "same" or raw("default", TZ, "37") != "same":
        raise RuntimeError("The mechanism-roll pair changed the theoretical roll.")
    if raw("roll", TZ, "39") != "negative":
        raise RuntimeError("When γ_M>0 the sign of the theoretical roll did not become negative.")
    if raw("roll", TZ, "43") == raw("default", TZ, "43"):
        raise RuntimeError("The abnormal mechanism roll did not change the roll matrix.")

    if raw("mount", TZ, "11") != "inverted" or raw("mount", TZ, "12") != "opposite":
        raise RuntimeError("The inverted pair was not written as inverted, with normal and lateral opposite.")
    if raw("mount", TZ, "13") != "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]":
        raise RuntimeError(f"Inverted RP2 is not the expected matrix: {raw('mount', TZ, '13')}")
    if raw("default", TZ, "13") != "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]":
        raise RuntimeError("The default RP2 is not the identity matrix.")

    if raw("body", TZ, "5") != "YTZ" or raw("body", TZ, "6") != "XTZ" or raw("body", TZ, "7") != "-ZTZ":
        raise RuntimeError("The left-handed body frame changed the declared X or Y, or Z was not reversed.")
    if raw("default", TZ, "7") != "ZTZ":
        raise RuntimeError("The default body-axis lateral axis is not ZTZ.")
    if raw("body", TZ, "8") == raw("default", TZ, "8"):
        raise RuntimeError("The left-handed body frame did not change RP1.")

    if [raw("wrist", TZ, "16"), raw("wrist", TZ, "17"), raw("wrist", TZ, "18")] != ["sideslip", "angle of attack", "roll"]:
        raise RuntimeError("The wrist order is not yaw, pitch, roll.")
    if [raw("default", TZ, "16"), raw("default", TZ, "17"), raw("default", TZ, "18")] != ["angle of attack", "sideslip", "roll"]:
        raise RuntimeError("The angle-rod order is not angle of attack, sideslip, roll.")
    if [raw("wrist", TZ, "47"), raw("wrist", TZ, "48"), raw("wrist", TZ, "49")] != ["R_γ_M", "R_α_M", "R_β_M"]:
        raise RuntimeError("The wrist R1 R2 R3 is not roll, angle of attack, sideslip.")
    if [raw("default", TZ, "47"), raw("default", TZ, "48"), raw("default", TZ, "49")] != ["R_γ_M", "R_β_M", "R_α_M"]:
        raise RuntimeError("The angle-rod R1 R2 R3 is not roll, sideslip, angle of attack.")
    if raw("default", TZ, "23") != "intermediate body-axis frame" or raw("default", TZ, "24") != "ZTZ":
        raise RuntimeError("The default angle of attack is not written on the lateral axis of the intermediate body-axis frame.")
    if raw("mount", TZ, "23") != "intermediate body-axis frame" or raw("mount", TZ, "24") != "ZTZ":
        raise RuntimeError("Inverted installation moved the attitude axis out of the intermediate body-axis frame.")
    if raw("body", TZ, "34") == raw("default", TZ, "34") or raw("body", TZ, "43") == raw("default", TZ, "43"):
        raise RuntimeError("The left-handed body frame did not rewrite the attitude matrix in the intermediate body-axis frame.")
    if raw("wrist", TZ, "55") == raw("default", TZ, "55"):
        raise RuntimeError("The two mechanism orders have the same total matrix, so they cannot be a pair.")
    if raw("wrist", TZ, "11") != "upright" or raw("wrist", TZ, "22") != "positive" or raw("wrist", TZ, "35") != "same":
        raise RuntimeError("The wrist order changed the installation, the angle-of-attack sign, or the roll direction.")


def _check_factor(factor: dict, rule: dict, standard: dict) -> list[str]:
    if rule["excluded_ids"] or standard["excluded_ids"]:
        raise RuntimeError(f"{factor['key']} dropped a rule question.")
    if "This item specially stipulates" in rule["declaration"] or "This item specially stipulates" in standard["declaration"]:
        raise RuntimeError(f"The declaration of {factor['key']} carries an extra warning.")
    rule_ids = {item["id"] for item in rule["questions"]}
    standard_ids = {item["id"] for item in standard["questions"]}
    if rule_ids != standard_ids:
        raise RuntimeError(f"The probes of {factor['key']} do not match the standard manuscript.")
    diff = _diff_ids(_answers(rule), _answers(standard))
    if not diff:
        raise RuntimeError(f"The answers of {factor['key']} are identical to the standard.")
    changed = _spec_keys(rule["spec"], standard["spec"])
    if factor["key"] == "order":
        if changed != ["body_frame"]:
            raise RuntimeError(f"The order pair changed {changed}.")
        left = deepcopy(rule["spec"]["body_frame"])
        right = deepcopy(standard["spec"]["body_frame"])
        if left.pop("euler_order") != ["alpha", "beta", "gamma"] or right.pop("euler_order") != ["beta", "alpha", "gamma"]:
            raise RuntimeError("The Euler order of the order pair is not the angle rod against the wrist.")
        if left != right:
            raise RuntimeError("The order pair also changed a body-axis coordinate axis.")
    elif factor["key"] == "body":
        if changed != ["body_frame"]:
            raise RuntimeError(f"The body-axis pair changed {changed}.")
        if rule["spec"]["body_frame"]["chirality"] != "left":
            raise RuntimeError("The body-axis pair was not changed to left-handed.")
        if rule["spec"]["body_frame"]["XTZ"] != standard["spec"]["body_frame"]["XTZ"]:
            raise RuntimeError("The body-axis pair changed XTZ.")
        if rule["spec"]["body_frame"]["YTZ"] != standard["spec"]["body_frame"]["YTZ"]:
            raise RuntimeError("The body-axis pair changed YTZ.")
    else:
        expected = {"alpha": ["alpha"], "roll": ["gamma_m"], "mount": ["placement"]}[factor["key"]]
        if changed != expected:
            raise RuntimeError(f"{factor['key']} changed {changed}; it should be {expected}.")
    return diff


def _index(rows: list[dict], diffs: dict[str, list[str]]) -> str:
    by_id = {row["id"]: row for row in rows}
    lines = [
        "# Single-factor engineering-contract pairs",
        "",
        "Five abnormal definitions, each with one manuscript whose probes are complete: the definition is in the declaration, every rule question is kept, and there is no extra warning.",
        "The standard side also keeps every rule question. The common default is standard-default. The wrist order is standard-order.",
        "The standard side of the theoretical angle of attack, mechanism roll, upright or inverted, and body-axis chirality is the common default.",
        "The standard side of the mechanism-angle order is a three-degree-of-freedom moving-axis spherical wrist, executed as yaw, pitch, roll. The abnormal side keeps the current angle-rod mechanism.",
        "",
        "Model input must not include the file title or the review note. Evaluation uses the declaration and process_text in samples.jsonl.",
        "",
        "| Pair | Standard | Manuscript | Questions | Blanks different from the standard |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for factor in FACTORS:
        rule = by_id[f"{factor['key']}-rule"]
        lines.append(
            "| {title} | {standard} | {key}-rule | {count} | {diff} |".format(
                title=factor["title"],
                standard=factor["standard_id"],
                key=factor["key"],
                count=len(rule["questions"]),
                diff=len(diffs[factor["key"]]),
            )
        )
    lines.extend(["", "## Blanks whose answers differ from the standard", ""])
    for factor in FACTORS:
        lines.append(f"### {factor['title']}")
        lines.append("")
        lines.append(factor["change"])
        lines.append("")
        lines.append("Different from the standard: " + ", ".join(diffs[factor["key"]]))
        lines.append("")
    return "\n".join(lines)


def _markdown(record: dict) -> str:
    text = _render_markdown(record)
    return text.replace("\n\n", "\n\n" + record["review_note"] + "\n\n", 1)


def _check_standards(default_row: dict, wrist_row: dict) -> None:
    default_spec = deepcopy(STANDARD_SPEC)
    built = build_record({
        "id": "01",
        "title": "common default",
        "change": "Every definition stays at the common default.",
        "spec": default_spec,
    })
    if built["declaration"] != default_row["declaration"] or built["processes"] != default_row["processes"]:
        raise RuntimeError("The common-default pair manuscript does not match the current generator.")
    wrist_spec = deepcopy(STANDARD_SPEC)
    _wrist(wrist_spec)
    wrist_built = build_record({
        "id": "wrist",
        "title": "usual mechanism-angle order",
        "change": "Three-degree-of-freedom moving-axis spherical wrist.",
        "spec": wrist_spec,
    })
    if wrist_built["declaration"] != wrist_row["declaration"] or wrist_built["processes"] != wrist_row["processes"]:
        raise RuntimeError("The wrist standard manuscript does not match the generator.")
    if "three-degree-of-freedom moving-axis spherical wrist" not in wrist_row["declaration"] or "yaw, pitch, roll" not in wrist_row["declaration"]:
        raise RuntimeError("The wrist declaration does not state the order.")
    if "pitch sting" in wrist_row["declaration"]:
        raise RuntimeError("The wrist declaration still uses the pitch sting.")
    if "This item executes the mechanism attitude angles in the order yaw, pitch, roll." not in wrist_row["processes"][TZ]:
        raise RuntimeError("The wrist process still writes the order as inferred from the mechanism layout.")
    if "From the mechanism layout, the first rotation angle is" in wrist_row["processes"][TZ]:
        raise RuntimeError("The wrist process still keeps the inference sentence.")
    if "spherical wrist" in default_row["declaration"] or "pitch sting" not in default_row["declaration"]:
        raise RuntimeError("The common-default mechanism description was replaced by the wrist.")


def main() -> None:
    default_spec = deepcopy(STANDARD_SPEC)
    wrist_spec = deepcopy(STANDARD_SPEC)
    _wrist(wrist_spec)
    specs = {
        "default": default_spec,
        "wrist": wrist_spec,
        "alpha": deepcopy(STANDARD_SPEC),
        "roll": deepcopy(STANDARD_SPEC),
        "mount": deepcopy(STANDARD_SPEC),
        "body": deepcopy(STANDARD_SPEC),
        "order": deepcopy(STANDARD_SPEC),
    }
    _alpha(specs["alpha"])
    _roll(specs["roll"])
    _mount(specs["mount"])
    _body(specs["body"])
    _check_physics(default_spec, specs)

    standards = {
        "standard-default": _make(
            default_spec, "standard-default", "common default",
            "Every definition stays at the common default. The theoretical angle of attack, mechanism roll, upright or inverted, and body-axis chirality use this as the standard side.",
            "default", "standard", "standard-default",
        ),
        "standard-order": _make(
            wrist_spec, "standard-order", "usual mechanism-angle order",
            "The angle mechanism is changed to a three-degree-of-freedom moving-axis spherical wrist, and the execution order is yaw, pitch, roll. The other definitions stay at the common default.",
            "order", "standard", "standard-order",
        ),
    }
    _check_standards(standards["standard-default"], standards["standard-order"])

    rows = [standards["standard-default"], standards["standard-order"]]
    diffs = {}
    for factor in FACTORS:
        spec = specs[factor["key"]]
        standard = standards[factor["standard_id"]]
        rule = _make(
            spec,
            f"{factor['key']}-rule",
            f"{factor['title']} ask the rule first",
            factor["change"],
            factor["key"],
            "rule",
            factor["standard_id"],
        )
        if factor["key"] != "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError(f"The manuscript of {factor['key']} rewrote a process sentence that is not about the target definition.")
        if factor["key"] == "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError("The abnormal-order manuscript did not keep the current mechanism description.")
        diffs[factor["key"]] = _check_factor(factor, rule, standard)
        rows.append(rule)

    if [row["id"] for row in rows] != list(RULE_IDS):
        raise RuntimeError(f"The manuscripts should be {RULE_IDS}, but they are {[row['id'] for row in rows]}.")
    if any(row["condition"] not in {"standard", "rule"} for row in rows):
        raise RuntimeError("The result contains a manuscript that is not a rule manuscript.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {f"{row['id']}.md" for row in rows} | {"00_index.md"}
    for path in OUT_DIR.glob("*.md"):
        if path.name not in keep:
            path.unlink()
    stored = []
    for row in rows:
        (OUT_DIR / f"{row['id']}.md").write_text(_markdown(row), encoding="utf-8")
        stored.append({key: value for key, value in row.items() if key not in {"processes", "review_note"}})
        print(f"{row['id']}: {len(row['questions'])} questions", flush=True)
    (OUT_DIR / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stored),
        encoding="utf-8",
    )
    (OUT_DIR / "00_index.md").write_text(_index(rows, diffs) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} manuscripts to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
