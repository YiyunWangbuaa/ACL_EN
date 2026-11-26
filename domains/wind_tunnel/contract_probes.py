# -*- coding: utf-8 -*-
"""Side-channel contract probes for the wind-tunnel cloze.

A probe asks one local contract question on a frozen prefix. The answer is
recorded and is never written into a later sentence or into the target call.
The probe does not ask for the target matrix or formula. The target execution
stays the answer already produced by the original run.

Current public ids, not the schematic numbers in the design note:
    roll direction ID87 then matrix ID88; left-handed cells before ID34 and ID43;
    moment lever ID184, MCN_CZ sign ID192, formula ID198.
"""
from __future__ import annotations

import hashlib
import re

from cloze_score import normalize, sympy_equal
from domains.wind_tunnel.dataset_generate import PRIOR, SLOT_MARK, _period_index, _sentence_groups

# These items are not primary rollback probes. A two-way sign collides with the
# other definition, ID80 is an ordinary formula miss, and ID104 can be produced
# by two inverted matrices cancelling.
EXCLUDED_TARGETS = frozenset({"22", "68", "80", "104"})

PROBES = (
    {
        "probe_id": "roll_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "mechanism roll",
        "group": "roll",
        "sample_id": "roll-rule",
        "process": "balance frame to body-axis frame",
        "after": "87",
        "target": "88",
        "kind": "cells",
        "cells": ((2, 3), (3, 2)),
        "repair_parent": ("85", "86"),
        "repair_other": ("87",),
        "question": (
            "From the roll direction already fixed above, when this rotation matrix is written, "
            "what sign should the sin term in row 2 column 3 and the sin term in row 3 column 2 take? "
            "Answer as (2,3)=positive or negative, (3,2)=positive or negative. Do not write the full matrix."
        ),
        "hidden_stem": "R_(γ_M-γ_tp) is",
    },
    {
        "probe_id": "beta_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "left-handed frame",
        "group": "body",
        "sample_id": "body-rule",
        "process": "wind frame to body-axis frame",
        "after": "33",
        "target": "34",
        "kind": "cells",
        "cells": ((1, 3), (3, 1)),
        "repair_parent": ("32", "33"),
        "repair_other": ("31",),
        "question": (
            "From the intermediate body-axis frame and the rotation direction already fixed above, "
            "what sign should sin(β_M) take in row 1 column 3 and in row 3 column 1 of R_β_M? "
            "Answer as (1,3)=positive or negative, (3,1)=positive or negative. Do not write the full matrix."
        ),
        "hidden_stem": "R_β_M is",
    },
    {
        "probe_id": "gamma_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "left-handed frame",
        "group": "body",
        "sample_id": "body-rule",
        "process": "wind frame to body-axis frame",
        "after": "42",
        "target": "43",
        "kind": "cells",
        "cells": ((2, 3), (3, 2)),
        "repair_parent": ("41", "42"),
        "repair_other": ("40",),
        "question": (
            "From the intermediate body-axis frame and the rotation direction already fixed above, "
            "what sign should sin(γ_M) take in row 2 column 3 and in row 3 column 2 of R_γ_M? "
            "Answer as (2,3)=positive or negative, (3,2)=positive or negative. Do not write the full matrix."
        ),
        "hidden_stem": "R_γ_M is",
    },
    {
        "probe_id": "mcn_cz_sign",
        "node_role": "probe",
        "new_label": "A-SPATIAL",
        "subtype": "moment_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "moment formula",
        "group": "body",
        "sample_id": "body-rule",
        "process": "moment reference point",
        "after": "191",
        "target": "198",
        "kind": "sign",
        "source_id": "192",
        "repair_parent": ("184",),
        "repair_other": ("190",),
        "question": (
            "From the body-axis frame and the moment arm already fixed above, "
            "should the whole sign of the equivalent moment coefficient MCN_CZ be positive or negative? "
            "Answer only positive or negative. Do not write the full formula."
        ),
        "hidden_stem": "The sign of MCN_CZ is",
    },
    {
        "probe_id": "mcn_cz_place",
        "node_role": "probe",
        "new_label": "A-MATH-FORMAT",
        "subtype": "sign_placement",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "moment formula",
        "group": "body",
        "sample_id": "body-rule",
        "process": "moment reference point",
        "after": "188",
        "target": "198",
        "kind": "placement",
        "repair_parent": ("184",),
        "repair_other": ("185",),
        "question": (
            "From the moment arm already fixed above, if MCN_CZ needs a minus sign, "
            "should that minus sign be written outside the whole expression, or inside the moment arm? "
            "Answer only outside or inside. Do not write the full formula."
        ),
        "hidden_stem": "the arm itself must be positive",
    },
)

CONDITIONS = (
    "deep_b",
    "deep_gold",
    "replay",
    "matched",
    "repair_parent",
    "repair_other",
)
DEEP_CONDITIONS = frozenset({"deep_b", "deep_gold", "replay", "matched", "repair_parent", "repair_other"})

# ID58 only checks whether an earlier chirality choice reaches the component.
CONSEQUENCE = {
    "group": "body",
    "sample_id": "body-rule",
    "target": "58",
    "role": "combined consequence node",
}


def _prepared(chain: dict) -> str:
    template = chain["process_text"]
    for token, text in (chain.get("phrases") or {}).items():
        template = template.replace(token, text)
    return template


def _groups(chain: dict) -> list[list[str]]:
    return _sentence_groups(chain["process"], chain.get("phrases"), chain.get("process_text"))


def filled_prefix(chain: dict, through_id: str, memory: dict[str, str]) -> str:
    """Process text through the sentence that ends at through_id, with blanks filled."""
    through = str(through_id)
    groups = _groups(chain)
    index = next((i for i, group in enumerate(groups) if through in group), None)
    if index is None:
        raise KeyError(f"{chain['process']} has no blank {through}")
    if groups[index][-1] != through:
        raise RuntimeError(
            f"Probe anchor {through} is not the last blank of its sentence; the sentence is {groups[index]}. "
            "Cutting mid-sentence would leave later blanks of that sentence in the prefix."
        )
    filled = [blank_id for group in groups[: index + 1] for blank_id in group]
    missing = [blank_id for blank_id in filled if blank_id not in memory]
    if missing:
        raise KeyError(f"The prefix is missing answers for these blanks: {missing}")
    template = _prepared(chain)
    pieces: list[str] = []
    cursor = 0
    for match in SLOT_MARK.finditer(template):
        blank_id = match.group(1)
        pieces.append(template[cursor:match.start()])
        if blank_id not in set(filled):
            raise RuntimeError(f"The probe prefix reached blank {blank_id} after anchor {through}")
        pieces.append(memory[blank_id])
        cursor = match.end()
        if blank_id == through:
            tail = template[match.end():]
            period = _period_index(tail)
            if period < 0:
                raise RuntimeError(f"The sentence of {through} in {chain['process']} has no period.")
            if SLOT_MARK.search(tail[:period]):
                raise RuntimeError(f"The sentence of {through} still has an unfilled blank.")
            pieces.append(tail[: period + 1])
            return "".join(pieces).strip()
    raise RuntimeError(f"Did not cut {through} out of the problem text")


def _question(probe: dict, condition: str, phrases: dict[str, str]) -> str:
    text = probe["question"]
    if "{" in text or "}" in text:
        raise RuntimeError(f"The stem of {probe['probe_id']} cannot be filled with a process answer.")
    if phrases:
        return text
    return text


def build_target_prompt(chain: dict, probe: dict, memory: dict[str, str]) -> str:
    """Ask the real target sentence on a repaired prefix, then stop.

    Earlier blanks use the branch memory. The target sentence stays open.
    Text after that sentence is omitted, and this answer is not a probe.
    """
    from domains.wind_tunnel.dataset_generate import _injection

    target = str(probe["target"])
    groups = _groups(chain)
    index = next((i for i, group in enumerate(groups) if target in group), None)
    if index is None:
        raise KeyError(f"{chain['process']} has no target blank {target}")
    needed = [blank_id for group in groups[:index] for blank_id in group]
    missing = [blank_id for blank_id in needed if blank_id not in memory]
    if missing:
        raise KeyError(f"The target branch of {probe['probe_id']} is missing answers for these blanks: {missing}")
    prompt = _injection(
        chain["declaration"],
        chain["process"],
        chain["by_id"],
        target,
        phrases=chain.get("phrases"),
        memory=memory,
        keep_suffix=False,
        template=chain.get("process_text"),
    )
    if f"{{ID:{target}," not in prompt:
        raise RuntimeError(f"The target branch of {probe['probe_id']} did not leave the open slot for ID{target}.")
    if "[Probe]" in prompt:
        raise RuntimeError(f"The target branch of {probe['probe_id']} mixed in a probe question.")
    process = prompt.split("[Process]", 1)[-1].split("[Output]", 1)[0]
    if index + 1 < len(groups) and f"{{ID:{groups[index + 1][0]}," in process:
        raise RuntimeError(f"The target branch of {probe['probe_id']} did not stop at the target sentence.")
    return prompt


def build_prompt(chain: dict, probe: dict, condition: str, memory: dict[str, str], phrases: dict[str, str] | None = None) -> str:
    """One isolated probe call. The returned text is the whole model input."""
    if probe["target"] in EXCLUDED_TARGETS:
        raise RuntimeError(f"ID{probe['target']} is not a target of this contract-probe set.")
    anchor = probe["after"]
    prefix = filled_prefix(chain, anchor, memory)
    question = _question(probe, condition, phrases or {})
    if question in prefix:
        raise RuntimeError(f"The probe stem of {probe['probe_id']} was written into the process prefix.")
    hidden = probe.get("hidden_stem") or ""
    if hidden and hidden in prefix:
        raise RuntimeError(f"The prefix of {probe['probe_id']} contains a stem that should stay hidden: {hidden}")
    prior = chain.get("prior") or PRIOR
    return "\n".join([
        "This is a side probe. Answer only the question below.",
        "Do not fill blanks in the process and do not continue into later sentences.",
        "This answer is only a record and is not written back into the original flow.",
        'Write only one JSON object {"probe":"answer"}.',
        "",
        "[Declaration]",
        chain["declaration"],
        "",
        "[Background]",
        prior,
        "",
        "[Subprocess]",
        chain["process"],
        "",
        "[Process]",
        prefix,
        "",
        "[Probe]",
        question,
        "",
        "[Output]",
        '{"probe":"answer"}',
    ])


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _cell_sign(matrix_text: str, row: int, col: int) -> str:
    import sympy

    from cloze_score import _parse_expr, _sympy_modules

    _sympy, locals_map = _sympy_modules()
    parsed = _parse_expr(matrix_text, sympy, locals_map)
    expr = sympy.simplify(parsed[row - 1, col - 1])
    sins = list(expr.atoms(sympy.sin))
    if len(sins) != 1:
        raise RuntimeError(f"Row {row} column {col} of the matrix is not a single sin term: {expr}")
    coeff = sympy.simplify(expr / sins[0])
    if coeff == 1:
        return "positive"
    if coeff == -1:
        return "negative"
    raise RuntimeError(f"The coefficient of the sin term is not ±1: {expr}")


def _parsed_expr(text: str):
    import sympy

    from cloze_score import _parse_expr, _sympy_modules

    _sympy, locals_map = _sympy_modules()
    return sympy.simplify(_parse_expr(text, sympy, locals_map))


def _entry_sign(matrix_text: str, row: int, col: int) -> str:
    expr = _parsed_expr(matrix_text)[row - 1, col - 1]
    if expr == 1:
        return "positive"
    if expr == -1:
        return "negative"
    raise RuntimeError(f"Row {row} column {col} of the matrix is not ±1: {expr}")


def _leading_sign(text: str) -> str:
    expr = _parsed_expr(text)
    return "negative" if expr.could_extract_minus_sign() else "positive"


def contract_key(probe: dict, gold_text: str) -> dict:
    """The local contract value a probe is scored against. It comes from the manuscript."""
    if probe["kind"] == "direction":
        return {"text": gold_text, "label": _direction_label(gold_text)}
    if probe["kind"] == "cells":
        signs = {f"{row},{col}": _cell_sign(gold_text, row, col) for row, col in probe["cells"]}
        return {"text": gold_text, "signs": signs}
    if probe["kind"] == "entry":
        row, col = probe["entry"]
        return {"text": gold_text, "label": _entry_sign(gold_text, row, col)}
    if probe["kind"] == "leading":
        return {"text": gold_text, "label": _leading_sign(gold_text)}
    if probe["kind"] == "sign":
        folded = normalize(gold_text)
        label = "negative" if "negative" in folded else "positive" if "positive" in folded else ""
        return {"text": gold_text, "label": label}
    if probe["kind"] == "placement":
        return {"text": "outside", "label": "outside"}
    if probe["kind"] == "expression":
        return {"text": gold_text}
    raise KeyError(probe["kind"])


def recognition_text(probe: dict, phrase_for) -> str:
    """Text the probe is judged against. Cell probes use the target matrix."""
    if probe["kind"] == "placement":
        return "outside"
    if probe["kind"] in {"sign", "direction"}:
        return phrase_for(probe["source_id"])
    return phrase_for(probe["target"])


def _direction_label(text: str) -> str:
    raw = normalize(text)
    if any(token in raw for token in ("reverse", "negative", "opposite")):
        return "reverse"
    if "positive" in raw or raw in {"+", "positive"}:
        return "positive"
    return ""


def parse_probe(probe: dict, response: str) -> dict:
    body = response.strip()
    match = re.search(r'"probe"\s*:\s*"(.*?)"', body, flags=re.S)
    if match:
        body = match.group(1).strip()
    if probe["kind"] == "direction":
        label = _direction_label(body)
        return {"text": body, "label": label, "parsed": bool(label)}
    if probe["kind"] == "cells":
        signs = {}
        for row, col in probe["cells"]:
            found = re.search(
                rf"(?:\({row}\s*,\s*{col}\)|row {row} column {col})"
                r"\s*[=:]?\s*(negative|positive|\+|−|-)",
                body,
            )
            if not found:
                signs[f"{row},{col}"] = ""
                continue
            token = found.group(1)
            signs[f"{row},{col}"] = "negative" if token in {"negative", "-", "−", "reverse"} else "positive"
        if not any(signs.values()):
            parts = re.findall(r"negative|positive", body)
            if len(parts) == len(probe["cells"]):
                for (row, col), token in zip(probe["cells"], parts):
                    signs[f"{row},{col}"] = "negative" if "negative" in token else "positive"
        return {"text": body, "signs": signs, "parsed": all(signs.values())}
    if probe["kind"] in {"sign", "entry", "leading"}:
        negative = "negative" in body
        positive = "positive" in body
        label = "negative" if negative and not positive else "positive" if positive and not negative else ""
        return {"text": body, "label": label, "parsed": bool(label)}
    if probe["kind"] == "placement":
        outside = "outside" in body
        inside = "inside" in body
        label = "outside" if outside and not inside else "inside" if inside and not outside else ""
        return {"text": body, "label": label, "parsed": bool(label)}
    return {"text": body, "parsed": bool(body)}


def same_contract(probe: dict, parsed: dict, contract: dict) -> bool:
    if not parsed.get("parsed"):
        return False
    if probe["kind"] in {"direction", "sign", "entry", "leading", "placement"}:
        return parsed.get("label") == contract.get("label") and bool(parsed.get("label"))
    if probe["kind"] == "cells":
        return parsed.get("signs") == contract.get("signs")
    return sympy_equal(parsed.get("text") or "", contract.get("text") or "") or normalize(parsed.get("text") or "") == normalize(contract.get("text") or "")


def execution_class(prediction: str, contract_text: str, standard_text: str) -> str:
    """contract, standard, both, or neither. Algebraic equality counts."""
    hit_contract = sympy_equal(prediction, contract_text) or normalize(prediction) == normalize(contract_text)
    hit_standard = sympy_equal(prediction, standard_text) or normalize(prediction) == normalize(standard_text)
    if hit_contract and hit_standard:
        return "both"
    if hit_contract:
        return "contract"
    if hit_standard:
        return "standard"
    return "neither"


def classify(facts: dict) -> list[str]:
    """Labels from the side-probe protocol. A row may carry more than one."""
    labels = []
    deep_b = facts["deep_b_hit"]
    deep_gold = facts["deep_gold_hit"]
    target_b = facts["target_b"]
    target_gold = facts["target_gold_run"]
    deep_b_standard = facts["deep_b_standard"]
    if deep_b and target_b in {"contract", "both"}:
        labels.append("contract recognition transfers to contract-consistent execution")
    if deep_b and target_b == "standard":
        labels.append("contract recognition and execution disagree")
    if deep_b_standard and target_b == "standard":
        labels.append("the probe matches the standard definition and cannot yet separate a lost contract, a template, and an ordinary error")
    if deep_b and target_b == "neither":
        labels.append("the contract is recognized, but the execution fails")
    if (
        deep_b
        and deep_gold
        and target_b not in {"contract", "both"}
        and target_gold not in {"contract", "both"}
    ):
        labels.append("the error is more likely in execution or formula assembly than in contract recognition")
    if facts["deep_b_parsed"] and not deep_b and not deep_b_standard:
        labels.append("the probe answer is neither the contract nor the standard")
    if not facts["parents_correct"] and target_b not in {"contract", "both"}:
        labels.append("a direct parent is wrong, so the execution result cannot be explained alone")
    return labels or ["does not fall under a stated explanation"]


_LEAKS = ("left-hand", "right-hand", "re-derive", "write it again", "defined as opposite", "{", "}")


def _chains() -> tuple[dict[str, dict], dict[str, dict]]:
    from cloze_eval import contexts_for, load_samples
    from cloze_eval import ROOT as root

    wanted = {item["sample_id"] for item in PROBES} | {"standard-default"}
    rows = {row["id"]: row for row in load_samples(root / "data" / "wind_tunnel" / "pair" / "samples.jsonl", wanted)}
    chains = {}
    standard = {}
    for probe in PROBES:
        key = (probe["sample_id"], probe["process"])
        if key not in chains:
            row = rows[probe["sample_id"]]
            chains[key] = next(item for item in contexts_for(row) if item["process"] == probe["process"])
        if probe["process"] not in standard:
            row = rows["standard-default"]
            standard[probe["process"]] = next(item for item in contexts_for(row) if item["process"] == probe["process"])
    return chains, standard


def _blank_map(chain: dict) -> dict[str, dict]:
    found = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            found[str(blank["blank_id"])] = blank
    return found


def _gold_memory(chain: dict) -> dict[str, str]:
    from cloze_eval import _repair_phrase

    memory = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            memory[str(blank["blank_id"])] = _repair_phrase(blank)
    return memory


def check_probes() -> None:
    """Build every probe on the manuscript and reject leaked or full-answer questions."""
    chains, standard_chains = _chains()
    for probe in PROBES:
        chain = chains[(probe["sample_id"], probe["process"])]
        blanks = _blank_map(chain)
        standard_blanks = _blank_map(standard_chains[probe["process"]])
        memory = _gold_memory(chain)
        for condition in ("deep_b",):
            question = _question(probe, condition, {})
            for leak in _LEAKS:
                if leak in question:
                    raise RuntimeError(f"The stem of {probe['probe_id']} contains {leak}")
            if "Do not write the full" not in question:
                raise RuntimeError(f"{probe['probe_id']} does not forbid writing the full target")
            prompt = build_prompt(chain, probe, condition, memory, {})
            target_prompt = build_target_prompt(chain, probe, memory)
            if f"{{ID:{probe['target']}," not in target_prompt:
                raise RuntimeError(f"The target branch of {probe['probe_id']} has no open target blank")
            if "is not written back into the original flow" not in prompt:
                raise RuntimeError(f"{probe['probe_id']} does not say the answer is not written back")
            if prompt.index("[Probe]") < prompt.index("[Process]"):
                raise RuntimeError(f"The probe of {probe['probe_id']} does not come after the process")
            prefix = prompt.split("[Probe]", 1)[0]
            if f"@@{probe['target']}@@" in prefix:
                raise RuntimeError(f"The prefix of {probe['probe_id']} still has the target slot")
        contract = contract_key(probe, recognition_text(probe, lambda blank_id: memory[blank_id]))
        standard_memory = _gold_memory(standard_chains[probe["process"]])
        standard = contract_key(
            probe,
            recognition_text(probe, lambda blank_id, bank=standard_memory: bank[blank_id]),
        )
        if probe["kind"] in {"cells", "sign", "entry", "leading"} and contract == standard:
            raise RuntimeError(f"The contract and the standard definition of {probe['probe_id']} have no distinguishable local answer")
        if probe["kind"] == "placement" and contract.get("label") != "outside":
            raise RuntimeError(f"The placement rule of {probe['probe_id']} is not outside")
        deep_ids = []
        for group in _groups(chain):
            deep_ids.extend(group)
            if probe["after"] in group:
                break
        for parent in probe["repair_parent"] + probe["repair_other"]:
            if parent not in deep_ids:
                raise RuntimeError(f"Repair point ID{parent} of {probe['probe_id']} is not in the probe prefix")
        if probe["kind"] == "sign" and probe["source_id"] in deep_ids:
            raise RuntimeError(f"The probe prefix of {probe['probe_id']} already contains the sign blank being measured")
        if probe["target"] not in blanks or probe["target"] not in standard_blanks:
            raise RuntimeError(f"{probe['probe_id']} is missing the target blank")
    labels = " ".join(classify({
        "deep_b_hit": True,
        "deep_gold_hit": True,
        "deep_b_standard": False,
        "deep_b_parsed": True,
        "target_b": "standard",
        "target_gold_run": "contract",
        "parents_correct": True,
    }))
    if "contract recognition and execution disagree" not in labels or "rollback" in labels:
        raise RuntimeError(f"Unexpected probe explanation labels: {labels}")
    print(f"probe check ok: {len(PROBES)} side probes")


if __name__ == "__main__":
    check_probes()

