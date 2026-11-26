# -*- coding: utf-8 -*-
"""Error-first diagnosis. Every wrong answer becomes a case; chains only explain it.

A wrong node is registered even when its contract is contradictory or its cause
cannot be checked. Consistency metrics may exclude that case. The audit may not.
"""
from __future__ import annotations

from domains.fault_class import EXE, RSN

import json
from dataclasses import dataclass, field
from pathlib import Path

from cloze_score import identified_option, judge, resolve_choice, sympy_equal
from cloze_score import _parse_expr, _sympy_modules
from domains.wind_tunnel.contract_executor import (
    ALPHA_SYM,
    BETA_SYM,
    GAMMA_SYM,
    attitude_target,
    generated_answers,
    roll_target,
)
from domains.wind_tunnel.contract_executor import _model
from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.contract_verifiers import validator_of
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.fault_layers import level_of, major

VERSION = "v1"
PROCESS_SLUG = {
    "wind frame to body-axis frame": "wind_to_body",
    "balance frame to body-axis frame": "balance_to_body",
    "balance frame to wind frame": "balance_to_wind",
    "moment reference point": "moment_reference",
}
PHI = {
    "26": ("attitude", ALPHA_SYM, ("23", "24", "25"), "alpha"),
    "34": ("attitude", BETA_SYM, ("31", "32", "33"), "beta"),
    "43": ("attitude", GAMMA_SYM, ("40", "41", "42"), "gamma"),
    "88": ("roll", None, ("83", "84", "86", "87", "89", "90", "91"), "roll"),
}
ANCHOR_AXIS = {"24": "23", "32": "31", "41": "40"}
FRAME_FAMILY = {
    "intermediate body-axis frame": "TZ",
    "body-axis frame": "TZ",
    "wind frame": "FD",
    "balance frame": "TP",
}
AXIS_SLOT = {
    "XTZ": "X", "XFD": "X", "XTP": "X",
    "YTZ": "Y", "YFD": "Y", "YTP": "Y",
    "ZTZ": "Z", "ZFD": "Z", "ZYP": "Z",
}
CLOSED = {"ROOT_ERROR", "PROPAGATED_ERROR", "NEW_LOCAL_ERROR", "LOCAL_CONTRACT_INCONSISTENCY", "FORMAT_ERROR", "CONDITIONAL_FAILURE"}
REVIEW = {"UNPARSEABLE_PENDING_REVIEW", "AMBIGUOUS"}
SURFACE_OF_LABEL = {
    "C-PATH-ANCHOR": "PATH_ANCHOR",
    "C-PATH-DIRECTION": "PATH_DIRECTION",
    "C-PATH-EDGE": "PATH_EDGE",
    "C-PATH-BRANCH": "PATH_BRANCH",
    "C-PATH-SEQUENCE": "ORDER",
    "C-PATH-STATE": "PATH_STATE",
    "C-PATH-STATE-TRANSITION": "PATH_STATE",
    "C-PATH-CONVENTION": "PATH_CONVENTION",
    "C-PATH-RELATION": "PATH_RELATION",
    "C-PATH-REFERENCE": "PATH_REFERENCE",
    "C-PATH-REPRESENTATION": "PATH_REPRESENTATION",
    "C-AXIS": "AXIS_BINDING",
    "C-SIGN": "SIGN",
    "C-FRM": "FRAME_BINDING",
    "C-ORDER": "ORDER",
    "C-DEF": "CHOICE_VALUE",
    "C-ASSOC": "CHOICE_VALUE",
    "A-FULL-EXEC": "MATRIX_STRUCTURE",
    "A-MATH": "EXPRESSION",
    "A-LOCAL-EXEC": "EXPRESSION",
    "A-SPATIAL": "CHOICE_VALUE",
}
CLASS_ZH = {
    "ROOT_ERROR": "root error",
    "CONDITIONAL_FAILURE": "current-layer condition failed",
    "PROPAGATED_ERROR": "verified propagation",
    "NEW_LOCAL_ERROR": "new local execution error",
    "LOCAL_CONTRACT_INCONSISTENCY": "local contract execution inconsistency",
    "CONTRACT_TARGET_DISCONNECT": "answer correct but does not follow the stated contract",
    "RESULT_EQUIVALENCE": "result equivalent under the wrong contract",
    "FORMAT_ERROR": "format error",
    "UNPARSEABLE_PENDING_REVIEW": "cannot parse, pending review",
    "AMBIGUOUS": "ambiguous answer, pending review",
    "UNRESOLVED": "unresolved",
}


@dataclass
class Observation:
    sample_id: str
    condition: str
    prefix: str
    process: str
    node_id: str
    gold_answer: str
    model_answer: str
    standard_answer: str
    options: list[str]
    format_code: str
    parse_status: str
    correct: bool
    error_kind: str
    label: str
    subtype: str
    asked: bool = True
    node_level: str = ""


@dataclass
class Case:
    error_id: str
    sample_id: str
    condition: str
    prefix: str
    process: str
    node_id: str
    chain_id: str
    role: str
    gold_answer: str
    model_answer: str
    case_origin: str
    parse_status: str
    surface_type: str
    final_class: str
    propagation_status: str
    causal_relation: str
    eligibility_for_consistency: str
    propagated_from: list[str] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    phi_matrix: str = ""
    analysis_version: str = VERSION
    node_level: str = ""
    failure_level: str = ""
    cause_level: str = ""
    primary_cause: str = ""
    secondary_causes: list[str] = field(default_factory=list)
    minimal_fault_set: list[str] = field(default_factory=list)
    candidate_fault_sets: list[list[str]] = field(default_factory=list)
    mixed_fault: bool = False
    phi_verified: bool = False
    validator_id: str = ""
    validator_version: str = ""
    gold_prefix_status: str = ""
    model_prefix_status: str = ""
    evidence_level: str = "none"
    upstream_wrong: list[str] = field(default_factory=list)

    @property
    def bucket(self) -> str:
        if self.final_class in CLOSED:
            return "closed"
        if self.final_class in REVIEW:
            return "review"
        return "unresolved"


def _phrase(raw: str, options: list[str]) -> str:
    text = str(raw or "").strip()
    usable = [item for item in options if item not in {"", "none"}]
    if not usable:
        return text
    _letter, body = resolve_choice(text, usable)
    return body or text


def _options(blank: dict) -> list[str]:
    return [str(item) for item in (blank.get("options") or []) if str(item) not in {"", "none"}]


def _label(node_id: str) -> tuple[str, str]:
    return TARGETS.get(node_id, ("", ""))


def _parse_status(prediction: str, options: list[str], accepted: bool, format_code: str) -> str:
    text = prediction.strip()
    if accepted:
        return "correct"
    if not text:
        return "unparseable"
    if options:
        if identified_option(text, options) is None:
            letters = [item for item in options if item and item in text]
            if len(letters) > 1:
                return "ambiguous"
            return "wrong"
        return "wrong"
    if format_code in {"matrix", "vector", "expression"}:
        try:
            sympy, names = _sympy_modules()
            _parse_expr(text, sympy, names)
        except Exception:
            return "unparseable"
    return "wrong"


def _surface(label: str, format_code: str) -> str:
    if format_code == "matrix":
        return "MATRIX_STRUCTURE"
    if format_code == "vector":
        return "VECTOR_COMPONENT"
    if format_code == "expression":
        return "EXPRESSION"
    return SURFACE_OF_LABEL.get(label, "CHOICE_VALUE")


def _matrix_findings(prediction: str, gold: str) -> list[dict]:
    try:
        sympy, names = _sympy_modules()
        left = _parse_expr(prediction, sympy, names)
        right = _parse_expr(gold, sympy, names)
    except Exception:
        return []
    if not hasattr(left, "rows") or not hasattr(right, "rows"):
        return []
    findings = []
    if (left.rows, left.cols) != (right.rows, right.cols):
        findings.append({"type": "MATRIX_STRUCTURE", "detail": "row and column sizes differ"})
        return findings
    for row in range(left.rows):
        for col in range(left.cols):
            if sympy.simplify(left[row, col] - right[row, col]) == 0:
                continue
            flipped = sympy.simplify(left[row, col] + right[row, col]) == 0
            findings.append({
                "type": "EXEC_SIGN" if flipped else "EXEC_VALUE",
                "detail": f"row {row + 1} column {col + 1}",
            })
    return findings


def _family(axis: str) -> str:
    if axis.endswith("TZ"):
        return "TZ"
    if axis.endswith("FD"):
        return "FD"
    if axis.endswith("TP") or axis == "ZYP":
        return "TP"
    return ""


def load_observations(path: Path, sample_id: str, condition: str, gold: dict[str, str], standard: dict[str, str]) -> list[Observation]:
    prefix = "model" if condition in {"self-trunc", "self-keep", "session-model"} else "gold"
    found: dict[str, Observation] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("sample_id") or "") != sample_id or row.get("branch") == "probe":
            continue
        if str(row.get("group") or "main") not in {"main", ""}:
            continue
        if str(row.get("condition") or condition) not in {condition, ""}:
            continue
        process = str(row.get("process") or "")
        for blank in row.get("blanks") or []:
            node_id = str(blank.get("blank_id") or "")
            options = _options(blank)
            prediction = _phrase(str(blank.get("prediction") or ""), options)
            answer = _phrase(str(blank.get("answer") or ""), options)
            scored = judge(
                str(blank.get("prediction") or ""),
                str(blank.get("answer") or ""),
                str(blank.get("format_code") or ""),
                options,
                standard.get(node_id),
            )
            status = _parse_status(prediction, options, bool(scored["normalized"]), str(blank.get("format_code") or ""))
            label, subtype = _label(node_id)
            found[node_id] = Observation(
                sample_id=sample_id,
                condition=condition,
                prefix=prefix,
                process=process,
                node_id=node_id,
                gold_answer=answer,
                model_answer=prediction,
                standard_answer=standard.get(node_id, ""),
                options=options,
                format_code=str(blank.get("format_code") or ""),
                parse_status="correct" if scored["normalized"] else status,
                correct=bool(scored["normalized"]),
                error_kind=str(scored.get("error_kind") or ""),
                label=label,
                subtype=subtype,
            )
    return list(found.values())


def _parents(node_id: str) -> tuple[str, ...]:
    node = NODE_BY_ID.get(node_id)
    if node is None:
        return ()
    return node.depends_on


def _value(obs: dict[str, Observation], gold: dict[str, str], node_id: str) -> str:
    item = obs.get(node_id)
    if item is None:
        return gold.get(node_id, "")
    return item.model_answer


def _model_texts(by_id: dict[str, Observation], gold: dict[str, str], parents: tuple[str, ...]) -> dict[str, str]:
    """Contract cognition from recognition answers, then these direct parents."""
    from domains.contract_cognition import answer_sheet, working_texts
    from domains.wind_tunnel.contract_cognition import recognition_ids

    sheet = answer_sheet({node: item.model_answer for node, item in by_id.items()})
    return working_texts(gold, sheet, recognition_ids(), parents)


def _phi(model, node_id: str, phrases: list[str], texts: dict[str, str] | None = None):
    kind, symbol, _parents_ids, _chain = PHI[node_id]
    if kind == "attitude":
        return attitude_target(model, symbol, phrases[0], phrases[1], phrases[2], texts)
    return roll_target(model, *phrases)


def _axis_relation(obs: dict[str, Observation], node_id: str) -> tuple[str, str] | None:
    anchor_id = ANCHOR_AXIS.get(node_id)
    if anchor_id is None:
        return None
    anchor = obs.get(anchor_id)
    current = obs.get(node_id)
    if anchor is None or current is None or anchor.correct:
        return None
    declared = FRAME_FAMILY.get(anchor.model_answer)
    axis = current.model_answer
    if declared is None or axis not in AXIS_SLOT:
        return None
    if _family(axis) == declared and AXIS_SLOT.get(current.gold_answer) == AXIS_SLOT[axis]:
        return "follows_declared_frame", anchor_id
    if axis == current.gold_answer:
        return "parent_wrong_but_not_reflected", anchor_id
    return None


def _case_id(obs: Observation, origin: str) -> str:
    mark = "lat" if origin == "latent" else "err"
    return f"{mark}_{obs.sample_id}_{obs.condition}_ID{obs.node_id}"


def _base_case(obs: Observation, origin: str) -> Case:
    chain = PHI.get(obs.node_id, (None, None, None, ""))[3]
    for node_id, spec in PHI.items():
        if obs.node_id in spec[2]:
            chain = spec[3]
    return Case(
        error_id=_case_id(obs, origin),
        sample_id=obs.sample_id,
        condition=obs.condition,
        prefix=obs.prefix,
        process=PROCESS_SLUG.get(obs.process, obs.process),
        node_id=f"ID{obs.node_id}",
        chain_id=chain,
        role=obs.label,
        gold_answer=obs.gold_answer,
        model_answer=obs.model_answer,
        case_origin=origin,
        parse_status=obs.parse_status,
        surface_type=_surface(obs.label, obs.format_code),
        final_class="UNRESOLVED",
        propagation_status="pending",
        causal_relation="unresolved",
        eligibility_for_consistency="eligible",
    )


def diagnose(model, observations: list[Observation], gold: dict[str, str]) -> list[Case]:
    by_id = {item.node_id: item for item in observations}
    cases: list[Case] = []
    for obs in observations:
        if obs.parse_status == "correct" and obs.node_id not in PHI:
            continue
        if obs.parse_status == "correct":
            cases.extend(_latent(model, obs, by_id, gold))
            continue
        cases.append(_diagnose_wrong(model, obs, by_id, gold))
    return cases


def _diagnose_wrong(model, obs: Observation, by_id: dict[str, Observation], gold: dict[str, str]) -> Case:
    case = _base_case(obs, "wrong_answer")
    case.findings.append({"type": case.surface_type, "detail": obs.error_kind or "differs from the gold of this question"})
    if obs.format_code == "matrix":
        case.findings.extend(_matrix_findings(obs.model_answer, obs.gold_answer))
    if obs.parse_status == "unparseable":
        case.final_class = "UNPARSEABLE_PENDING_REVIEW"
        case.propagation_status = "not_applicable"
        case.causal_relation = "unresolved"
        case.eligibility_for_consistency = "excluded_unparseable"
        return case
    if obs.parse_status == "ambiguous":
        case.final_class = "AMBIGUOUS"
        case.propagation_status = "not_applicable"
        case.causal_relation = "unresolved"
        case.eligibility_for_consistency = "excluded_ambiguous"
        return case
    if obs.prefix == "gold":
        return _diagnose_gold_prefix(model, case, obs, gold)
    axis = None if validator_of(obs.node_id) else _axis_relation(by_id, obs.node_id)
    if axis is not None:
        relation, parent = axis
        case.relations.append({"related_error_id": f"ID{parent}", "relation_type": relation})
        if relation == "follows_declared_frame":
            case.final_class = "PROPAGATED_ERROR"
            case.propagation_status = "verified_axis_follows_declared_frame"
            case.causal_relation = "propagated_from"
            case.propagated_from = [f"ID{parent}"]
            case.findings.append({"type": "WRONG_PATH_AXIS", "detail": f"the axis lies in the frame declared by ID{parent}"})
            return case
        case.final_class = "NEW_LOCAL_ERROR"
        case.propagation_status = "parent_wrong_but_not_reflected"
        case.causal_relation = "not_reflected"
        case.propagated_from = []
        return case
    if obs.node_id in PHI:
        return _diagnose_phi(model, case, obs, by_id, gold)
    wrong_parents = [
        parent for parent in _parents(obs.node_id)
        if parent in by_id and not by_id[parent].correct
    ]
    for parent in wrong_parents:
        case.relations.append({"related_error_id": f"ID{parent}", "relation_type": "upstream_wrong"})
    if major(level_of(obs.node_id)) == "L0":
        case.final_class = "ROOT_ERROR"
        case.propagation_status = "no_wrong_parent" if not wrong_parents else "upstream_wrong_unverified"
        case.causal_relation = "no_parent_error" if not wrong_parents else "contract_error"
        return case
    if not wrong_parents:
        case.final_class = "UNRESOLVED"
        case.propagation_status = "no_verifier"
        case.causal_relation = "unresolved"
        case.eligibility_for_consistency = "no_verifier"
        return case
    case.final_class = "UNRESOLVED"
    case.propagation_status = "upstream_wrong_unverified"
    case.causal_relation = "unresolved"
    case.eligibility_for_consistency = "no_verifier"
    return case


def _diagnose_gold_prefix(model, case: Case, obs: Observation, gold: dict[str, str]) -> Case:
    """A wrong answer under the gold prefix is a conditional failure, not a proven root."""
    case.propagation_status = "not_applicable_gold_prefix"
    case.eligibility_for_consistency = "gold_prefix_not_self_consistency"
    case.causal_relation = "conditional_failure"
    if obs.node_id in PHI:
        phrases = [gold.get(parent, "") for parent in PHI[obs.node_id][2]]
        derived = _phi(model, obs.node_id, phrases, gold)
        if not derived.ok:
            case.final_class = "UNRESOLVED"
            case.causal_relation = "unresolved"
            case.findings.append({"type": "CONTRACT_CONTRADICTION", "detail": derived.reason})
            return case
        case.phi_matrix = derived.matrix
    case.final_class = "CONDITIONAL_FAILURE"
    return case


def _diagnose_phi(model, case: Case, obs: Observation, by_id: dict[str, Observation], gold: dict[str, str]) -> Case:
    parent_ids = PHI[obs.node_id][2]
    phrases = [_value(by_id, gold, parent) for parent in parent_ids]
    wrong_parents = [parent for parent in parent_ids if parent in by_id and not by_id[parent].correct]
    for parent in wrong_parents:
        case.relations.append({"related_error_id": f"ID{parent}", "relation_type": "upstream_wrong"})
    derived = _phi(model, obs.node_id, phrases, _model_texts(by_id, gold, parent_ids))
    if not derived.ok:
        from domains.wind_tunnel.eval_scope import scope_of

        case.findings.append({"type": "CONTRACT_CONTRADICTION", "detail": derived.reason})
        if scope_of(obs.node_id) in {RSN, EXE}:
            case.final_class = "UNRESOLVED"
            case.propagation_status = "contract_contradiction"
            case.causal_relation = "indistinguishable"
            case.eligibility_for_consistency = "conservative_union"
            case.candidate_fault_sets = []
            if wrong_parents:
                case.candidate_fault_sets.append([f"ID{parent}" for parent in wrong_parents])
            case.candidate_fault_sets.append([f"ID{obs.node_id}"])
            case.upstream_wrong = [f"ID{parent}" for parent in wrong_parents]
            return case
        case.final_class = "UNRESOLVED"
        case.propagation_status = "contract_contradiction"
        case.causal_relation = "unresolved"
        case.eligibility_for_consistency = "excluded_contradiction"
        return case
    case.phi_matrix = derived.matrix
    matches_phi = sympy_equal(obs.model_answer, derived.matrix)
    if matches_phi and wrong_parents:
        case.final_class = "PROPAGATED_ERROR"
        case.propagation_status = "verified_phi"
        case.causal_relation = "propagated_from"
        case.propagated_from = [f"ID{parent}" for parent in wrong_parents]
        case.eligibility_for_consistency = "eligible"
        return case
    if matches_phi and not wrong_parents:
        case.final_class = "UNRESOLVED"
        case.propagation_status = "phi_matches_without_wrong_parent"
        case.causal_relation = "unresolved"
        return case
    case.final_class = "NEW_LOCAL_ERROR"
    case.propagation_status = "not_equal_to_phi"
    case.causal_relation = "new_local"
    case.eligibility_for_consistency = "eligible"
    return case


def _latent(model, obs: Observation, by_id: dict[str, Observation], gold: dict[str, str]) -> list[Case]:
    if obs.prefix != "model" or not obs.correct:
        return []
    parent_ids = PHI[obs.node_id][2]
    phrases = [_value(by_id, gold, parent) for parent in parent_ids]
    wrong_parents = [parent for parent in parent_ids if parent in by_id and not by_id[parent].correct]
    if not wrong_parents:
        return []
    derived = _phi(model, obs.node_id, phrases, _model_texts(by_id, gold, parent_ids))
    if not derived.ok:
        return []
    case = _base_case(obs, "latent")
    case.phi_matrix = derived.matrix
    case.eligibility_for_consistency = "latent_not_a_wrong_answer"
    matches_phi = sympy_equal(obs.model_answer, derived.matrix)
    if matches_phi:
        case.final_class = "RESULT_EQUIVALENCE"
        case.propagation_status = "verified_phi"
        case.causal_relation = "wrong_contract_same_result"
        case.propagated_from = [f"ID{parent}" for parent in wrong_parents]
        case.findings.append({"type": "RESULT_EQUIVALENCE", "detail": "the matrix derived from the wrong contract equals the gold of this question, and the model answer equals that matrix"})
    else:
        case.final_class = "CONTRACT_TARGET_DISCONNECT"
        case.propagation_status = "parent_wrong_but_not_reflected"
        case.causal_relation = "answer_matches_gold_not_phi"
        case.findings.append({"type": "CONTRACT_TARGET_DISCONNECT", "detail": "the answer equals the gold of this question, but it does not equal the matrix derived from the model contract"})
    for parent in wrong_parents:
        case.relations.append({"related_error_id": f"ID{parent}", "relation_type": "upstream_wrong"})
    return [case]


def audit(observations: list[Observation], cases: list[Case]) -> dict:
    wrong = [item for item in observations if item.parse_status != "correct"]
    wrong_cases = [item for item in cases if item.case_origin == "wrong_answer"]
    ids = [item.node_id for item in wrong_cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("One wrong question produced more than one error case.")
    if len(wrong_cases) != len(wrong):
        raise RuntimeError(f"There are {len(wrong)} wrong questions and {len(wrong_cases)} registered cases.")
    missing = {f"ID{item.node_id}" for item in wrong} - {item.node_id for item in wrong_cases}
    if missing:
        raise RuntimeError(f"These wrong questions have no case: {sorted(missing)}")
    for case in wrong_cases:
        if case.final_class == "PROPAGATED_ERROR" and not case.propagated_from:
            raise RuntimeError(f"{case.error_id} is marked propagated but has no parent.")
        if case.final_class == "ROOT_ERROR" and case.propagated_from:
            raise RuntimeError(f"{case.error_id} is marked a root error but still points at a parent.")
        if case.bucket not in {"closed", "review", "unresolved"}:
            raise RuntimeError(f"{case.error_id} was not placed in an audit bucket.")
    closed = [item for item in wrong_cases if item.bucket == "closed"]
    review = [item for item in wrong_cases if item.bucket == "review"]
    unresolved = [item for item in wrong_cases if item.bucket == "unresolved"]
    if len(closed) + len(review) + len(unresolved) != len(wrong_cases):
        raise RuntimeError("Not every wrong-question case is closed, in review, or unresolved.")
    return {
        "nodes": len(observations),
        "correct": sum(item.correct for item in observations),
        "wrong": len(wrong),
        "registered": len(wrong_cases),
        "closed": len(closed),
        "review": len(review),
        "unresolved": len(unresolved),
        "latent": sum(item.case_origin == "latent" for item in cases),
        "auto": len(closed) + len(unresolved),
        "registration_coverage": 1,
        "diagnosis_coverage": 1,
    }


def _dump(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_run(directory: Path, observations: list[Observation], cases: list[Case], summary: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _dump(directory / "observations.jsonl", [item.__dict__ for item in observations])
    _dump(directory / "error_cases.jsonl", [
        {key: value for key, value in item.__dict__.items() if key != "findings" and key != "relations"}
        for item in cases
    ])
    findings = []
    relations = []
    for case in cases:
        for index, finding in enumerate(case.findings, start=1):
            findings.append({"error_id": case.error_id, "finding_id": f"{case.error_id}_{index}", **finding})
        for relation in case.relations:
            relations.append({"error_id": case.error_id, **relation})
    _dump(directory / "findings.jsonl", findings)
    _dump(directory / "relations.jsonl", relations)
    if cases:
        from domains.wind_tunnel.conflict_diagnosis import stored_report

        report = stored_report(cases[0].sample_id, cases[0].condition)
        if report is not None:
            (directory / "conflict_diagnosis.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            summary["conflict_status"] = report["status"]
            summary["diagnosis_cardinality"] = report["diagnosis_cardinality"]
            summary["repair_cardinality"] = report["repair_cardinality"]
            summary["targets"] = report.get("targets") or []
    (directory / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_report(runs: list[tuple[str, dict, list[Case]]]) -> str:
    lines = [
        "# Error-first diagnosis",
        "",
        "Each wrong question has exactly one error case. A chain only explains a registered wrong question. It does not decide which questions enter the analysis.",
        "Wrong questions with a contradictory contract or with no verifier stay in the audit. They may be left out of the contract-consistency rate.",
        "An answer that is algebraically equivalent to the gold is not a wrong question. An answer that is correct but inconsistent with the model contract is a separate latent case and is not mixed into the wrong-question identity.",
        "",
    ]
    for name, summary, cases in runs:
        wrong_cases = [item for item in cases if item.case_origin == "wrong_answer"]
        lines.extend(_run_section(name, summary, wrong_cases, [item for item in cases if item.case_origin == "latent"]))
    return "\n".join(lines) + "\n"


def _run_section(name: str, summary: dict, cases: list[Case], latent: list[Case]) -> list[str]:
    from domains.conflict_diagnosis import target_note

    lines = [
        f"## {name}",
        "",
        "| Item | Count |",
        "| --- | ---: |",
        f"| Scorable nodes | {summary['nodes']} |",
        f"| Nodes judged correct | {summary['correct']} |",
        f"| Nodes judged wrong | {summary['wrong']} |",
        f"| Registered error cases | {summary['registered']} |",
        f"| Automatically diagnosed | {summary['auto']} |",
        f"| Sent to review | {summary['review']} |",
        f"| Closed cases | {summary['closed']} |",
        f"| unresolved cases | {summary['unresolved']} |",
        f"| Latent cases | {summary['latent']} |",
        "",
        "Nodes judged wrong = registered error cases = closed cases + cases in review + unresolved cases.",
        "",
        "### Error cases",
        "",
        "| Final class | Cases |",
        "| --- | ---: |",
    ]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.final_class] = counts.get(case.final_class, 0) + 1
    for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {CLASS_ZH.get(key, key)} | {count} |")
    lines.extend(["", "### Root cause and propagation", ""])
    roots = sum(item.final_class == "ROOT_ERROR" for item in cases)
    propagated = sum(item.final_class == "PROPAGATED_ERROR" for item in cases)
    local = sum(item.final_class == "NEW_LOCAL_ERROR" for item in cases)
    lines.extend([
        f"- Root errors {roots}. Propagation {propagated}. Local contract execution inconsistency {sum(item.final_class == 'LOCAL_CONTRACT_INCONSISTENCY' for item in cases)}. New local execution errors {local}. Current-layer condition failed {sum(item.final_class == 'CONDITIONAL_FAILURE' for item in cases)}.",
        "- A wrong question under the gold prefix only says that a correct prefix still fails. It is not recorded as a proven root cause. R1, R2, and R3 are the exception: the transform order given in the problem is right, and a matrix written in the wrong place is local contract execution inconsistency.",
        "- Propagation holds only when an axis follows the declared frame, or when the target matrix equals phi of the model contract. L1, L2, and L3 wrong questions with no verifier stay unresolved even when every upstream answer is right. An L0 wrong question is a root error, except where an axis-following verifier already explains it. The transform order ID16-ID18 is derived directly from the declaration, and a wrong answer is a root error. Product positions depend on that order. A right order with a matrix in the wrong place is only local contract execution inconsistency.",
        *(
            [
                f"- Graph candidates {summary['conflict_status']}. Repair cardinality {summary['repair_cardinality']}. The repair set is the local deviation inside the ancestor cone of the endpoint, read by a local recompute."
                + target_note(summary.get("targets") or [])
            ]
            if "conflict_status" in summary
            else []
        ),
        "",
        "### Priority review",
        "",
    ])
    highlighted = [
        item for item in latent + cases
        if item.final_class in {
            "RESULT_EQUIVALENCE", "CONTRACT_TARGET_DISCONNECT", "PROPAGATED_ERROR",
            "UNPARSEABLE_PENDING_REVIEW", "AMBIGUOUS",
        } or (item.final_class == "NEW_LOCAL_ERROR" and item.surface_type == "MATRIX_STRUCTURE")
        or (item.final_class in {"UNRESOLVED", "ROOT_ERROR"} and item.role.startswith("C-PATH"))
        or (item.final_class in {"UNRESOLVED", "ROOT_ERROR", "NEW_LOCAL_ERROR", "CONDITIONAL_FAILURE"} and item.surface_type == "MATRIX_STRUCTURE")
    ]
    if not highlighted:
        lines.append("This group has no case in priority review.")
        lines.append("")
        return lines
    lines.extend([
        "| Case | Node | Class | Model answer | Correct answer |",
        "| --- | --- | --- | --- | --- |",
    ])
    for case in highlighted:
        lines.append(
            f"| {case.error_id} | {case.node_id} {case.role} | {CLASS_ZH.get(case.final_class, case.final_class)} | "
            f"{_cell(case.model_answer)} | {_cell(case.gold_answer)} |"
        )
    lines.append("")
    return lines


def _cell(text: str) -> str:
    flat = " ".join(str(text).split()).replace("|", "∣")
    if len(flat) > 160:
        return flat[:157] + "..."
    return flat
