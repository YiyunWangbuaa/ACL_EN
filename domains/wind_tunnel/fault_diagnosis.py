# -*- coding: utf-8 -*-
"""Layered fault diagnosis on top of the error-first cases. No model calls."""
from __future__ import annotations

from itertools import combinations

from cloze_score import sympy_equal
from domains.wind_tunnel.contract_executor import attitude_target, roll_target
from domains.wind_tunnel.contract_labels import TARGETS
from domains.wind_tunnel.contract_verifiers import derive, input_ids, validator_of
from domains.wind_tunnel.error_workflow import ANCHOR_AXIS, PHI, PROCESS_SLUG, Case, Observation, _surface
from domains.wind_tunnel.fault_layers import JOINT_GROUPS, LEVELS, audit_layers, closure, level_of, major

VALIDATOR_VERSION = "v1"
COLUMNS = ("primary L0", "primary L1", "primary L2", "primary L3", "multiple faults", "current-layer condition failed", "unresolved")


def _bare(node_id: str) -> str:
    return str(node_id).removeprefix("ID")


def _tagged(node_id: str) -> str:
    return node_id if str(node_id).startswith("ID") else f"ID{node_id}"


def _same(left: str, right: str) -> bool:
    if str(left).strip() == str(right).strip():
        return True
    return sympy_equal(left, right)


def _label(node_id: str) -> str:
    label, _subtype = TARGETS.get(_bare(node_id), ("", ""))
    return label or "unlabeled"


def _stamp(observations: list[Observation]) -> None:
    for item in observations:
        item.node_level = LEVELS.get(item.node_id, "")


def _by_id(observations: list[Observation]) -> dict[str, Observation]:
    return {item.node_id: item for item in observations}


def _phrase(observations: dict[str, Observation], gold: dict[str, str], node_id: str) -> str:
    item = observations.get(node_id)
    if item is None or not item.model_answer.strip():
        return gold[node_id]
    return item.model_answer


def _derive(model, node_id: str, phrases: list[str], texts: dict[str, str]):
    kind, symbol, _parents, _chain = PHI[node_id]
    if kind == "attitude":
        return attitude_target(model, symbol, *phrases, texts)
    return roll_target(model, *phrases)


def _matches(execution, answer: str) -> bool:
    return execution.ok and _same(execution.matrix, answer)


def _minimal(model, node_id: str, parents: tuple[str, ...], observed: dict[str, Observation], gold: dict[str, str], answer: str):
    model_phrases = [_phrase(observed, gold, parent) for parent in parents]
    gold_phrases = [gold[parent] for parent in parents]
    wrong = [parent for parent, left, right in zip(parents, model_phrases, gold_phrases) if not _same(left, right)]

    def run(active: set[str]):
        from domains.contract_cognition import answer_sheet, working_texts
        from domains.wind_tunnel.contract_cognition import recognition_ids

        sheet = answer_sheet({node: item.model_answer for node, item in observed.items()})
        phrases = []
        for index, parent in enumerate(parents):
            phrase = model_phrases[index] if parent in active else gold_phrases[index]
            sheet[parent] = phrase
            phrases.append(phrase)
        texts = working_texts(gold, sheet, recognition_ids(), parents)
        return _derive(model, node_id, phrases, texts)

    if not _matches(run(set(wrong)), answer):
        return wrong, None
    sufficient = []
    for size in range(len(wrong) + 1):
        for combo in combinations(wrong, size):
            if _matches(run(set(combo)), answer):
                sufficient.append(set(combo))
    minimal = []
    for item in sufficient:
        if not any(other < item for other in sufficient):
            minimal.append(sorted(item, key=int))
    return wrong, minimal


def _status(item: Observation | None) -> str:
    if item is None:
        return "unpaired"
    return "right" if item.correct else "wrong"


def _pair(case: Case, own: dict[str, Observation], other: dict[str, Observation] | None) -> None:
    bare = _bare(case.node_id)
    mine = _status(own.get(bare))
    paired = _status(None if other is None else other.get(bare))
    if case.prefix == "model":
        case.model_prefix_status = mine
        case.gold_prefix_status = paired
    else:
        case.gold_prefix_status = mine
        case.model_prefix_status = paired


def _upstream(node_id: str, observed: dict[str, Observation]) -> list[str]:
    return [
        parent for parent in closure(node_id)
        if parent in observed and not observed[parent].correct
    ]


def _reset(case: Case) -> None:
    case.failure_level = ""
    case.cause_level = ""
    case.primary_cause = ""
    case.secondary_causes = []
    case.minimal_fault_set = []
    case.candidate_fault_sets = []
    case.mixed_fault = False
    case.phi_verified = False
    case.validator_id = ""
    case.validator_version = ""
    case.evidence_level = "none"


def _validator(node_id: str) -> str:
    kind = PHI[node_id][0]
    if kind == "roll":
        return "roll_matrix_phi"
    return "pose_matrix_phi"


def _bind(case: Case, validator: str) -> None:
    case.validator_id = validator
    case.validator_version = VALIDATOR_VERSION
    case.phi_verified = True


def annotate(model, observations: list[Observation], cases: list[Case], gold: dict[str, str], paired: list[Observation] | None) -> None:
    _add_composed_latent(observations, cases, gold)
    _stamp(observations)
    if paired is not None:
        _stamp(paired)
    observed = _by_id(observations)
    other = None if paired is None else _by_id(paired)
    for case in cases:
        bare = _bare(case.node_id)
        case.node_level = LEVELS.get(bare, "")
        _reset(case)
        _pair(case, observed, other)
        case.upstream_wrong = [_tagged(node_id) for node_id in _upstream(bare, observed)]
        if case.case_origin not in {"wrong_answer", "latent"}:
            continue
        if case.parse_status in {"unparseable", "ambiguous"}:
            continue
        if validator_of(bare) == "path_order" and case.case_origin == "wrong_answer":
            _declaration_root(model, case, bare)
            continue
        if validator_of(bare) == "product_order" and case.case_origin == "wrong_answer":
            _product_slots(model, case, bare, observed, gold)
            continue
        if case.prefix == "gold" and case.case_origin == "wrong_answer":
            if case.final_class == "UNRESOLVED":
                continue
            case.failure_level = case.node_level
            case.cause_level = major(case.node_level) if case.node_level else ""
            case.evidence_level = "conditional"
            continue
        if bare in PHI and case.prefix == "model":
            _matrix(model, case, bare, observed, gold)
            continue
        if validator_of(bare) and case.prefix == "model":
            _composed(model, case, bare, observed, gold)
            continue
        if bare in ANCHOR_AXIS and case.case_origin == "wrong_answer":
            _axis(case, bare, observed, gold)
            continue
        if case.case_origin == "wrong_answer" and major(case.node_level) != "L0" and case.upstream_wrong:
            case.candidate_fault_sets = [case.upstream_wrong]
        elif case.case_origin == "wrong_answer" and major(case.node_level) == "L0":
            case.cause_level = "L0"
    cases[:] = [
        case for case in cases
        if not (case.case_origin == "latent" and case.propagation_status == "pending")
    ]
    if observations:
        from domains.wind_tunnel.conflict_diagnosis import explain_observations

        explain_observations(model, observations, gold)


def _add_composed_latent(observations: list[Observation], cases: list[Case], gold: dict[str, str]) -> None:
    present = {_bare(case.node_id) for case in cases}
    observed = _by_id(observations)
    for item in observations:
        if item.prefix != "model" or not item.correct or item.node_id in present:
            continue
        if validator_of(item.node_id) is None:
            continue
        if all(_same(_phrase(observed, gold, parent), gold[parent]) for parent in input_ids(item.node_id)):
            continue
        cases.append(Case(
            error_id=f"lat_{item.sample_id}_{item.condition}_ID{item.node_id}",
            sample_id=item.sample_id,
            condition=item.condition,
            prefix=item.prefix,
            process=PROCESS_SLUG.get(item.process, item.process),
            node_id=f"ID{item.node_id}",
            chain_id="",
            role=item.label,
            gold_answer=item.gold_answer,
            model_answer=item.model_answer,
            case_origin="latent",
            parse_status=item.parse_status,
            surface_type=_surface(item.label, item.format_code),
            final_class="UNRESOLVED",
            propagation_status="pending",
            causal_relation="unresolved",
            eligibility_for_consistency="latent_not_a_wrong_answer",
        ))


def _local_inconsistency(case: Case, bare: str) -> None:
    """The stated order is right, and the matrix slot does not follow it."""
    _bind(case, "product_order")
    case.final_class = "LOCAL_CONTRACT_INCONSISTENCY"
    case.propagation_status = "sequence_held_order_wrong"
    case.causal_relation = "local_contract_inconsistency"
    case.eligibility_for_consistency = "eligible"
    case.propagated_from = []
    case.upstream_wrong = []
    case.relations = [item for item in case.relations if item.get("relation_type") != "upstream_wrong"]
    case.minimal_fault_set = [case.node_id]
    case.mixed_fault = False
    case.candidate_fault_sets = []
    case.cause_level = major(case.node_level)
    case.primary_cause = _label(bare)
    case.secondary_causes = []
    case.failure_level = case.node_level
    case.evidence_level = "high"


def _product_slots(model, case: Case, bare: str, observed: dict[str, Observation], gold: dict[str, str]) -> None:
    """R1–R3 are the multiplication order. A correct sequence makes a wrong slot local."""
    if case.prefix == "gold":
        texts = {node_id: gold[node_id] for node_id in input_ids(bare)}
        derived = derive(model, bare, texts)
        _bind(case, "product_order")
        if derived is None or not derived.ok:
            case.final_class = "UNRESOLVED"
            case.propagation_status = "contract_contradiction"
            case.causal_relation = "unresolved"
            case.evidence_level = "none"
            return
        case.phi_matrix = derived.matrix
        if not _same(derived.matrix, case.model_answer):
            _local_inconsistency(case, bare)
        return
    _composed(model, case, bare, observed, gold)
    if case.final_class == "NEW_LOCAL_ERROR" and not case.secondary_causes:
        _local_inconsistency(case, bare)


def _declaration_root(model, case: Case, bare: str) -> None:
    """The angle sequence is a contract root. The declaration states it; no earlier blank does."""
    derived = derive(model, bare, {})
    _bind(case, "path_order")
    case.upstream_wrong = []
    case.relations = [item for item in case.relations if item.get("relation_type") != "upstream_wrong"]
    if derived is None or not derived.ok:
        case.final_class = "UNRESOLVED"
        case.propagation_status = "contract_contradiction"
        case.causal_relation = "unresolved"
        case.evidence_level = "none"
        return
    case.phi_matrix = derived.matrix
    if _same(derived.matrix, case.model_answer):
        return
    case.final_class = "ROOT_ERROR"
    case.propagation_status = "declaration_contract"
    case.causal_relation = "contract_root"
    case.eligibility_for_consistency = "eligible"
    case.propagated_from = []
    case.minimal_fault_set = [case.node_id]
    case.cause_level = "L0"
    case.primary_cause = "C-PATH-SEQUENCE"
    case.failure_level = case.node_level
    case.evidence_level = "high"


def _composed(model, case: Case, bare: str, observed: dict[str, Observation], gold: dict[str, str]) -> None:
    parents = input_ids(bare)
    validator = validator_of(bare) or ""

    def run(active: set[str]):
        texts = {
            node_id: _phrase(observed, gold, node_id) if node_id in active else gold[node_id]
            for node_id in parents
        }
        return derive(model, bare, texts)

    wrong = [node_id for node_id in parents if not _same(_phrase(observed, gold, node_id), gold[node_id])]
    full = run(set(wrong))
    if full is None or not full.ok:
        if case.case_origin == "wrong_answer":
            case.final_class = "UNRESOLVED"
            case.propagation_status = "contract_contradiction"
            case.causal_relation = "unresolved"
            case.eligibility_for_consistency = "excluded_contradiction"
            if full is not None and full.reason:
                case.findings.append({"type": "CONTRACT_CONTRADICTION", "detail": full.reason})
        return
    case.phi_matrix = full.matrix
    matched = _same(full.matrix, case.model_answer)
    if not matched:
        if case.case_origin != "wrong_answer":
            case.final_class = "CONTRACT_TARGET_DISCONNECT"
            case.propagation_status = "parent_wrong_but_not_reflected"
            case.causal_relation = "answer_matches_gold_not_phi"
            return
        _bind(case, validator)
        case.final_class = "NEW_LOCAL_ERROR"
        case.propagation_status = "not_equal_to_phi"
        case.causal_relation = "new_local"
        case.eligibility_for_consistency = "eligible"
        case.propagated_from = []
        case.minimal_fault_set = [case.node_id]
        case.cause_level = major(case.node_level)
        case.primary_cause = _label(bare)
        case.secondary_causes = [_label(node_id) for node_id in wrong]
        case.evidence_level = "high"
        return
    if not wrong:
        return
    if case.case_origin == "latent":
        _bind(case, validator)
        case.final_class = "RESULT_EQUIVALENCE"
        case.propagation_status = "verified_phi"
        case.causal_relation = "wrong_contract_same_result"
        case.minimal_fault_set = [_tagged(node_id) for node_id in wrong]
        case.mixed_fault = len(wrong) > 1
        case.evidence_level = "high"
        if len(wrong) == 1:
            case.cause_level = major(level_of(wrong[0]))
            case.primary_cause = _label(wrong[0])
        return
    minimal = _minimal_sets(
        wrong,
        lambda active: bool(run(active) and run(active).ok and _same(run(active).matrix, case.model_answer)),
    )
    _finish_propagation(case, validator, wrong, minimal)


def _minimal_sets(wrong: list[str], holds) -> list[list[str]] | None:
    if len(wrong) > 8:
        return None
    sufficient = []
    for size in range(len(wrong) + 1):
        for combo in combinations(wrong, size):
            if holds(set(combo)):
                sufficient.append(set(combo))
    return [
        sorted(item, key=int)
        for item in sufficient
        if not any(other < item for other in sufficient)
    ]


def _finish_propagation(case: Case, validator: str, wrong: list[str], minimal: list[list[str]] | None) -> None:
    if not minimal or minimal == [[]]:
        case.candidate_fault_sets = [[_tagged(node_id) for node_id in wrong]]
        return
    if len(minimal) != 1:
        _bind(case, validator)
        case.evidence_level = "none"
        case.candidate_fault_sets = [[_tagged(node_id) for node_id in item] for item in minimal]
        case.final_class = "UNRESOLVED" if case.case_origin == "wrong_answer" else case.final_class
        return
    only = minimal[0]
    _bind(case, validator)
    case.minimal_fault_set = [_tagged(node_id) for node_id in only]
    case.evidence_level = "high"
    if len(only) > 1:
        case.mixed_fault = True
        case.cause_level = ""
        case.primary_cause = ""
    else:
        case.cause_level = major(level_of(only[0]))
        case.primary_cause = _label(only[0])
        case.secondary_causes = [_label(node_id) for node_id in wrong if node_id not in only]
    if case.case_origin == "wrong_answer":
        case.final_class = "PROPAGATED_ERROR"
        case.propagation_status = "verified_phi"
        case.causal_relation = "propagated_from"
        case.propagated_from = case.minimal_fault_set
        case.eligibility_for_consistency = "eligible"


def _axis(case: Case, bare: str, observed: dict[str, Observation], gold: dict[str, str]) -> None:
    anchor = ANCHOR_AXIS[bare]
    _bind(case, "axis_follows_frame")
    if case.final_class == "PROPAGATED_ERROR":
        case.minimal_fault_set = [_tagged(anchor)]
        case.cause_level = major(level_of(anchor))
        case.primary_cause = _label(anchor)
        case.evidence_level = "high"
        return
    case.minimal_fault_set = [case.node_id]
    case.cause_level = major(case.node_level)
    case.primary_cause = _label(bare)
    if _phrase(observed, gold, anchor).strip() != gold[anchor].strip():
        case.secondary_causes = [_label(anchor)]
    case.evidence_level = "high"


def _matrix(model, case: Case, bare: str, observed: dict[str, Observation], gold: dict[str, str]) -> None:
    parents = PHI[bare][2]
    wrong, minimal = _minimal(model, bare, parents, observed, gold, case.model_answer)
    if minimal is None:
        if case.case_origin != "wrong_answer":
            return
        _bind(case, _validator(bare))
        case.minimal_fault_set = [case.node_id]
        case.cause_level = major(case.node_level)
        case.primary_cause = _label(bare)
        case.secondary_causes = [_label(node_id) for node_id in wrong]
        case.evidence_level = "high"
        return
    if case.final_class == "RESULT_EQUIVALENCE":
        _bind(case, _validator(bare))
        case.minimal_fault_set = [_tagged(node_id) for node_id in wrong]
        case.mixed_fault = len(wrong) > 1
        case.evidence_level = "high"
        if len(wrong) == 1:
            case.cause_level = major(level_of(wrong[0]))
            case.primary_cause = _label(wrong[0])
        return
    if not minimal or minimal == [[]]:
        case.candidate_fault_sets = [[_tagged(node_id) for node_id in wrong]] if wrong else []
        return
    if len(minimal) != 1:
        _bind(case, _validator(bare))
        case.phi_verified = True
        case.evidence_level = "none"
        case.candidate_fault_sets = [[_tagged(node_id) for node_id in item] for item in minimal]
        return
    only = minimal[0]
    _bind(case, _validator(bare))
    case.minimal_fault_set = [_tagged(node_id) for node_id in only]
    case.evidence_level = "high"
    if only == [bare]:
        case.cause_level = major(case.node_level)
        case.primary_cause = _label(bare)
        return
    if len(only) == 1:
        case.cause_level = major(level_of(only[0]))
        case.primary_cause = _label(only[0])
        case.secondary_causes = [_label(node_id) for node_id in wrong if node_id not in only]
        return
    case.mixed_fault = True
    case.cause_level = ""
    case.primary_cause = ""


def column_of(case: Case) -> str | None:
    if case.case_origin != "wrong_answer" or not case.node_level or major(case.node_level) == "L0":
        return None
    if case.evidence_level == "conditional":
        return "current-layer condition failed"
    if case.evidence_level == "high" and case.mixed_fault and case.minimal_fault_set:
        return "multiple faults"
    if case.evidence_level == "high" and len(case.minimal_fault_set) == 1 and case.cause_level in {"L0", "L1", "L2", "L3"}:
        return f"primary {case.cause_level}"
    return "unresolved"


def edges_of(cases: list[Case]) -> list[dict]:
    rows = []
    for case in cases:
        if case.prefix != "model" or not case.phi_verified or not case.validator_id:
            continue
        if case.final_class == "RESULT_EQUIVALENCE" and case.minimal_fault_set:
            sources = case.minimal_fault_set
            rows.append(_edge(
                ", ".join(sources),
                case,
                "the wrong contract and the target matrix are the same",
                ", ".join(_label(item) for item in sources),
            ))
            continue
        if case.evidence_level != "high" or not case.minimal_fault_set:
            continue
        if case.minimal_fault_set == [case.node_id]:
            continue
        kind = "explained by several faults together" if case.mixed_fault else "verified propagation"
        for source in case.minimal_fault_set:
            if source != case.node_id:
                rows.append(_edge(source, case, kind))
    return rows


def _edge(source: str, case: Case, kind: str, source_label: str | None = None) -> dict:
    return {
        "source": source if ", " in source else _tagged(source),
        "source_label": source_label or _label(source),
        "target": case.node_id,
        "target_label": _label(case.node_id),
        "kind": kind,
        "status": "verified",
        "validator_id": case.validator_id,
        "validator_version": case.validator_version,
    }


def _eligible(item: Observation, observed: dict[str, Observation]) -> bool:
    for node_id in closure(item.node_id):
        parent = observed.get(node_id)
        if parent is not None and not parent.correct:
            return False
    return True


def ability_rows(observations: list[Observation]) -> list[dict]:
    observed = _by_id(observations)
    rows = []
    for item in observations:
        if major(item.node_level) not in {"L1", "L2", "L3"}:
            continue
        eligible = _eligible(item, observed)
        rows.append({
            "node_id": item.node_id,
            "node_level": item.node_level,
            "raw_n": 1,
            "raw_correct": int(item.correct),
            "eligible_n": int(eligible),
            "eligible_correct": int(eligible and item.correct),
        })
    return rows


def assert_invariants(cases: list[Case], found_edges: list[dict]) -> None:
    audit_layers()
    seen = [_bare(case.node_id) for case in cases if case.case_origin == "wrong_answer"]
    if len(seen) != len(set(seen)):
        raise RuntimeError("One wrong question produced more than one case.")
    for case in cases:
        upper = bool(case.node_level) and major(case.node_level) != "L0"
        if upper and case.cause_level == "L0" and (not case.phi_verified or not case.minimal_fault_set):
            raise RuntimeError(f"{case.node_id} records the primary cause as L0, but it has no verified fault set.")
        if _bare(case.node_id) == "80" and case.case_origin == "wrong_answer" and case.cause_level == "L0":
            raise RuntimeError("ID80 cannot be attributed to L0 when it has no formula verifier.")
    for edge in found_edges:
        if not edge["validator_id"] or not edge["validator_version"]:
            raise RuntimeError("A verified fault chain is missing a verifier id or version.")
    buckets: dict[str, dict[str, int]] = {}
    for case in cases:
        column = column_of(case)
        if column is None:
            continue
        buckets.setdefault(case.node_id, {name: 0 for name in COLUMNS})
        buckets[case.node_id][column] += 1
    for node_id, counts in buckets.items():
        total = sum(1 for case in cases if case.case_origin == "wrong_answer" and case.node_id == node_id)
        if sum(counts.values()) != total:
            raise RuntimeError(f"The attribution columns of {node_id} do not sum to the number of wrong questions.")
    for node_id, expected in {
        "20": "L1-SPACE", "28": "L1-SPACE", "36": "L1-SPACE",
        "149": "L0-CDEF", "150": "L0-CDEF",
        "179": "L3-COMPENSATION-SIGN", "180": "L3-COMPENSATION-SIGN",
        "57": "L0-CPATH", "85": "L0-CPATH", "80": "L3-TARGET",
    }.items():
        if LEVELS[node_id] != expected:
            raise RuntimeError(f"The level of ID{node_id} is not {expected}.")


def render(runs: list[tuple[str, list[Observation], list[Case]]]) -> str:
    layer = audit_layers()
    lines = [
        "# Layered fault diagnosis",
        "",
        "Levels are fixed by the problem text. A primary cause is filled only when a verifier gives one sufficient fault set and the answer has no extra deviation.",
        "A wrong question under the gold prefix is current-layer condition failed and does not enter a primary-cause column. The transform order ID16-ID18 is the exception: it is derived directly from the declaration, a wrong answer under either prefix is a contract root error, and the primary cause is L0. R1, R2, and R3 are a further exception: the transform order given in the problem is right, a matrix written in the wrong place is local contract execution inconsistency, and the primary cause is that blank's own level. Each group has one sample. A cell is right or wrong for that sample, not a probability.",
        "The mechanism order stays in L0 and is derived directly from the declaration. The product-stage order stays in L2a. R1, R2, and R3 depend on a transform order that is already fixed. A right order with a product written in the wrong place is local contract execution inconsistency, not a contract root error.",
        "",
        "## Level audit",
        "",
        f"Layered nodes {layer['nodes']}. L0 {layer['L0']}, L1 {layer['L1']}, L2 {layer['L2']}, L3 {layer['L3']}. Unlayered 0, level conflicts 0.",
        "",
        "ID20, ID28, and ID36 are in L1-SPACE and do not enter the L0 contract-recognition rate. ID149 and ID150 are in L0-CDEF and do not enter L2. ID179 and ID180 are in L3. ID57 and ID85 are in L0-CPATH.",
        "",
    ]
    for name, observations, cases in runs:
        lines.extend(_run_section(name, observations, cases))
    lines.extend([
        "## Boundary",
        "",
        "Every A blank has a calculator. A blank that the model's earlier answers decide is generated from those answers. A rule or declaration written into the problem is generated from the sample declaration. The transform order uses path_order v1 and reads only the declaration. R1, R2, and R3 use product_order v1 and are generated from the answered transform order. A verifier is first checked against the gold contract.",
        "ID23 enters only the L0 ability count. Table three mentions it only when an axis-following verifier or an attitude-matrix verifier shows that the downstream answer has no extra deviation.",
        "",
    ])
    return "\n".join(lines)


def _run_section(name: str, observations: list[Observation], cases: list[Case]) -> list[str]:
    found = edges_of(cases)
    assert_invariants(cases, found)
    lines = [f"## {name}", "", "### Table 1 layered ability", ""]
    lines.extend(_level_table(observations))
    lines.extend(_joint_table(observations))
    lines.extend(_conditional_table(observations))
    lines.extend(_pair_notes(cases))
    lines.extend(_cause_table(cases))
    lines.extend(_chain_table(found))
    return lines


def _level_table(observations: list[Observation]) -> list[str]:
    lines = ["| Level | Questions | Correct |", "| --- | ---: | ---: |"]
    for level in ("L0", "L1", "L2", "L3"):
        rows = [item for item in observations if major(item.node_level) == level]
        lines.append(f"| {level} | {len(rows)} | {sum(item.correct for item in rows)} |")
    lines.append("")
    lines.append("| Split | Questions | Correct |")
    lines.append("| --- | ---: | ---: |")
    names: list[str] = []
    for item in observations:
        if item.node_level and item.node_level not in names:
            names.append(item.node_level)
    for level in names:
        rows = [item for item in observations if item.node_level == level]
        lines.append(f"| {level} | {len(rows)} | {sum(item.correct for item in rows)} |")
    lines.append("")
    return lines


def _joint_table(observations: list[Observation]) -> list[str]:
    observed = _by_id(observations)
    lines = ["Joint groups declared in advance. A blank that is not asked is not in a group.", "", "| Level | Group | Questions on the paper | All correct |", "| --- | --- | --- | --- |"]
    for layer, title, members in JOINT_GROUPS:
        present = [node_id for node_id in members if node_id in observed]
        if not present:
            continue
        ok = all(observed[node_id].correct for node_id in present)
        lines.append(f"| {layer} | {title} | {', '.join(_tagged(node_id) for node_id in present)} | {'yes' if ok else 'no'} |")
    lines.append("")
    return lines


def _conditional_table(observations: list[Observation]) -> list[str]:
    rows = ability_rows(observations)
    wrong = sorted((row for row in rows if row["raw_correct"] == 0), key=lambda item: int(item["node_id"]))
    blocked = sorted((row for row in rows if row["raw_correct"] == 1 and row["eligible_n"] == 0), key=lambda item: int(item["node_id"]))
    lines = [
        "### Table 1 conditional execution",
        "",
        "A conditional sample means every upstream question that is actually asked in the diagnosis closure was answered correctly. The table below lists only the upper questions that were originally wrong. The original correct count is 0.",
        "",
        "| Id | Level | Original samples | Original correct | Conditional samples | Conditional correct |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if not wrong:
        lines.append("| — | — | 0 | 0 | 0 | 0 |")
    for row in wrong:
        lines.append(
            f"| {_tagged(row['node_id'])} | {row['node_level']} | {row['raw_n']} | {row['raw_correct']} | "
            f"{row['eligible_n']} | {row['eligible_correct']} |"
        )
    lines.append("")
    if blocked:
        names = ", ".join(_tagged(row["node_id"]) for row in blocked)
        lines.append(f"Another {len(blocked)} upper questions are themselves correct, but their upstream is not all correct, so they do not enter the conditional sample: {names}.")
    else:
        lines.append("There is no upper question that is itself correct while its upstream is not all correct.")
    lines.append("")
    return lines


def _pair_notes(cases: list[Case]) -> list[str]:
    chosen: dict[str, Case] = {}
    for case in cases:
        if case.case_origin not in {"wrong_answer", "latent"}:
            continue
        if case.node_id not in chosen or case.prefix == "model":
            chosen[case.node_id] = case
    buckets = {("wrong", "right"): [], ("right", "wrong"): [], ("wrong", "wrong"): []}
    for node_id in sorted(chosen, key=lambda item: int(_bare(item))):
        case = chosen[node_id]
        pair = (case.model_prefix_status, case.gold_prefix_status)
        if pair in buckets:
            buckets[pair].append(case.node_id)
    lines = ["### Two prefixes", "", "This counts only the wrong questions in this prefix and pairs them with the other prefix.", ""]
    lines.append(f"model wrong, gold right: {len(buckets[('wrong', 'right')])}. These are propagation candidates. A verifier is still required before they are written as propagation.")
    if buckets[("wrong", "right")]:
        lines.append(", ".join(buckets[("wrong", "right")]))
    lines.append(f"model right, gold wrong: {len(buckets[('right', 'wrong')])}. Listed separately, and not recorded as a strong ability.")
    if buckets[("right", "wrong")]:
        lines.append(", ".join(buckets[("right", "wrong")]))
    lines.append(f"Both prefixes wrong: {len(buckets[('wrong', 'wrong')])}. This is not a proven root cause.")
    lines.append("")
    return lines


def _cause_table(cases: list[Case]) -> list[str]:
    grouped: dict[str, dict[str, int]] = {}
    levels = {}
    for case in cases:
        column = column_of(case)
        if column is None:
            continue
        grouped.setdefault(case.node_id, {name: 0 for name in COLUMNS})
        grouped[case.node_id][column] += 1
        levels[case.node_id] = case.node_level
    lines = [
        "### Table 2 target-error attribution",
        "",
        "One row is one L1, L2, or L3 wrong question. A wrong question under the gold prefix is in current-layer condition failed. A case with no high-evidence fault set is unresolved.",
        "",
        "| Id | Level | Wrong questions | " + " | ".join(COLUMNS) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in COLUMNS) + " |",
    ]
    if not grouped:
        lines.append("| — | — | 0 | " + " | ".join("0" for _ in COLUMNS) + " |")
    for node_id in sorted(grouped, key=lambda item: int(_bare(item))):
        counts = grouped[node_id]
        body = " | ".join(str(counts[name]) for name in COLUMNS)
        lines.append(f"| {node_id} | {levels[node_id]} | {sum(counts.values())} | {body} |")
    pending = [case for case in cases if column_of(case) == "unresolved"]
    if pending:
        lines.extend(["", "An unresolved question may list upstream wrong questions. That list is not the primary cause.", ""])
        for case in sorted(pending, key=lambda item: int(_bare(item.node_id))):
            upstream = ", ".join(case.upstream_wrong) or "none"
            lines.append(f"- {case.node_id} ({case.node_level}) upstream wrong questions: {upstream}")
    lines.append("")
    return lines


def _chain_table(found: list[dict]) -> list[str]:
    lines = [
        "### Table 3 verified fault chains",
        "",
        "| Root fault | Downstream | Kind | Status | Verifier | Version |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    if not found:
        lines.append("| — | — | — | — | — | — |")
    for edge in found:
        lines.append(
            f"| {edge['source']} {edge['source_label']} | {edge['target']} {edge['target_label']} | "
            f"{edge['kind']} | {edge['status']} | {edge['validator_id']} | {edge['validator_version']} |"
        )
    lines.append("")
    return lines
