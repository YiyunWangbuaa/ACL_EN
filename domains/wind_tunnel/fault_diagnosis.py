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
COLUMNS = ("主因L0", "主因L1", "主因L2", "主因L3", "多故障", "当前层条件失败", "未解决")


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
    return label or "未标注"


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
        return "未配对"
    return "对" if item.correct else "错"


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
        return "当前层条件失败"
    if case.evidence_level == "high" and case.mixed_fault and case.minimal_fault_set:
        return "多故障"
    if case.evidence_level == "high" and len(case.minimal_fault_set) == 1 and case.cause_level in {"L0", "L1", "L2", "L3"}:
        return f"主因{case.cause_level}"
    return "未解决"


def edges_of(cases: list[Case]) -> list[dict]:
    rows = []
    for case in cases:
        if case.prefix != "model" or not case.phi_verified or not case.validator_id:
            continue
        if case.final_class == "RESULT_EQUIVALENCE" and case.minimal_fault_set:
            sources = case.minimal_fault_set
            rows.append(_edge(
                "、".join(sources),
                case,
                "错误契约与目标矩阵相同",
                "、".join(_label(item) for item in sources),
            ))
            continue
        if case.evidence_level != "high" or not case.minimal_fault_set:
            continue
        if case.minimal_fault_set == [case.node_id]:
            continue
        kind = "多故障共同解释" if case.mixed_fault else "已验证传播"
        for source in case.minimal_fault_set:
            if source != case.node_id:
                rows.append(_edge(source, case, kind))
    return rows


def _edge(source: str, case: Case, kind: str, source_label: str | None = None) -> dict:
    return {
        "source": source if "、" in source else _tagged(source),
        "source_label": source_label or _label(source),
        "target": case.node_id,
        "target_label": _label(case.node_id),
        "kind": kind,
        "status": "已验证",
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
        raise RuntimeError("同一道错题出现了多条案例。")
    for case in cases:
        upper = bool(case.node_level) and major(case.node_level) != "L0"
        if upper and case.cause_level == "L0" and (not case.phi_verified or not case.minimal_fault_set):
            raise RuntimeError(f"{case.node_id} 把主因写成 L0，但没有已验证的故障集。")
        if _bare(case.node_id) == "80" and case.case_origin == "wrong_answer" and case.cause_level == "L0":
            raise RuntimeError("ID80 在没有公式验证器时不能归因到 L0。")
    for edge in found_edges:
        if not edge["validator_id"] or not edge["validator_version"]:
            raise RuntimeError("已验证故障链缺少验证器编号或版本。")
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
            raise RuntimeError(f"{node_id} 的归因列之和不等于错题数。")
    for node_id, expected in {
        "20": "L1-SPACE", "28": "L1-SPACE", "36": "L1-SPACE",
        "149": "L0-CDEF", "150": "L0-CDEF",
        "179": "L3-COMPENSATION-SIGN", "180": "L3-COMPENSATION-SIGN",
        "57": "L0-CPATH", "85": "L0-CPATH", "80": "L3-TARGET",
    }.items():
        if LEVELS[node_id] != expected:
            raise RuntimeError(f"ID{node_id} 的层级不是 {expected}。")


def render(runs: list[tuple[str, list[Observation], list[Case]]]) -> str:
    layer = audit_layers()
    lines = [
        "# 分层故障诊断",
        "",
        "层级按题面固定。主因只在验证器给出唯一充分故障集，并且答案没有额外偏离时填写。",
        "gold 前缀下的错题记为当前层条件失败，不进入主因列。变换顺序 ID16–ID18 除外：它由定义声明直接推出，两种前缀下答错都记为契约根错误，主因在 L0。R1、R2、R3 再除外：题面给出的变换顺序是对的，矩阵位置写错记为契约局部执行不一致，主因在该空自己的层级。每组一个样本，表中的数是这一次的对错，不是概率。",
        "机构顺序留在 L0，由定义声明直接推出。连乘阶段顺序留在 L2a。R1、R2、R3 依赖已经确定的变换顺序；顺序答对而乘法位置写错，是契约局部执行不一致，不是契约根错误。",
        "",
        "## 层级审计",
        "",
        f"可分层节点 {layer['nodes']}。L0 {layer['L0']}，L1 {layer['L1']}，L2 {layer['L2']}，L3 {layer['L3']}。未分层 0，层级冲突 0。",
        "",
        "ID20、ID28、ID36 在 L1-SPACE，不进入 L0 契约识别率。ID149、ID150 在 L0-CDEF，不进入 L2。ID179、ID180 在 L3。ID57、ID85 在 L0-CPATH。",
        "",
    ]
    for name, observations, cases in runs:
        lines.extend(_run_section(name, observations, cases))
    lines.extend([
        "## 边界",
        "",
        "A 类空都有计算器。能由模型前置答案决定的，按那些答案生成；题面写死的规则和声明按样本声明生成。变换顺序使用 path_order v1，只读定义声明。R1、R2、R3 使用 product_order v1，按已答的变换顺序生成。验证器先用金标契约复核。",
        "ID23 只进入 L0 能力统计。表三提到它，只发生在轴跟随或姿态矩阵验证器证明下游没有额外偏离的时候。",
        "",
    ])
    return "\n".join(lines)


def _run_section(name: str, observations: list[Observation], cases: list[Case]) -> list[str]:
    found = edges_of(cases)
    assert_invariants(cases, found)
    lines = [f"## {name}", "", "### 表一 分层能力", ""]
    lines.extend(_level_table(observations))
    lines.extend(_joint_table(observations))
    lines.extend(_conditional_table(observations))
    lines.extend(_pair_notes(cases))
    lines.extend(_cause_table(cases))
    lines.extend(_chain_table(found))
    return lines


def _level_table(observations: list[Observation]) -> list[str]:
    lines = ["| 层级 | 设问数 | 正确数 |", "| --- | ---: | ---: |"]
    for level in ("L0", "L1", "L2", "L3"):
        rows = [item for item in observations if major(item.node_level) == level]
        lines.append(f"| {level} | {len(rows)} | {sum(item.correct for item in rows)} |")
    lines.append("")
    lines.append("| 细分 | 设问数 | 正确数 |")
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
    lines = ["预先声明的联合组。没有设问的空不在组内。", "", "| 层 | 组 | 在卷题 | 全对 |", "| --- | --- | --- | --- |"]
    for layer, title, members in JOINT_GROUPS:
        present = [node_id for node_id in members if node_id in observed]
        if not present:
            continue
        ok = all(observed[node_id].correct for node_id in present)
        lines.append(f"| {layer} | {title} | {', '.join(_tagged(node_id) for node_id in present)} | {'是' if ok else '否'} |")
    lines.append("")
    return lines


def _conditional_table(observations: list[Observation]) -> list[str]:
    rows = ability_rows(observations)
    wrong = sorted((row for row in rows if row["raw_correct"] == 0), key=lambda item: int(item["node_id"]))
    blocked = sorted((row for row in rows if row["raw_correct"] == 1 and row["eligible_n"] == 0), key=lambda item: int(item["node_id"]))
    lines = [
        "### 表一 条件执行",
        "",
        "条件样本指诊断闭包里实际设问的上游全部答对。下表只列原始答错的上层题。原始正确数为 0。",
        "",
        "| 题号 | 层级 | 原始样本数 | 原始正确数 | 条件样本数 | 条件正确数 |",
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
        names = "、".join(_tagged(row["node_id"]) for row in blocked)
        lines.append(f"另有 {len(blocked)} 道上层题本身答对，但上游未全对，不进入条件样本：{names}。")
    else:
        lines.append("没有「本身答对、但上游未全对」的上层题。")
    lines.append("")
    return lines


def _pair_notes(cases: list[Case]) -> list[str]:
    chosen: dict[str, Case] = {}
    for case in cases:
        if case.case_origin not in {"wrong_answer", "latent"}:
            continue
        if case.node_id not in chosen or case.prefix == "model":
            chosen[case.node_id] = case
    buckets = {("错", "对"): [], ("对", "错"): [], ("错", "错"): []}
    for node_id in sorted(chosen, key=lambda item: int(_bare(item))):
        case = chosen[node_id]
        pair = (case.model_prefix_status, case.gold_prefix_status)
        if pair in buckets:
            buckets[pair].append(case.node_id)
    lines = ["### 两种前缀", "", "这里只统计本前缀记录里的错题，并配上另一前缀的对错。", ""]
    lines.append(f"model 错、gold 对：{len(buckets[('错', '对')])}。这是传播候选，仍要验证器才能写成传播。")
    if buckets[("错", "对")]:
        lines.append("、".join(buckets[("错", "对")]))
    lines.append(f"model 对、gold 错：{len(buckets[('对', '错')])}。单独列出，不记成能力好。")
    if buckets[("对", "错")]:
        lines.append("、".join(buckets[("对", "错")]))
    lines.append(f"两种前缀都错：{len(buckets[('错', '错')])}。这不是已经证明的根因。")
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
        "### 表二 目标错误归因",
        "",
        "一行一道 L1、L2 或 L3 错题。gold 前缀的错题在「当前层条件失败」。没有高证据故障集的案例在「未解决」。",
        "",
        "| 题号 | 层级 | 错题数 | " + " | ".join(COLUMNS) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in COLUMNS) + " |",
    ]
    if not grouped:
        lines.append("| — | — | 0 | " + " | ".join("0" for _ in COLUMNS) + " |")
    for node_id in sorted(grouped, key=lambda item: int(_bare(item))):
        counts = grouped[node_id]
        body = " | ".join(str(counts[name]) for name in COLUMNS)
        lines.append(f"| {node_id} | {levels[node_id]} | {sum(counts.values())} | {body} |")
    pending = [case for case in cases if column_of(case) == "未解决"]
    if pending:
        lines.extend(["", "未解决的题可以列出上游错题，这些名单不是主因。", ""])
        for case in sorted(pending, key=lambda item: int(_bare(item.node_id))):
            upstream = "、".join(case.upstream_wrong) or "无"
            lines.append(f"- {case.node_id}（{case.node_level}）上游错题：{upstream}")
    lines.append("")
    return lines


def _chain_table(found: list[dict]) -> list[str]:
    lines = [
        "### 表三 已验证故障链",
        "",
        "| 根故障 | 下游 | 类型 | 状态 | 验证器 | 版本 |",
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
