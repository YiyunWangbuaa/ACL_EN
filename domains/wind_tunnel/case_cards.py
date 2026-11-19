# -*- coding: utf-8 -*-
"""Compile narrative cases into comparable case cards.

A card is one minimal chain: a definition or parent node, the direct parent
the record actually uses, and one target node. Contract answers, standard
answers and the model answer are scored element by element. The behavior
label names the output pattern. It does not name a mechanism.

Run from the repository root::

    python analysis/organize_cases.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from cloze_score import normalize, resolve_choice, sympy_equal
from domains.wind_tunnel.error_graph import NODE_BY_ID


ROOT = Path(__file__).resolve().parents[2]

LABELS = (
    "contract_consistent",
    "standard_consistent",
    "mixed_representation",
    "faithful_propagation",
    "local_recovery",
    "local_execution_failure",
    "underdetermined",
)
MECHANISM_LABELS = ("fallback", "hallucination", "reasoning failure", "reasoning_failure")
TRI = ("yes", "no", "uncertain", "not_applicable")
RELATIONS = ("contract", "standard", "both", "neither", "mixed", "uncertain")
SYMBOLIC = {"matrix", "expression", "vector"}
PREFIX_NAME = {
    "self-trunc": "model-prefix",
    "session-model": "model-prefix",
    "gold-trunc": "gold-prefix",
    "session-gold": "gold-prefix",
}
# Older pair records and the current dependency graph use different blank numbers.
# The key is the id stored in that record. The value is the current graph id.
LEGACY_TO_CURRENT = {
    "82": "85",
    "83": "86",
    "84": "87",
    "85": "88",
    "98": "104",
    "178": "184",
    "186": "192",
    "192": "198",
}
EVIDENCE_ROWS = (
    ("answers_differ", "契约答案与标准答案不同"),
    ("not_binary", "目标节点不是二选一"),
    ("direct_parent_correct", "直接父节点正确"),
    ("model_matches_standard", "模型输出标准答案"),
    ("model_misses_contract", "模型输出不符合契约"),
    ("answer_leakage", "存在题面答案泄露"),
    ("error_cancellation", "存在错误相消风险"),
    ("elementwise_check", "可逐元素验证"),
    ("local_probe_possible", "可设计局部探针"),
    ("parent_repair_possible", "可做父节点修复"),
    ("non_ancestor_repair_possible", "可做非祖先修复对照"),
    ("evidence_strength", "当前证据强度"),
)
_FORMULA = re.compile(
    r"^(?P<sign>-?)(?P<force>CA|CN|CZ)\*?"
    r"(?:\((?P<arm>[^()]+)\)|(?P<bare>[A-Za-z][A-Za-z0-9_]*))"
    r"/(?P<den>[A-Za-z][A-Za-z0-9_]*)$"
)


def canonical_id(scheme: str, source_id: str) -> str:
    """Current-graph id for a blank id taken from a record."""
    source_id = str(source_id)
    if scheme == "legacy_pair":
        return LEGACY_TO_CURRENT.get(source_id, source_id)
    return source_id


def _unwrap(text: str) -> tuple[str, bool]:
    raw = str(text or "").strip()
    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw, False
        if isinstance(payload, dict) and len(payload) == 1:
            value = next(iter(payload.values()))
            if isinstance(value, str):
                return value.strip(), True
    return raw, False


def _hidden(text: str) -> bool:
    return "隐去" in str(text or "")


def shown(raw: str, options: list | None) -> str:
    """Option text for a choice, or the extracted symbolic text."""
    if raw is None:
        return ""
    text, _wrapped = _unwrap(str(raw))
    if not text or _hidden(text):
        return ""
    opts = [item for item in (options or []) if item and item != "无"]
    if opts:
        _letter, body = resolve_choice(text, opts)
        if body:
            return str(body)
    return text.strip()


def _symbolic(node_type: str) -> bool:
    return node_type in SYMBOLIC


def texts_match(left: str, right: str, symbolic: bool) -> bool:
    if not left or not right:
        return False
    if normalize(left) == normalize(right):
        return True
    if symbolic and sympy_equal(left, right):
        return True
    return False


def relation_of(model: str, contract: str, standard: str, symbolic: bool) -> str:
    if not model or not contract or not standard:
        return "uncertain"
    hit_contract = texts_match(model, contract, symbolic)
    hit_standard = texts_match(model, standard, symbolic)
    if hit_contract and hit_standard:
        return "both"
    if hit_contract:
        return "contract"
    if hit_standard:
        return "standard"
    return "neither"


def _parser():
    import sympy

    from cloze_score import _parse_expr, _sympy_modules

    sympy_module, locals_map = _sympy_modules()
    return sympy_module, locals_map, _parse_expr


def _parse(text: str):
    sympy_module, locals_map, parse = _parser()
    return parse(text, sympy_module, locals_map)


def _sin_sign(text: str, row: int, col: int) -> str:
    try:
        parsed = _parse(text)
    except Exception:
        return "uncertain"
    sympy_module, _locals, _parse_expr = _parser()
    if not isinstance(parsed, sympy_module.MatrixBase) or parsed.shape != (3, 3):
        return "uncertain"
    expr = sympy_module.simplify(parsed[row - 1, col - 1])
    if expr == 0:
        return "0"
    sins = list(expr.atoms(sympy_module.sin))
    if len(sins) != 1:
        return "uncertain"
    coeff = sympy_module.simplify(expr / sins[0])
    if coeff == 1:
        return "+"
    if coeff == -1:
        return "-"
    return "uncertain"


def _slot_verdict(model: str, contract: str, standard: str) -> tuple[str, str]:
    if "uncertain" in {model, contract, standard} or "" in {model, contract, standard}:
        return "不确定", "uncertain"
    if contract == standard:
        return "无判别力", "not_applicable"
    if model == contract:
        return "契约一致", "contract"
    if model == standard:
        return "标准一致", "standard"
    return "两者都不是", "neither"


def _aggregate(kinds: list[str]) -> str:
    useful = [kind for kind in kinds if kind != "not_applicable"]
    if not useful:
        return "both"
    if any(kind == "uncertain" for kind in useful):
        return "uncertain"
    if all(kind == "contract" for kind in useful):
        return "contract"
    if all(kind == "standard" for kind in useful):
        return "standard"
    if "contract" in useful and "standard" in useful:
        return "mixed"
    if all(kind == "neither" for kind in useful):
        return "neither"
    return "neither"


def score_matrix(model: str, contract: str, standard: str, slots: list[dict]) -> dict:
    rows = []
    for slot in slots:
        contract_sign = _sin_sign(contract, slot["row"], slot["col"])
        standard_sign = _sin_sign(standard, slot["row"], slot["col"])
        model_sign = _sin_sign(model, slot["row"], slot["col"]) if model else "uncertain"
        verdict, kind = _slot_verdict(model_sign, contract_sign, standard_sign)
        rows.append({
            "id": slot["id"],
            "row": slot["row"],
            "col": slot["col"],
            "contract": contract_sign,
            "standard": standard_sign,
            "model": model_sign,
            "verdict": verdict,
            "kind": kind,
        })
    relation = _aggregate([row["kind"] for row in rows])
    discriminative = [row["id"] for row in rows if row["kind"] != "not_applicable"]
    checked = bool(rows) and all(row["kind"] != "uncertain" for row in rows if row["kind"] != "not_applicable")
    return {
        "rows": rows,
        "relation": relation,
        "discriminative": discriminative,
        "checked": checked,
        "full_relation": relation_of(model, contract, standard, True),
    }


def parse_formula(text: str) -> dict | None:
    extracted, wrapped = _unwrap(text)
    if not extracted or _hidden(extracted):
        return None
    normalized = normalize(extracted).replace("·", "*")
    match = _FORMULA.match(normalized)
    if not match:
        return None
    arm = match.group("arm") or match.group("bare")
    return {
        "outer_sign": "-" if match.group("sign") else "+",
        "force": match.group("force"),
        "arm": arm,
        "denominator": match.group("den"),
        "wrapped": wrapped,
        "normalized": normalized,
    }


def _arm_form(arm: str, template: str) -> str:
    if not arm or not template:
        return "uncertain"
    if sympy_equal(arm, template):
        return "template"
    if sympy_equal(arm, f"-({template})"):
        return "sign_absorbed"
    return "other"


def _format_token(parsed: dict | None, template_arm: str) -> str:
    if parsed is None:
        return "uncertain"
    form = _arm_form(parsed["arm"], template_arm)
    if form == "uncertain":
        return "uncertain"
    if form == "template":
        return "outer:" + parsed["outer_sign"]
    if form == "sign_absorbed":
        return "sign_absorbed"
    return "other"


def score_formula(model: str, contract: str, standard: str) -> dict:
    contract_parsed = parse_formula(contract)
    standard_parsed = parse_formula(standard)
    model_parsed = parse_formula(model)
    template = ""
    if contract_parsed is not None:
        template = contract_parsed["arm"]
    elif standard_parsed is not None:
        template = standard_parsed["arm"]
    values = {
        "force": (
            None if contract_parsed is None else contract_parsed["force"],
            None if standard_parsed is None else standard_parsed["force"],
            None if model_parsed is None else model_parsed["force"],
        ),
        "arm": (
            None if contract_parsed is None else contract_parsed["arm"],
            None if standard_parsed is None else standard_parsed["arm"],
            None if model_parsed is None else model_parsed["arm"],
        ),
        "outer_sign": (
            None if contract_parsed is None else contract_parsed["outer_sign"],
            None if standard_parsed is None else standard_parsed["outer_sign"],
            None if model_parsed is None else model_parsed["outer_sign"],
        ),
        "denominator": (
            None if contract_parsed is None else contract_parsed["denominator"],
            None if standard_parsed is None else standard_parsed["denominator"],
            None if model_parsed is None else model_parsed["denominator"],
        ),
        "format": (
            _format_token(contract_parsed, template),
            _format_token(standard_parsed, template),
            _format_token(model_parsed, template),
        ),
    }
    labels = {
        "force": "力系数",
        "arm": "力臂",
        "outer_sign": "整体符号",
        "denominator": "分母",
        "format": "格式",
    }
    rows = []
    for key, title in labels.items():
        contract_value, standard_value, model_value = values[key]
        shown_contract = "不确定" if contract_value is None else str(contract_value)
        shown_standard = "不确定" if standard_value is None else str(standard_value)
        shown_model = "不确定" if model_value is None else str(model_value)
        if key == "arm" and contract_value and standard_value and model_value:
            same_contract = sympy_equal(model_value, contract_value)
            same_standard = sympy_equal(model_value, standard_value)
            arms_differ = not sympy_equal(contract_value, standard_value)
            if not arms_differ:
                kind = "not_applicable" if same_contract else "neither"
                verdict = "无判别力" if same_contract else "两者都不是"
            elif same_contract and not same_standard:
                kind, verdict = "contract", "契约一致"
            elif same_standard and not same_contract:
                kind, verdict = "standard", "标准一致"
            else:
                kind, verdict = "neither", "两者都不是"
        else:
            verdict, kind = _slot_verdict(shown_model, shown_contract, shown_standard)
        rows.append({
            "id": key,
            "content": title,
            "contract": shown_contract,
            "standard": shown_standard,
            "model": shown_model,
            "verdict": verdict,
            "kind": kind,
        })
    algebraic = relation_of(model, contract, standard, True)
    parsed = contract_parsed is not None and standard_parsed is not None and model_parsed is not None
    format_kind = next(row["kind"] for row in rows if row["id"] == "format")
    format_value = next(row["model"] for row in rows if row["id"] == "format")
    format_split = parsed and format_value in {"sign_absorbed", "other"} and algebraic in {"contract", "standard"}
    if not parsed:
        relation = "uncertain"
        checked = False
    elif format_split:
        relation = "mixed"
        checked = True
    else:
        relation = _aggregate([row["kind"] for row in rows])
        checked = format_kind != "uncertain" and all(
            row["kind"] != "uncertain" for row in rows if row["kind"] != "not_applicable"
        )
    return {
        "rows": rows,
        "relation": relation,
        "checked": checked,
        "algebraic": algebraic,
        "format_split": format_split,
        "discriminative": [row["id"] for row in rows if row["kind"] != "not_applicable"],
    }


def _combine(flags: list[str]) -> str:
    if not flags:
        return "uncertain"
    if any(flag == "no" for flag in flags):
        return "no"
    if any(flag == "uncertain" for flag in flags):
        return "uncertain"
    if all(flag == "yes" for flag in flags):
        return "yes"
    return "uncertain"


def _parent_verdict(flags: list[str]) -> str:
    """A declared parent that is not an incoming edge does not answer the parent check."""
    if not flags:
        return "uncertain"
    if any(flag == "not_applicable" for flag in flags):
        return "not_applicable"
    return _combine(flags)


def _node_fields(spec: dict, blank: dict | None, standard: str) -> dict:
    options = list((blank or {}).get("options") or [])
    contract = shown((blank or {}).get("answer") or "", options)
    model_raw = (blank or {}).get("prediction") or ""
    extracted, wrapped = _unwrap(str(model_raw))
    model = "" if _hidden(extracted) else shown(extracted, options)
    node_type = spec.get("node_type") or (blank or {}).get("error_type") or ""
    symbolic = _symbolic(node_type) or _symbolic(str((blank or {}).get("answer_type") or ""))
    return {
        "id": str(spec["id"]),
        "content": spec.get("content") or (blank or {}).get("role") or "",
        "node_type": node_type,
        "decides_target": spec.get("decides_target") or "no",
        "legacy_id": spec.get("legacy_id") or "",
        "contract": contract,
        "standard": standard,
        "model": model,
        "wrapped": wrapped,
        "hidden": _hidden(str(model_raw)),
        "relation": relation_of(model, contract, standard, symbolic),
        "symbolic": symbolic,
        "record_role": (blank or {}).get("role") or "",
        "record_parents": [str(item) for item in ((blank or {}).get("depends_on") or [])],
        "options": [item for item in options if item and item != "无"],
    }


def _same_definition(contract: str, standard: str) -> bool:
    return bool(contract and standard and texts_match(contract, standard, True))


def _closeness(relation: str, contract: str, standard: str) -> str:
    if relation == "uncertain":
        return "不确定"
    if _same_definition(contract, standard):
        return "无判别力" if relation in {"both", "contract", "standard"} else "两者都不是"
    return {
        "contract": "异构契约",
        "standard": "标准定义",
        "both": "无判别力",
        "neither": "两者都不是",
        "mixed": "两者都有",
        "uncertain": "不确定",
    }[relation]


def _contract_state(relation: str, contract: str, standard: str) -> str:
    if relation == "uncertain":
        return "不确定"
    if _same_definition(contract, standard):
        return "无判别力" if relation in {"both", "contract", "standard"} else "契约错误"
    if relation in {"contract", "both"}:
        return "契约正确"
    if relation in {"standard", "neither"}:
        return "契约错误"
    return "不确定"


def behavior_label(
    output_relation: str,
    binary: str,
    direct_parent_correct: str,
    contrast_relation: str,
    format_split: bool,
) -> tuple[str, str]:
    """Neutral output label. Mechanism names are not produced here."""
    if format_split:
        return "mixed_representation", "代数结果和规定格式落在不同一侧。"
    if output_relation == "mixed":
        return "mixed_representation", "判别元素分别靠近契约和标准定义。"
    if output_relation in {"uncertain", "both"}:
        return "underdetermined", "判别元素不足以比较契约和标准定义。"
    if output_relation == "contract":
        if direct_parent_correct == "no":
            return "local_recovery", "直接父节点与契约不一致，目标节点与契约一致。"
        return "contract_consistent", "目标节点的判别元素与契约一致。"
    if output_relation == "neither":
        if direct_parent_correct == "yes":
            return "local_execution_failure", "直接父节点与契约一致，目标节点与契约和标准定义都不一致。"
        return "underdetermined", "目标节点与两种答案都不一致，父节点也没有把失败隔离出来。"
    if binary == "yes":
        return "underdetermined", "目标只有两个选项，选错以后会落到另一个选项上。"
    if contrast_relation == "standard":
        return "underdetermined", "承载契约差异的节点本身已与标准定义一致，目标节点不能单独当成一次局部执行。"
    if direct_parent_correct != "yes":
        return "underdetermined", "直接父节点没有同时具备依赖边和契约答案，目标节点虽与标准定义一致，这次输出隔离不开。"
    return "standard_consistent", "目标节点的判别元素与标准定义一致，直接父节点与契约一致。"


def pattern_strength(facts: dict) -> str:
    """How cleanly this run isolates a standard-consistent target. Not a mechanism score."""
    if facts["output_relation"] != "standard":
        return "insufficient"
    if facts["binary"] == "yes" or facts["answers_differ"] != "yes":
        return "insufficient"
    if facts["contrast_source_correct"] != "yes" or facts["direct_parent_correct"] != "yes":
        return "insufficient"
    if facts["elementwise_check"] != "yes":
        return "insufficient"
    if facts["relevant_parent_wrong"] == "yes" or facts["direct_parent_edge"] != "yes":
        return "weak"
    if facts["error_cancellation"] == "yes" or facts["error_cancellation"] == "uncertain":
        return "weak"
    if facts["answer_leakage"] == "yes":
        return "moderate"
    return "strong"


def _binary(node: dict, contract: str, standard: str) -> str:
    options = node.get("options") or []
    if len(options) != 2:
        return "no"
    if not contract or not standard or texts_match(contract, standard, False):
        return "no"
    return "yes" if {contract, standard} <= set(options) else "no"


def _leakage(spec: dict, stem: str, prompt: str) -> tuple[str, str]:
    needles = [str(item) for item in (spec.get("leakage_needles") or []) if str(item)]
    if not needles:
        return "no", spec.get("leakage_note") or "题干没有申报会给出目标答案的片段。"
    blob = stem or ""
    if spec.get("confirm_in_prompt"):
        blob = blob + "\n" + (prompt or "")
    found = [needle for needle in needles if needle in blob]
    if not found and spec.get("confirm_in_prompt") and not prompt:
        return "uncertain", "需要在作答提示里复核题面片段，但这条记录没有提示文本。"
    if found:
        note = spec.get("leakage_note") or ("题面含有：" + "、".join(found))
        return "yes", note
    return "no", "申报的题面片段没有出现在题干里。"


def _graph_node(scheme: str, source_id: str):
    current = canonical_id(scheme, source_id)
    return current, NODE_BY_ID.get(current)


def compile_case(spec: dict, observations: dict[str, dict], meta: dict | None = None) -> dict:
    """Score one case spec against one run's blanks."""
    if spec.get("forced_label") in MECHANISM_LABELS:
        raise RuntimeError(f"{spec.get('case_id')} 不能把机制词写成行为标签。")
    meta = meta or {}
    scheme = spec["source"]["id_scheme"]
    standards = {str(key): str(value) for key, value in spec["standard_answers"].items()}
    nodes = []
    by_id = {}
    for item in spec["nodes"]:
        source_id = str(item["id"])
        blank = observations.get(source_id)
        fields = _node_fields(item, blank, standards.get(source_id, ""))
        current, graph = _graph_node(scheme, source_id)
        fields["canonical_id"] = current
        fields["graph_role"] = "" if graph is None else graph.role
        fields["graph_parents"] = [] if graph is None else [f"ID{parent}" for parent in graph.depends_on]
        fields["roles_agree"] = bool(graph is not None and fields["record_role"] and fields["record_role"] == graph.role)
        nodes.append(fields)
        by_id[source_id] = fields

    target_id = str(spec["target"])
    target = by_id[target_id]
    target_blank = observations.get(target_id) or {}
    recorded_parents = list(target["record_parents"])
    declared_parents = [str(item) for item in spec["direct_parents"]]
    edge_flags = ["yes" if parent in recorded_parents else "no" for parent in declared_parents]
    direct_edge = _combine(edge_flags) if edge_flags else "uncertain"
    relevant_ids = [str(item) for item in spec.get("relevant_parents") or declared_parents]

    parent_flags = []
    for parent_id in declared_parents:
        parent = by_id.get(parent_id)
        if parent_id not in recorded_parents:
            parent_flags.append("not_applicable")
            continue
        if parent is None or parent["relation"] == "uncertain":
            parent_flags.append("uncertain")
        elif parent["relation"] in {"contract", "both"}:
            parent_flags.append("yes")
        else:
            parent_flags.append("no")
    direct_parent_correct = _parent_verdict(parent_flags)
    recorded_flags = []
    for parent_id in recorded_parents:
        parent = by_id.get(parent_id)
        if parent is None or parent["relation"] == "uncertain":
            recorded_flags.append("uncertain")
        elif parent["relation"] in {"contract", "both"}:
            recorded_flags.append("yes")
        else:
            recorded_flags.append("no")
    recorded_parent_correct = _combine(recorded_flags) if recorded_parents else "uncertain"

    relevant_flags = []
    for parent_id in relevant_ids:
        parent = by_id.get(parent_id)
        if parent is None:
            relevant_flags.append("uncertain")
        elif parent["relation"] in {"contract", "both"}:
            relevant_flags.append("yes")
        elif parent["relation"] == "uncertain":
            relevant_flags.append("uncertain")
        else:
            relevant_flags.append("no")
    relevant_correct = _combine(relevant_flags)
    relevant_wrong = "yes" if any(flag == "no" for flag in relevant_flags) else "no" if relevant_flags else "uncertain"

    contrast_ids = [str(item) for item in spec.get("contrast_sources") or declared_parents]
    contrast_flags = []
    contrast_relations = []
    for source_id in contrast_ids:
        source = by_id.get(source_id)
        if source is None:
            contrast_flags.append("uncertain")
            continue
        contrast_relations.append(source["relation"])
        if source["relation"] in {"contract", "both"}:
            contrast_flags.append("yes")
        elif source["relation"] == "uncertain":
            contrast_flags.append("uncertain")
        else:
            contrast_flags.append("no")
    contrast_correct = _combine(contrast_flags)
    contrast_relation = contrast_relations[0] if len(contrast_relations) == 1 else (
        "uncertain" if not contrast_relations else "mixed"
    )

    evidence = spec["evidence_elements"]
    element_rows = []
    format_split = False
    checked = False
    full_relation = ""
    if evidence["kind"] == "matrix_sin":
        scored = score_matrix(target["model"], target["contract"], target["standard"], evidence["slots"])
        element_rows = scored["rows"]
        output_relation = scored["relation"]
        checked = scored["checked"]
        full_relation = scored["full_relation"]
        discriminative = scored["discriminative"]
    elif evidence["kind"] == "formula":
        scored = score_formula(target["model"], target["contract"], target["standard"])
        element_rows = scored["rows"]
        output_relation = scored["relation"]
        checked = scored["checked"]
        format_split = scored["format_split"]
        full_relation = scored["algebraic"]
        discriminative = scored["discriminative"]
    else:
        raise RuntimeError(f"{spec['case_id']} 的证据类型 {evidence['kind']} 无法整理。")

    answers_differ = "uncertain"
    if target["contract"] and target["standard"]:
        answers_differ = "no" if texts_match(target["contract"], target["standard"], target["symbolic"]) else "yes"
        if evidence["kind"] == "matrix_sin":
            answers_differ = "yes" if discriminative else "no"
        if evidence["kind"] == "formula":
            answers_differ = "yes" if discriminative else "no"

    binary = _binary(target, target["contract"], target["standard"]) if evidence["kind"] != "matrix_sin" else "no"
    if evidence["kind"] == "formula":
        binary = "no"
    stem = str(spec.get("stem") or "")
    leakage, leakage_note = _leakage(spec, stem, str(target_blank.get("prompt") or ""))
    cancellation = spec.get("error_cancellation") or "uncertain"
    if cancellation not in TRI:
        raise RuntimeError(f"{spec['case_id']} 的错误相消标记无法识别。")

    parent_discriminative = "uncertain"
    if declared_parents and all(parent in by_id and by_id[parent]["contract"] and by_id[parent]["standard"] for parent in declared_parents):
        parent_discriminative = "no" if all(
            texts_match(by_id[parent]["contract"], by_id[parent]["standard"], by_id[parent]["symbolic"])
            for parent in declared_parents
        ) else "yes"

    label, reason = behavior_label(
        output_relation, binary, direct_parent_correct, contrast_relation, format_split,
    )
    if direct_edge != "yes":
        reason += "指定的直接父节点不在目标的记录依赖里。"
    if label == "standard_consistent" and parent_discriminative == "no":
        reason += "直接父节点在两种定义下答案相同。"
    if relevant_wrong == "yes" and contrast_relation != "standard":
        reason += "另有相关父节点与契约不一致。"
    if label not in LABELS:
        raise RuntimeError(f"行为标签 {label} 不在中性标签表内。")

    probe = spec["probe"]
    if probe.get("write_back"):
        raise RuntimeError(f"{spec['case_id']} 的探针答案不能写回前缀。")
    repairs = spec["repairs"]
    facts = {
        "output_relation": output_relation,
        "binary": binary,
        "answers_differ": answers_differ,
        "contrast_source_correct": contrast_correct,
        "direct_parent_correct": direct_parent_correct,
        "elementwise_check": "yes" if checked else "uncertain",
        "relevant_parent_wrong": relevant_wrong,
        "direct_parent_edge": direct_edge,
        "error_cancellation": cancellation,
        "answer_leakage": leakage,
    }
    strength = pattern_strength(facts)
    model_matches_standard = {"standard": "yes", "both": "yes"}.get(output_relation, "uncertain" if output_relation == "uncertain" else "no")
    model_misses_contract = {
        "standard": "yes",
        "neither": "yes",
        "mixed": "yes",
        "contract": "no",
        "both": "no",
        "uncertain": "uncertain",
    }[output_relation]
    evidence_matrix = {
        "answers_differ": answers_differ,
        "not_binary": "no" if binary == "yes" else "yes",
        "direct_parent_correct": direct_parent_correct,
        "model_matches_standard": model_matches_standard,
        "model_misses_contract": model_misses_contract,
        "answer_leakage": leakage,
        "error_cancellation": cancellation,
        "elementwise_check": facts["elementwise_check"],
        "local_probe_possible": "yes" if probe.get("question") and probe.get("write_back") is False else "no",
        "parent_repair_possible": "yes" if declared_parents or recorded_parents else "no",
        "non_ancestor_repair_possible": "yes" if repairs.get("non_ancestor") else "no",
        "evidence_strength": strength,
    }
    for key, value in evidence_matrix.items():
        if key == "evidence_strength":
            if value not in {"strong", "moderate", "weak", "insufficient"}:
                raise RuntimeError(f"{spec['case_id']} 的证据强度无法识别。")
        elif value not in TRI:
            raise RuntimeError(f"{spec['case_id']} 的证据项 {key} 无法识别。")

    condition = meta.get("condition") or spec["source"].get("condition") or ""
    card = {
        "case_id": spec["case_id"],
        "name": spec["name"],
        "question": spec["question"],
        "phenomenon": spec.get("phenomenon") or "",
        "selectivity_group": spec.get("selectivity_group") or "",
        "handedness_slot": spec.get("handedness_slot") or "",
        "source_file": spec["source"]["records"],
        "sample_id": spec["source"]["sample_id"],
        "paired_standard_id": spec["source"].get("paired_standard_id") or "standard-default",
        "id_scheme": scheme,
        "model": meta.get("model") or "",
        "model_version": meta.get("model_version") or "unspecified",
        "thinking": meta.get("thinking") or "unspecified",
        "record_condition": condition,
        "injection_condition": PREFIX_NAME.get(condition, condition or "unspecified"),
        "task_id": spec["source"]["sample_id"],
        "definition_version": spec["source"]["sample_id"],
        "target": target_id,
        "target_canonical": target["canonical_id"],
        "direct_parents": declared_parents,
        "recorded_parents": recorded_parents,
        "relevant_parents": relevant_ids,
        "contrast_sources": contrast_ids,
        "contract_definition": spec["contract_definition"],
        "standard_definition": spec["standard_definition"],
        "contract_answer": target["contract"],
        "standard_answer": target["standard"],
        "model_answer": target["model"],
        "output_relation": output_relation,
        "full_form_relation": full_relation,
        "format_split": format_split,
        "answers_differ": answers_differ,
        "binary": binary,
        "direct_parent_edge": direct_edge,
        "direct_parent_correct": direct_parent_correct,
        "recorded_parent_correct": recorded_parent_correct,
        "direct_parent_discriminative": parent_discriminative,
        "relevant_parent_correct": relevant_correct,
        "relevant_parent_wrong": relevant_wrong,
        "contrast_source_correct": contrast_correct,
        "contrast_source_relation": contrast_relation,
        "answer_leakage": leakage,
        "leakage_note": leakage_note,
        "error_cancellation": cancellation,
        "elements": element_rows,
        "discriminative_elements": discriminative,
        "nodes": nodes,
        "identity_ok": all(node["roles_agree"] for node in nodes),
        "label": label,
        "label_reason": reason,
        "evidence_for_fallback": "insufficient",
        "probe_pattern": "未测",
        "evidence": evidence_matrix,
        "probe": {
            "question": probe["question"],
            "after": str(probe["after"]),
            "before": str(probe["before"]),
            "write_back": False,
            "answers_recorded_only": True,
        },
        "repairs": repairs,
        "narrative_claims": spec.get("narrative_claims") or {},
        "same_chain_selective_failure": "",
        "handedness": {},
    }
    card["experiment"] = experiment_manifest(card)
    card["organization"] = organization_status(card)
    return card


def experiment_manifest(card: dict) -> dict:
    """One case as the input of the next round. Probe answers stay out of the prefix."""
    scheme = card["id_scheme"]
    target = card["target"]
    recorded = card["recorded_parents"]
    declared = card["direct_parents"]
    return {
        "definition_version": card["definition_version"],
        "task_instance": card["task_id"],
        "target_node": f"ID{card['target_canonical']}",
        "source_target_node": f"ID{target}",
        "direct_parent": [f"ID{canonical_id(scheme, parent)}" for parent in declared],
        "source_direct_parent": [f"ID{parent}" for parent in declared],
        "recorded_parent": [f"ID{canonical_id(scheme, parent)}" for parent in recorded],
        "source_recorded_parent": [f"ID{parent}" for parent in recorded],
        "contract_answer": card["contract_answer"],
        "standard_answer": card["standard_answer"],
        "model_answer": card["model_answer"],
        "prefix_condition": card["injection_condition"],
        "probe": {
            "after": f"ID{canonical_id(scheme, card['probe']['after'])}",
            "before": f"ID{canonical_id(scheme, card['probe']['before'])}",
            "question": card["probe"]["question"],
            "write_back": False,
        },
        "repairs": {
            "replay_model_prefix": {"memory": "model", "through": f"ID{canonical_id(scheme, card['probe']['after'])}"},
            "repair_declared_parent": {
                "nodes": [f"ID{canonical_id(scheme, parent)}" for parent in declared],
                "edge_recorded": card["direct_parent_edge"],
            },
            "repair_recorded_parent": {
                "nodes": [f"ID{canonical_id(scheme, parent)}" for parent in recorded],
            },
            "repair_non_ancestor": card["repairs"].get("non_ancestor") or {},
        },
        "thinking": ["disabled", "enabled"],
        "models": [card["model"]] if card["model"] else [],
        "verdict": {
            "label": card["label"],
            "evidence_strength": card["evidence"]["evidence_strength"],
            "evidence_for_fallback": card["evidence_for_fallback"],
            "probe_pattern": card["probe_pattern"],
        },
    }


def organization_status(card: dict) -> dict:
    checks = {
        "target_and_parent": bool(card["target"] and card["direct_parents"]),
        "three_answers": bool(card["contract_answer"] and card["standard_answer"] and card["model_answer"]),
        "itemized": bool(card["elements"]),
        "parent_versus_local": card["direct_parent_correct"] in TRI and card["relevant_parent_correct"] in TRI,
        "confounders": all(card[key] in TRI for key in ("answer_leakage", "binary", "error_cancellation")),
        "probe": card["probe"]["write_back"] is False and bool(card["probe"]["question"]),
        "parent_repair": "repair_recorded_parent" in card["experiment"]["repairs"],
        "strength_without_mechanism": card["label"] in LABELS and card["evidence_for_fallback"] == "insufficient",
        "experiment": {"replay_model_prefix", "repair_declared_parent", "repair_recorded_parent", "repair_non_ancestor"} <= set(card["experiment"]["repairs"]),
    }
    return {"complete": all(checks.values()), "checks": checks}


def apply_groups(cards: list[dict]) -> None:
    grouped = defaultdict(list)
    for card in cards:
        if card["selectivity_group"]:
            grouped[card["selectivity_group"]].append(card)
    for members in grouped.values():
        relations = [card["output_relation"] for card in members]
        if any(relation == "uncertain" for relation in relations):
            selective = "uncertain"
        elif "standard" in relations and "contract" in relations:
            selective = "yes"
        else:
            selective = "no"
        by_slot = {card["handedness_slot"]: card for card in members if card["handedness_slot"]}
        definition_flags = [card["contrast_source_correct"] for card in members]
        visible = _combine(definition_flags)

        def _used(slot: str) -> str:
            card = by_slot.get(slot)
            if card is None:
                return "uncertain"
            return {"contract": "yes", "standard": "no", "both": "yes"}.get(card["output_relation"], "uncertain")

        handedness = {
            "handedness_contract_visible_at_definition": visible,
            "handedness_used_in_beta_matrix": _used("beta"),
            "handedness_used_in_gamma_matrix": _used("gamma"),
            "same_chain_selective_failure": selective,
        }
        for card in members:
            card["same_chain_selective_failure"] = selective
            card["handedness"] = handedness


def check_narrative(card: dict) -> list[dict]:
    mismatches = []
    claims = card.get("narrative_claims") or {}
    nodes = {node["id"]: node for node in card["nodes"]}
    for key, expected in claims.items():
        if key == "parents":
            for parent_id, fields in expected.items():
                node = nodes.get(str(parent_id))
                if node is None:
                    mismatches.append({"claim": f"parents.{parent_id}", "expected": fields, "observed": "missing"})
                    continue
                for field, value in fields.items():
                    observed = node.get(field)
                    if str(observed) != str(value):
                        mismatches.append({
                            "claim": f"ID{parent_id}.{field}",
                            "expected": value,
                            "observed": observed,
                        })
            continue
        observed = card.get(key)
        if str(observed) != str(expected):
            mismatches.append({"claim": key, "expected": expected, "observed": observed})
    card["narrative_mismatches"] = mismatches
    return mismatches


def load_run(records_path: Path, meta_path: Path | None, sample_id: str) -> tuple[dict[str, dict], dict]:
    observations = {}
    header = {}
    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if sample_id and row.get("sample_id") != sample_id:
                continue
            header.setdefault("condition", row.get("condition"))
            header.setdefault("sample_id", row.get("sample_id"))
            for blank in row.get("blanks") or []:
                observations[str(blank["blank_id"])] = blank
    meta = {}
    if meta_path is not None and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = {**header, **meta}
    return observations, meta


def load_catalog(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compile_catalog(catalog: dict, root: Path | None = None) -> list[dict]:
    root = root or ROOT
    cache = {}
    cards = []
    for spec in catalog["cases"]:
        source = spec["source"]
        key = (source["records"], source.get("meta") or "", source["sample_id"])
        if key not in cache:
            records_path = root / source["records"]
            meta_path = root / source["meta"] if source.get("meta") else None
            cache[key] = load_run(records_path, meta_path, source["sample_id"])
        observations, meta = cache[key]
        cards.append(compile_case(spec, observations, meta))
    apply_groups(cards)
    for card in cards:
        check_narrative(card)
    comparable = len({tuple(card["evidence"]) for card in cards}) == 1
    for card in cards:
        card["organization"]["checks"]["shared_fields"] = comparable
        card["organization"]["complete"] = all(card["organization"]["checks"].values())
    return cards


def question_index(samples_path: Path, sample_id: str) -> dict[str, dict]:
    with samples_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("id") == sample_id:
                return {str(item["id"]): item for item in row.get("questions") or []}
    raise KeyError(sample_id)


def check_standard_answers(catalog: dict, samples_path: Path) -> list[str]:
    """Return mismatches between the catalog's standard answers and the paired sample."""
    cached = {}
    problems = []
    for spec in catalog["cases"]:
        sample_id = spec["source"].get("paired_standard_id") or "standard-default"
        if sample_id not in cached:
            cached[sample_id] = question_index(samples_path, sample_id)
        questions = cached[sample_id]
        scheme = spec["source"]["id_scheme"]
        node_types = {str(item["id"]): item.get("node_type") or "" for item in spec["nodes"]}
        for source_id, expected in spec["standard_answers"].items():
            current = canonical_id(scheme, str(source_id))
            question = questions.get(current)
            if question is None:
                problems.append(f"{spec['case_id']} ID{source_id} 在 {sample_id} 里没有现行编号 ID{current}")
                continue
            observed = shown(question.get("answer") or "", question.get("options") or [])
            symbolic = _symbolic(node_types.get(str(source_id), "")) or question.get("format_code") in {"矩阵", "向量", "表达式", "力臂"}
            if not texts_match(str(expected), observed, symbolic):
                problems.append(f"{spec['case_id']} ID{source_id} 标准答案与 {sample_id} ID{current} 不一致")
    return problems


def dump_yaml(value, indent: int = 0) -> str:
    pad = "  " * indent
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        if not value:
            return "[]"
        return "\n".join(f"{pad}- {dump_yaml(item, indent + 1).lstrip()}" if not isinstance(item, (dict, list))
                         else f"{pad}-\n{dump_yaml(item, indent + 1)}" for item in value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for key, item in value.items():
            rendered = dump_yaml(item, indent + 1)
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}{key}:\n{rendered}")
            else:
                lines.append(f"{pad}{key}: {rendered}")
        return "\n".join(lines)
    return json.dumps(str(value), ensure_ascii=False)


def _id_label(node: dict) -> str:
    source = node["id"]
    notes = []
    if node.get("legacy_id"):
        notes.append(f"旧编号 ID{node['legacy_id']}")
    if node["canonical_id"] != source:
        notes.append(f"现行 ID{node['canonical_id']}")
    if not notes:
        return f"ID{source}"
    return f"ID{source}（{'，'.join(notes)}）"


def _tri_zh(value: str) -> str:
    return {
        "yes": "yes",
        "no": "no",
        "uncertain": "uncertain",
        "not_applicable": "not applicable",
        "strong": "strong",
        "moderate": "moderate",
        "weak": "weak",
        "insufficient": "insufficient",
    }.get(value, value)


def _strength_zh(value: str) -> str:
    return {"insufficient": "不足", "weak": "弱", "moderate": "中等", "strong": "强"}.get(value, value)


def _shown_cell(value: str) -> str:
    return {
        "+": "正",
        "-": "负",
        "0": "0",
        "outer:+": "外侧为正",
        "outer:-": "外侧为负",
        "sign_absorbed": "负号进入力臂",
        "other": "其他格式",
        "uncertain": "不确定",
    }.get(value, value)


def render_report(cards: list[dict]) -> str:
    lines = [
        "# 案例卡",
        "",
        "每张卡只保留一条最小链。契约答案来自该次记录里的题面答案，标准答案来自配对的统一默认样本，模型答案来自同一次记录。",
        "表中的父节点编号沿用该次记录。C1 来自旧配对记录，现行编号写在节点名后。C2、C3 使用当前编号，力矩节点另标旧编号。",
        "行为标签只描述这次输出落在哪一类。证据强度描述这次输出能不能把局部执行隔离出来。",
        "",
        "## 证据矩阵",
        "",
    ]
    header = "| 证据项 | " + " | ".join(card["case_id"] for card in cards) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in cards) + " |"
    lines.extend([header, rule])
    for key, title in EVIDENCE_ROWS:
        cells = " | ".join(_tri_zh(card["evidence"][key]) for card in cards)
        lines.append(f"| {title} | {cells} |")
    lines.extend(["", "## 案例", ""])
    for card in cards:
        lines.extend(_render_card(card))
    lines.extend(["## 分组字段", ""])
    seen = set()
    for card in cards:
        if not card["selectivity_group"] or card["selectivity_group"] in seen:
            continue
        seen.add(card["selectivity_group"])
        lines.append(f"### {card['selectivity_group']}")
        lines.append("")
        for key, value in card["handedness"].items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    complete = all(card["organization"]["complete"] for card in cards)
    lines.extend([
        "## 整理状态",
        "",
        "这些案例使用同一组证据字段。" if complete else "整理尚未齐。",
        "下一步按实验单元重放前缀、修复父节点，探针答案留在旁路。",
        "",
    ])
    return "\n".join(lines)


def _render_card(card: dict) -> list[str]:
    thinking = {"enabled": "enabled", "disabled": "disabled"}.get(card["thinking"], card["thinking"])
    lines = [
        f"### {card['case_id']} {card['name']}",
        "",
        "研究问题：" + card["question"],
        "",
        "```text",
        f"case_id: {card['case_id']}",
        f"source_file: {card['source_file']}",
        f"model: {card['model']}",
        f"model_version: {card['model_version']}",
        f"thinking: {thinking}",
        f"injection_condition: {card['injection_condition']}",
        f"task_id: {card['task_id']}",
        f"id_scheme: {card['id_scheme']}",
        f"target: ID{card['target']} -> current ID{card['target_canonical']}",
        f"declared_direct_parent: {', '.join('ID' + item for item in card['direct_parents']) or '无'}",
        f"recorded_parent: {', '.join('ID' + item for item in card['recorded_parents']) or '无'}",
        "```",
        "",
    ]
    if not card["identity_ok"]:
        bad = [f"ID{node['id']}" for node in card["nodes"] if not node["roles_agree"]]
        lines.append("编号和现行依赖图的角色没有对齐：" + "、".join(bad))
        lines.append("")
    lines.extend([
        "契约定义：" + card["contract_definition"],
        "",
        "标准定义：" + card["standard_definition"],
        "",
        "| 节点 | 内容 | 类型 | 决定目标 | 模型答案 | 契约状态 | 更接近 | 记录父节点 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    decides = {"yes": "是", "no": "否", "indirect": "间接"}
    for node in card["nodes"]:
        parents = "、".join("ID" + item for item in node["record_parents"]) or "无"
        lines.append(
            "| {idn} | {content} | {kind} | {decides} | {model} | {state} | {close} | {parents} |".format(
                idn=_id_label(node),
                content=node["content"],
                kind=node["node_type"],
                decides=decides.get(node["decides_target"], node["decides_target"]),
                model=(node["model"] or "不确定") + ("（外层不是规定格式）" if node["wrapped"] else ""),
                state=_contract_state(node["relation"], node["contract"], node["standard"]),
                close=_closeness(node["relation"], node["contract"], node["standard"]),
                parents=parents,
            )
        )
    lines.extend([
        "",
        "| 元素 | 契约答案 | 标准答案 | 模型答案 | 判定 |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in card["elements"]:
        lines.append(
            f"| {row.get('content') or row['id']} | {_shown_cell(row['contract'])} | "
            f"{_shown_cell(row['standard'])} | {_shown_cell(row['model'])} | {row['verdict']} |"
        )
    if card["full_form_relation"]:
        lines.extend(["", f"整式对照（不替代逐项判定）：{card['full_form_relation']}"])
    lines.extend([
        "",
        f"行为标签：{card['label']}。{card['label_reason']}",
        f"证据强度：{_strength_zh(card['evidence']['evidence_strength'])}。回退证据：不足。",
        f"题面泄露：{card['answer_leakage']}。{card['leakage_note']}",
        f"指定父节点是否记录依赖：{card['direct_parent_edge']}。指定父节点是否与契约一致：{card['direct_parent_correct']}。",
        f"记录父节点是否与契约一致：{card['recorded_parent_correct']}。",
        f"相关父节点是否与契约一致：{card['relevant_parent_correct']}。",
        "",
        "探针（答案不写回）：" + card["probe"]["question"],
        "",
    ])
    if card["narrative_mismatches"]:
        lines.append("叙述核对：")
        lines.append("")
        for item in card["narrative_mismatches"]:
            lines.append(f"- {item['claim']}：叙述为 {item['expected']}，记录为 {item['observed']}")
        lines.append("")
    else:
        lines.extend(["叙述核对：申报的叙述与这次记录一致。", ""])
    experiment = card["experiment"]
    lines.extend([
        "```yaml",
        dump_yaml({
            "case_id": card["case_id"],
            "target_node": experiment["target_node"],
            "source_target_node": experiment["source_target_node"],
            "recorded_parent": experiment["recorded_parent"],
            "declared_parent": experiment["direct_parent"],
            "label": card["label"],
            "evidence_strength": card["evidence"]["evidence_strength"],
            "required_probe": experiment["probe"],
        }),
        "```",
        "",
    ])
    return lines


def write_outputs(cards: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cases": cards}
    (out_dir / "cards.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "cards.yaml").write_text(dump_yaml(payload) + "\n", encoding="utf-8")
    (out_dir / "experiments.json").write_text(
        json.dumps([card["experiment"] for card in cards], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_report(cards), encoding="utf-8")
