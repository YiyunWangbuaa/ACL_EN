# -*- coding: utf-8 -*-
"""Diagnose one finished wind-tunnel run.

The order is fixed: score the answers, label the four node states, find the
first wrong nodes, then judge whether each answer edge faithfully followed
the parents the model actually saw, and only then read the contract layer.

A wrong descendant is not a contract-execution failure. It may be faithful
propagation, a second local mistake, or a recovery. ``error_type`` stays the
answer category (matrix, expression, sign). The run state is ``state``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from cloze_score import resolve_choice
from domains.wind_tunnel.contract_graph import contract_layer
from domains.wind_tunnel.error_graph import (
    NODE_BY_ID,
    _observation_correct,
    _prefix_mode,
    classify_errors,
)


_MATRIX_OF_NAME = {
    "RP1": "8",
    "RP2": "13",
    "R_α_M": "26",
    "R_β_M": "34",
    "R_γ_M": "43",
    "R_(γ_M-γ_tp)": "88",
    "RTP_TZ1": "95",
    "RTP_TZ2": "97",
}
_COMPONENT_OF_NAME = {"XTZ": "59", "YTZ": "60", "ZTZ": "61"}
_AXIS_CELLS = {
    "X": ((1, 2), (2, 1)),
    "Y": ((0, 2), (2, 0)),
    "Z": ((0, 1), (1, 0)),
}
_SYMBOL_NAMES = (
    ("alpha_M", "α_M"), ("beta_M", "β_M"), ("gamma_M", "γ_M"),
    ("gamma_tp", "γ_tp"), ("alpha_t", "α_t"), ("beta_t", "β_t"),
    ("alpha_e", "α_e"), ("beta_e", "β_e"), ("gamma_e", "γ_e"),
    ("alpha", "α"), ("beta", "β"), ("gamma", "γ"),
)


def _layer() -> dict:
    if not hasattr(_layer, "value"):
        _layer.value = contract_layer()
    return _layer.value


def _options(blank: dict) -> list[str]:
    return [item for item in (blank.get("options") or []) if item and item != "none"]


def _wording(blank: dict, raw: str) -> str:
    text = str(raw or "").strip()
    options = _options(blank)
    if not text or not options:
        return text
    _letter, body = resolve_choice(text, options)
    return body or text


def _gold_text(blank: dict) -> str:
    return _wording(blank, str(blank.get("answer") or ""))


def _prediction_text(blank: dict) -> str:
    return _wording(blank, str(blank.get("prediction") or ""))


def _visible_text(blank: dict, prefix: str) -> str:
    """Answer text the next node was allowed to see."""
    if prefix == "gold":
        return _gold_text(blank)
    return _prediction_text(blank)


def _parser():
    from cloze_score import _is_zero, _parse_expr, _sympy_modules

    sympy, locals_map = _sympy_modules()
    return sympy, locals_map, _parse_expr, _is_zero


def _parse(text: str):
    sympy, locals_map, parse, _is_zero = _parser()
    return parse(text, sympy, locals_map)


def _equal(expr, text: str) -> bool:
    if expr is None or not str(text or "").strip():
        return False
    sympy, _locals, _parse_expr, is_zero = _parser()
    try:
        return bool(is_zero(expr - _parse(text), sympy))
    except Exception:
        return False


def _show(expr) -> str:
    sympy, _locals, _parse_expr, _is_zero = _parser()
    if isinstance(expr, sympy.MatrixBase):
        rows = []
        for row in range(expr.rows):
            cells = ", ".join(_show(expr[row, column]) for column in range(expr.cols))
            rows.append(f"[{cells}]")
        return "[" + ", ".join(rows) + "]"
    rendered = str(sympy.simplify(expr))
    for source, target in _SYMBOL_NAMES:
        rendered = rendered.replace(source, target)
    return rendered


def _cell_sign(expr) -> str:
    sympy, _locals, _parse_expr, is_zero = _parser()
    value = sympy.simplify(expr)
    if is_zero(value, sympy):
        return "0"
    if getattr(value, "could_extract_minus_sign", lambda: False)():
        return "-"
    return "+"


def _matrix_fields(gold_text: str, prediction_text: str, standard: dict) -> dict:
    axis = str(standard.get("axis") or "")
    cells = _AXIS_CELLS.get(axis)
    if cells is None:
        return {}
    try:
        gold = _parse(gold_text)
        prediction = _parse(prediction_text) if prediction_text else None
    except Exception:
        return {}
    sympy, _locals, _parse_expr, is_zero = _parser()
    if not isinstance(gold, sympy.MatrixBase) or gold.shape != (3, 3):
        return {}
    found = {}
    upper, lower = cells
    found["upper_off_diagonal_sign"] = {
        "gold": _cell_sign(gold[upper]),
        "prediction": None if prediction is None else _cell_sign(prediction[upper]),
    }
    found["lower_off_diagonal_sign"] = {
        "gold": _cell_sign(gold[lower]),
        "prediction": None if prediction is None else _cell_sign(prediction[lower]),
    }
    diagonal_match = prediction is not None and all(
        is_zero(gold[index, index] - prediction[index, index], sympy) for index in range(3)
    )
    found["diagonal_pattern"] = {
        "gold": ",".join(_show(gold[index, index]) for index in range(3)),
        "prediction": None if prediction is None else ",".join(_show(prediction[index, index]) for index in range(3)),
        "match": diagonal_match,
    }
    off = []
    for row in range(3):
        for column in range(3):
            if row == column or is_zero(gold[row, column], sympy):
                continue
            same = prediction is not None and is_zero(gold[row, column] - prediction[row, column], sympy)
            off.append(bool(same))
    if off:
        found["off_diagonal_sign"] = {
            "gold": "match",
            "prediction": "match" if off and all(off) else "changed",
            "match": bool(off) and all(off),
        }
    for name in ("upper_off_diagonal_sign", "lower_off_diagonal_sign"):
        slot = found[name]
        slot["match"] = slot["prediction"] is not None and slot["prediction"] == slot["gold"]
    return found


def _named_product(observations: dict, prefix: str, slots: tuple[str, ...]):
    matrices = []
    for slot in slots:
        blank = observations.get(slot)
        if blank is None:
            return None
        name = _visible_text(blank, prefix)
        target = _MATRIX_OF_NAME.get(name)
        matrix_blank = observations.get(target) if target else None
        if matrix_blank is None:
            return None
        try:
            matrices.append(_parse(_visible_text(matrix_blank, prefix)))
        except Exception:
            return None
    if len(matrices) != len(slots):
        return None
    product = matrices[0]
    for item in matrices[1:]:
        product = product * item
    return product


def _component_expr(observations: dict, prefix: str, name: str):
    if name == "1":
        sympy, _locals, _parse_expr, _is_zero = _parser()
        return sympy.Integer(1)
    blank_id = _COMPONENT_OF_NAME.get(name)
    blank = observations.get(blank_id) if blank_id else None
    if blank is None:
        return None
    return _parse(_visible_text(blank, prefix))


def _signed(sign: str, expr):
    if sign == "positive":
        return expr
    if sign == "negative":
        return -expr
    return None


def _angle_formula(observations: dict, prefix: str, spec: dict):
    try:
        outer = _visible_text(observations[spec["outer"]], prefix)
        inverse = _visible_text(observations[spec["inverse"]], prefix)
        num_sign = _visible_text(observations[spec["num_sign"]], prefix)
        num_name = _visible_text(observations[spec["num_name"]], prefix)
        den_sign = _visible_text(observations[spec["den_sign"]], prefix)
        den_name = _visible_text(observations[spec["den_name"]], prefix)
        numerator = _signed(num_sign, _component_expr(observations, prefix, num_name))
        denominator = _signed(den_sign, _component_expr(observations, prefix, den_name))
    except Exception:
        return None
    if numerator is None or denominator is None or inverse not in {"arctan", "arcsin"}:
        return None
    sympy, _locals, _parse_expr, is_zero = _parser()
    if den_name == "1" and den_sign == "positive":
        interior = numerator
    else:
        if is_zero(denominator, sympy):
            return None
        interior = numerator / denominator
    body = sympy.atan(interior) if inverse == "arctan" else sympy.asin(interior)
    if outer == "negative":
        body = -body
    elif outer != "positive":
        return None
    return body


_TRANSFERS = {
    "55": lambda observations, prefix: _named_product(observations, prefix, ("47", "48", "49")),
    "59": lambda observations, prefix: _projected(observations, prefix, 0),
    "60": lambda observations, prefix: _projected(observations, prefix, 1),
    "61": lambda observations, prefix: _projected(observations, prefix, 2),
    "80": lambda observations, prefix: _angle_formula(observations, prefix, {
        "outer": "68", "inverse": "69", "num_sign": "70", "num_name": "71",
        "den_sign": "72", "den_name": "73",
    }),
    "81": lambda observations, prefix: _angle_formula(observations, prefix, {
        "outer": "74", "inverse": "75", "num_sign": "76", "num_name": "77",
        "den_sign": "78", "den_name": "79",
    }),
    "104": lambda observations, prefix: _named_product(observations, prefix, ("98", "99", "100")),
}


def _projected(observations: dict, prefix: str, index: int):
    matrix_blank = observations.get("55")
    vector_blank = observations.get("58")
    if matrix_blank is None or vector_blank is None:
        return None
    try:
        matrix = _parse(_visible_text(matrix_blank, prefix))
        vector = _parse(_visible_text(vector_blank, prefix))
    except Exception:
        return None
    sympy, _locals, _parse_expr, _is_zero = _parser()
    if not isinstance(matrix, sympy.MatrixBase) or not isinstance(vector, sympy.MatrixBase):
        return None
    column = vector.reshape(3, 1) if vector.shape == (1, 3) else vector
    if column.shape != (3, 1) or matrix.shape != (3, 3):
        return None
    return sympy.simplify((matrix * column)[index])


def _recompute(blank_id: str, observations: dict, prefix: str):
    transfer = _TRANSFERS.get(blank_id)
    if transfer is None:
        return None
    try:
        return transfer(observations, prefix)
    except Exception:
        return None


def _sensitive(blank_id: str, blank: dict, bindings: dict) -> dict:
    rule = bindings.get(f"ID{blank_id}", {})
    names = list(rule.get("sensitive_fields") or [])
    if not names:
        return {}
    gold = _gold_text(blank)
    prediction = _prediction_text(blank)
    standard = {}
    for binding in _layer()["contract_bindings"]:
        if f"ID{blank_id}" in binding["standard_sensitive"]:
            standard.update(binding["standard_sensitive"][f"ID{blank_id}"])
    extracted = _matrix_fields(gold, prediction, standard) if standard.get("axis") else {}
    found = {}
    for name in names:
        if name in extracted:
            found[name] = extracted[name]
            continue
        found[name] = {
            "gold": gold,
            "prediction": prediction,
            "match": bool(_observation_correct(blank)),
        }
    return found


def _execution_ok(fields: dict, correct: bool | None) -> bool | None:
    if correct is None:
        return None
    if not fields:
        return correct
    matches = [item.get("match") for item in fields.values()]
    if any(item is False for item in matches):
        return False
    if any(item is None for item in matches):
        return correct
    return True


def _transition(parent_wrong: bool, child_correct: bool | None, matched: bool | None, available: bool) -> dict:
    if child_correct is None:
        return {"faithful_transition": None, "transition_status": "unobserved", "error_path_active": False}
    if child_correct and not parent_wrong:
        return {"faithful_transition": True if matched else None, "transition_status": "n/a", "error_path_active": False}
    if child_correct and parent_wrong:
        return {
            "faithful_transition": False if matched is False else None,
            "transition_status": "recovered",
            "error_path_active": False,
        }
    if not available or matched is None:
        if not parent_wrong:
            return {"faithful_transition": False, "transition_status": "local_failure", "error_path_active": False}
        return {"faithful_transition": None, "transition_status": "not_recomputed", "error_path_active": None}
    if matched:
        return {
            "faithful_transition": True,
            "transition_status": "faithful_propagation",
            "error_path_active": parent_wrong,
        }
    return {
        "faithful_transition": False,
        "transition_status": "local_transition_failure",
        "error_path_active": False,
    }


def _node_transition(kinds: list[str], state: str) -> str:
    if state == "correct":
        return "n/a"
    if state == "recovered":
        return "recovered"
    if "local_transition_failure" in kinds or "local_failure" in kinds:
        return "local_transition_failure"
    if state == "propagated_error" and kinds and all(item == "faithful_propagation" for item in kinds):
        return "faithful_propagation"
    if state == "propagated_error" and "not_recomputed" in kinds:
        return "not_recomputed"
    if state == "local_error":
        return "local_failure"
    return "not_recomputed"


def diagnose_run(observations: dict, sample_id: str, condition: str) -> dict:
    """One sample, one condition. Observations are scored blank dicts keyed by id."""
    prefix = _prefix_mode(condition)
    classified = classify_errors(observations, prefix)
    by_state = {item["target_id"]: item for item in classified["nodes"]}
    layer = _layer()
    bindings = layer["answer_bindings"]
    direct_edges = [
        item for item in layer["direct_edges"]
        if item["target"].removeprefix("ID") in observations
    ]
    direct_of = defaultdict(list)
    for edge in direct_edges:
        direct_of[edge["target"].removeprefix("ID")].append(edge["source"])

    recomputed = {}
    matched = {}
    for blank_id in observations:
        expr = _recompute(blank_id, observations, prefix)
        recomputed[blank_id] = expr
        if expr is None:
            continue
        matched[blank_id] = _equal(expr, _prediction_text(observations[blank_id]))

    nodes = []
    edges = []
    incoming = defaultdict(list)
    for blank_id in sorted((item for item in observations if item in NODE_BY_ID), key=int):
        blank = observations[blank_id]
        node = NODE_BY_ID[blank_id]
        state_row = by_state.get(f"ID{blank_id}")
        correct = _observation_correct(blank)
        fields = _sensitive(blank_id, blank, bindings)
        child_edges = []
        parents = [
            parent for parent in node.depends_on
            if parent in observations and parent in NODE_BY_ID
        ]
        seen_wrong = {
            parent: prefix == "model" and _observation_correct(observations[parent]) is False
            for parent in parents
        }
        transfer_known = blank_id in matched
        for parent in parents:
            parent_correct = _observation_correct(observations[parent]) is True
            if not transfer_known and any(seen_wrong.values()):
                if correct is True and seen_wrong[parent]:
                    verdict = {"faithful_transition": None, "transition_status": "recovered", "error_path_active": False}
                elif correct is True:
                    verdict = {"faithful_transition": None, "transition_status": "n/a", "error_path_active": False}
                else:
                    verdict = {"faithful_transition": None, "transition_status": "not_recomputed", "error_path_active": None}
            else:
                verdict = _transition(
                    seen_wrong[parent],
                    correct,
                    matched.get(blank_id),
                    transfer_known,
                )
            record = {
                "parent": f"ID{parent}",
                "child": f"ID{blank_id}",
                "parent_correct": parent_correct,
                "child_correct": correct is True,
                "faithful_transition": verdict["faithful_transition"],
                "transition_status": verdict["transition_status"],
                "error_path_active": verdict["error_path_active"],
            }
            edges.append(record)
            child_edges.append(record)
            incoming[blank_id].append(verdict["transition_status"])
        state = state_row["error_type"] if state_row else "unobserved"
        shown = None if recomputed.get(blank_id) is None else _show(recomputed[blank_id])
        nodes.append({
            "id": f"ID{blank_id}",
            "sample_id": sample_id,
            "condition": condition,
            "subprocess": node.process,
            "error_type": node.error_type,
            "role": node.role,
            "gold": _gold_text(blank),
            "prediction": _prediction_text(blank),
            "correct": correct is True,
            "state": state,
            "depends_on": [f"ID{item}" for item in node.depends_on],
            "contract_direct": list(dict.fromkeys(direct_of.get(blank_id, []))),
            "sensitive_fields": fields,
            "transition_status": _node_transition(incoming[blank_id], state),
            "recomputed": shown,
            "first_error_nodes": list(state_row.get("first_error_nodes") or []) if state_row else [],
        })

    by_id = {item["id"]: item for item in nodes}
    edge_by = {(item["parent"], item["child"]): item for item in edges}
    contracts = _contract_rows(by_id, edge_by, observations, direct_edges, layer)
    summary = _sample_summary(sample_id, condition, prefix, nodes, edges, contracts)
    return {
        "item_id": sample_id,
        "condition": condition,
        "prefix": prefix,
        "nodes": nodes,
        "edges": edges,
        "contract_diagnosis": contracts,
        "summary": summary,
    }


def _contract_rows(by_id, edge_by, observations, direct_edges, layer) -> list[dict]:
    single = defaultdict(list)
    joint = defaultdict(list)
    for edge in direct_edges:
        bucket = single if edge["combination"] == "single" else joint
        bucket[edge["source"]].append(edge["target"])
    relations = defaultdict(list)
    for slot in layer["answer_bindings"].values():
        for item in slot.get("contract_relations") or []:
            if item["relation"] == "indirect":
                relations[item["contract"]].append(item)
    rows = []
    for instance in layer["instances"]:
        contract = instance["id"]
        recognition_targets = list(dict.fromkeys(single.get(contract, [])))
        execution_targets = list(dict.fromkeys(joint.get(contract, []) or recognition_targets))
        recognition = _recognition(recognition_targets, by_id)
        execution = {}
        for target in execution_targets:
            node = by_id.get(target)
            if node is None:
                execution[target] = "unobserved"
                continue
            ok = _execution_ok(node["sensitive_fields"], node["correct"])
            execution[target] = "correct" if ok else "inconsistent"
        indirect_ids = []
        actual = []
        recovered = []
        seen_paths = set()
        for relation in relations.get(contract, []):
            route = relation["via"] + [relation["target"]]
            if any(item not in by_id for item in route):
                continue
            if route[-1] not in indirect_ids:
                indirect_ids.append(route[-1])
            if by_id[route[-1]]["state"] == "recovered" and route[-1] not in recovered:
                recovered.append(route[-1])
            if any(by_id[item]["correct"] for item in route):
                continue
            key = tuple(route)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            statuses = []
            for parent, child in zip(route, route[1:]):
                edge = edge_by.get((parent, child))
                statuses.append(None if edge is None else edge["transition_status"])
            if statuses and all(item == "faithful_propagation" for item in statuses):
                kind = "faithful_propagation"
            elif any(item == "local_transition_failure" for item in statuses):
                kind = "local_recomputation"
            else:
                kind = "unverified_error_path"
            actual.append({"path": route, "type": kind})
        direct_values = list(execution.values())
        if not execution_targets:
            direct_execution = "no_execution_target"
        elif any(item == "inconsistent" for item in direct_values):
            direct_execution = "inconsistent"
        elif direct_values and all(item == "correct" for item in direct_values):
            direct_execution = "consistent"
        else:
            direct_execution = "unobserved"
        wrong_downstream = [
            item for item in indirect_ids if by_id[item]["state"] in {"local_error", "propagated_error"}
        ]
        rows.append({
            "contract": contract,
            "process": instance["process"],
            "recognition": recognition,
            "quadrant": _quadrant(recognition, direct_execution),
            "direct_targets": list(dict.fromkeys(recognition_targets + execution_targets)),
            "direct_execution": execution,
            "direct_execution_status": direct_execution,
            "indirect_targets": indirect_ids,
            "indirect_errors": wrong_downstream,
            "actual_error_paths": actual,
            "recovered_nodes": recovered,
        })
    return rows


def _recognition(targets: list[str], by_id: dict) -> str:
    if not targets:
        return "no_question"
    states = [by_id.get(target) for target in targets]
    if any(item is None for item in states):
        if any(item is not None and not item["correct"] for item in states):
            return "incorrect"
        return "unobserved"
    if all(item["correct"] for item in states):
        return "correct"
    return "incorrect"


def _quadrant(recognition: str, execution: str) -> str:
    if recognition not in {"correct", "incorrect"} or execution not in {"consistent", "inconsistent"}:
        return ""
    table = {
        ("correct", "consistent"): "contract recognition and execution agree",
        ("correct", "inconsistent"): "recognition and execution disagree",
        ("incorrect", "consistent"): "recognition is wrong but the sensitive result is correct",
        ("incorrect", "inconsistent"): "contract recognition and execution both fail",
    }
    return table[(recognition, execution)]


def _rate(numerator: int, denominator: int):
    if not denominator:
        return None
    return round(numerator / denominator, 4)


def _sample_summary(sample_id, condition, prefix, nodes, edges, contracts) -> dict:
    evaluated = [item for item in nodes if item["state"] != "unobserved"]
    local = [item for item in evaluated if item["state"] == "local_error"]
    recognized = [item for item in contracts if item["recognition"] in {"correct", "incorrect"}]
    recognized_ok = [item for item in recognized if item["recognition"] == "correct"]
    executed_given = [
        item for item in recognized_ok
        if item["direct_execution_status"] in {"consistent", "inconsistent"}
    ]
    faithful = [item for item in edges if item["transition_status"] == "faithful_propagation"]
    unfaithful = [item for item in edges if item["transition_status"] == "local_transition_failure"]
    recovered = [item for item in edges if item["transition_status"] == "recovered"]
    return {
        "sample_id": sample_id,
        "condition": condition,
        "prefix": prefix,
        "evaluated_nodes": len(evaluated),
        "local_error_rate": _rate(len(local), len(evaluated)),
        "contract_recognition": _rate(
            sum(item["recognition"] == "correct" for item in recognized),
            len(recognized),
        ),
        "contract_execution_consistency": _rate(
            sum(item["direct_execution_status"] == "consistent" for item in executed_given),
            len(executed_given),
        ),
        "faithful_propagation_edges": len(faithful),
        "unfaithful_propagation_edges": len(unfaithful),
        "recovered_edges": len(recovered),
        "propagation_by_root": _propagation_by_root(nodes, edges),
    }


def _propagation_by_root(nodes: list[dict], edges: list[dict]) -> list[dict]:
    state = {item["id"].removeprefix("ID"): item for item in nodes}
    rows = []
    for item in nodes:
        if item["state"] != "local_error":
            continue
        root = item["id"].removeprefix("ID")
        descendants = [other for other in NODE_BY_ID[root].descendants if other in state]
        faithful = 0
        unfaithful = 0
        recovered = 0
        wrong = 0
        for other in descendants:
            other_state = state[other]["state"]
            if other_state == "recovered":
                recovered += 1
            elif other_state in {"local_error", "propagated_error"}:
                wrong += 1
            if state[other]["transition_status"] == "faithful_propagation":
                faithful += 1
            elif state[other]["transition_status"] == "local_transition_failure":
                unfaithful += 1
        rows.append({
            "root": f"ID{root}",
            "reachable_descendants": len(descendants),
            "downstream_errors": wrong,
            "faithful_propagation": faithful,
            "unfaithful_propagation": unfaithful,
            "recovered": recovered,
            "propagation_rate": _rate(faithful, len(descendants)),
            "downstream_error_rate": _rate(wrong, len(descendants)),
        })
    return rows


def _grouped(records: list[dict]) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("branch") == "probe":
            continue
        sample_id = str(row.get("sample_id") or "")
        condition = str(row.get("condition") or "")
        bucket = grouped.setdefault((sample_id, condition), {})
        for blank in row.get("blanks") or []:
            if str(blank.get("blank_id")) in NODE_BY_ID:
                bucket[str(blank["blank_id"])] = _unwrap_prediction(blank)
    return grouped


def _unwrap_prediction(blank: dict) -> dict:
    """Use the single JSON value when the reply was keyed by sentence number."""
    prediction = str(blank.get("prediction") or "").strip()
    if not prediction.startswith("{"):
        return blank
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError:
        return blank
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return blank
    value = next(iter(parsed.values()))
    if not isinstance(value, str):
        return blank
    from cloze_score import judge

    repaired = dict(blank)
    repaired["prediction"] = value
    repaired.update(judge(
        value,
        str(blank.get("answer") or ""),
        str(blank.get("format_code") or ""),
        blank.get("options"),
    ))
    return repaired


def write_run_diagnosis(records: list[dict], out: Path) -> Path | None:
    """Write the node, edge, and contract tables for each finished sample."""
    grouped = _grouped(records)
    if not grouped:
        return None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    node_rows = []
    edge_rows = []
    contract_rows = []
    summaries = []
    reports = []
    for (sample_id, condition), observations in sorted(grouped.items()):
        diagnosed = diagnose_run(observations, sample_id, condition)
        for row in diagnosed["nodes"]:
            node_rows.append(row)
        for row in diagnosed["edges"]:
            row["sample_id"] = sample_id
            row["condition"] = condition
            edge_rows.append(row)
        for row in diagnosed["contract_diagnosis"]:
            row["sample_id"] = sample_id
            row["condition"] = condition
            contract_rows.append(row)
        summaries.append(diagnosed["summary"])
        reports.append(_markdown(diagnosed))
    _write_jsonl(out / "diagnosis_nodes.jsonl", node_rows)
    _write_jsonl(out / "diagnosis_edges.jsonl", edge_rows)
    _write_jsonl(out / "diagnosis_contracts.jsonl", contract_rows)
    (out / "diagnosis_summary.json").write_text(
        json.dumps(summaries[0] if len(summaries) == 1 else summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out / "diagnosis.md").write_text("\n\n".join(reports).rstrip() + "\n", encoding="utf-8")
    return out / "diagnosis_nodes.jsonl"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cell(value) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def _markdown(diagnosed: dict) -> str:
    summary = diagnosed["summary"]
    lines = [
        f"# {diagnosed['item_id']} · {diagnosed['condition']}",
        "",
        f"Prefix {summary['prefix']}. Local error rate {summary['local_error_rate']}."
        f"Contract recognition rate {summary['contract_recognition']}."
        f"Execution consistency after recognition {summary['contract_execution_consistency']}.",
        "Downstream errors are split into faithful propagation, a failed local recompute, and recovery. They are not recorded as a contract violation.",
        "",
        "## Node status",
        "",
        "| ID | Class | Status | Transfer | Direct contract |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in diagnosed["nodes"]:
        if item["state"] == "correct":
            continue
        contracts = ", ".join(item["contract_direct"]) or "none"
        lines.append(
            f"| {item['id']} | {_cell(item['error_type'])} | {item['state']} | "
            f"{item['transition_status']} | {_cell(contracts)} |"
        )
    lines.extend([
        "",
        "## Contract summary",
        "",
        "| Contract | Recognition | Direct execution | Quadrant | Downstream errors | Recovered |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for item in diagnosed["contract_diagnosis"]:
        if item["recognition"] == "no_question" and item["direct_execution_status"] == "no_execution_target":
            continue
        if item["direct_execution_status"] == "consistent" and not item["indirect_errors"] and not item["recovered_nodes"]:
            continue
        lines.append(
            "| {contract} | {recognition} | {execution} | {quadrant} | {errors} | {recovered} |".format(
                contract=_cell(item["contract"]),
                recognition=item["recognition"],
                execution=item["direct_execution_status"],
                quadrant=item["quadrant"] or "—",
                errors=", ".join(item["indirect_errors"]) or "none",
                recovered=", ".join(item["recovered_nodes"]) or "none",
            )
        )
    return "\n".join(lines)


def _self_check() -> None:
    from cloze_score import answer_letter

    sample_path = Path("data/wind_tunnel/pair/samples.jsonl")
    sample = next(
        json.loads(line)
        for line in sample_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line)["id"] == "standard-default"
    )
    observations = {}
    for question in sample["questions"]:
        blank_id = str(question["id"])
        if blank_id not in NODE_BY_ID:
            continue
        options = [item for item in (question.get("options") or []) if item and item != "none"]
        answer = answer_letter(str(question["answer"]), options)
        observations[blank_id] = {
            "blank_id": blank_id,
            "answer": answer,
            "prediction": answer,
            "options": options,
            "normalized": True,
        }
    intact = diagnose_run(observations, "standard-default", "self-trunc")
    if any(item["state"] != "correct" for item in intact["nodes"]):
        raise RuntimeError("A gold prediction should not produce an error status.")
    for blank_id in ("55", "59", "60", "61", "80", "81", "104"):
        expr = _recompute(blank_id, observations, "model")
        if expr is None or not _equal(expr, _gold_text(observations[blank_id])):
            raise RuntimeError(f"The standard transfer of ID{blank_id} did not return to the gold.")
    flipped = json.loads(json.dumps(observations))
    flipped["43"]["prediction"] = "[[1, 0, 0], [0, cos(γ_M), -sin(γ_M)], [0, sin(γ_M), cos(γ_M)]]"
    flipped["43"]["normalized"] = False
    product = _recompute("55", flipped, "model")
    if product is None or _equal(product, _gold_text(observations["55"])):
        raise RuntimeError("After replacing ID43, the transfer of ID55 should leave the gold.")
    flipped["55"]["prediction"] = _show(product)
    flipped["55"]["normalized"] = False
    diagnosed = diagnose_run(flipped, "standard-default", "self-trunc")
    by_id = {item["id"]: item for item in diagnosed["nodes"]}
    if by_id["ID43"]["state"] != "local_error" or by_id["ID55"]["state"] != "propagated_error":
        raise RuntimeError("ID43 should be a local error, and ID55 multiplied by that wrong matrix should be a propagated error.")
    if by_id["ID55"]["transition_status"] != "faithful_propagation":
        raise RuntimeError("ID55 faithfully used the ID43 given by the model.")
    if by_id["ID60"]["state"] != "recovered":
        raise RuntimeError("When a parent matrix is already wrong and ID60 is still the gold, record a recovery. Do not string it into an error path.")
    link = next(item for item in diagnosed["edges"] if item["parent"] == "ID43" and item["child"] == "ID55")
    if link["error_path_active"] is not True:
        raise RuntimeError("The faithful propagation from ID43 to ID55 should be an active error edge.")
    blocked = next(item for item in diagnosed["edges"] if item["parent"] == "ID60" and item["child"] == "ID80")
    if blocked["error_path_active"] is not False:
        raise RuntimeError("ID60 is still correct, so ID80 cannot be attached to this edge.")
    alpha = next(
        item for item in diagnosed["contract_diagnosis"]
        if item["contract"] == "C_SIGN_ALPHA@wind_to_body"
    )
    if any("ID60" in item["path"] for item in alpha["actual_error_paths"]):
        raise RuntimeError("The error path of the theoretical angle-of-attack sign cannot pass through ID60.")
    if any("ID52" in item["path"] for item in diagnosed["contract_diagnosis"] for item in item["actual_error_paths"]):
        raise RuntimeError("A diagnosis path cannot use the removed ID52.")
    gold_seen = json.loads(json.dumps(observations))
    gold_seen["43"]["prediction"] = flipped["43"]["prediction"]
    gold_seen["43"]["normalized"] = False
    gold_prefix = diagnose_run(gold_seen, "standard-default", "gold-trunc")
    gold_nodes = {item["id"]: item for item in gold_prefix["nodes"]}
    if gold_nodes["ID43"]["state"] != "local_error" or gold_nodes["ID55"]["state"] != "correct":
        raise RuntimeError("Under the gold prefix, ID43 is a local error, and ID55 stays correct because the parent it sees is still the gold.")
    gold_link = next(item for item in gold_prefix["edges"] if item["parent"] == "ID43" and item["child"] == "ID55")
    if gold_link["error_path_active"] is not False:
        raise RuntimeError("The gold prefix does not treat the archived error of ID43 as an active propagation edge of ID55.")
    print(
        "self-check ok: "
        f"nodes {len(intact['nodes'])}, "
        f"faithful edge ID43→ID55, recovered ID60"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write a diagnosis table for a finished wind-tunnel evaluation.")
    parser.add_argument("--records", type=Path, help="records.jsonl. If omitted, only the self-check runs.")
    parser.add_argument("--out", type=Path, help="Directory for the diagnosis table. Defaults to the directory of the records.")
    args = parser.parse_args()
    if args.records is None:
        _self_check()
    else:
        rows = [
            json.loads(line)
            for line in args.records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        target = args.out or args.records.parent
        written = write_run_diagnosis(rows, target)
        print("Wrote" if written else "No wind-tunnel blanks in", written or args.records)
