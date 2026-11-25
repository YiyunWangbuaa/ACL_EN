# -*- coding: utf-8 -*-
"""Wind-tunnel trajectories use the shared graph diagnosis.

Candidate faults are the binary graph-reach rows. Each candidate is causal or
associative according to whether φ stays defined. Library witness text is not
replayed through the manuscript verifier. The repair set of a terminal is the
local-deviation nodes in its ancestor cone.
"""
from __future__ import annotations

from domains.conflict_diagnosis import (
    _ancestors,
    _label,
    _outcome,
    _process_sources,
    _repair_rows,
    assert_report,
    build_readings,
    attach_process,
    conflict_faults,
    contradiction_source,
    field_state,
    finish_report,
    minimal_hitting_sets,
    upstream_faults,
)
from domains.phi_result import from_execution

_REPORTS: dict[tuple[str, str], dict] = {}
_LIBRARIES: dict[str, tuple[dict, dict, dict]] = {}


def stored_report(sample_id: str, condition: str) -> dict | None:
    return _REPORTS.get((sample_id, condition))


def _library(sample_id: str):
    cached = _LIBRARIES.get(sample_id)
    if cached is not None:
        return cached
    from domains.wind_tunnel.signature import evaluate

    by_id = {}
    effects = {}
    injected = {}
    for result in evaluate(sample_id):
        fault = result.instance
        by_id[fault.id] = fault
        if result.det != 1:
            continue
        effects[fault.id] = frozenset(result.entries)
        injected[fault.id] = frozenset(node for node, _value in fault.injector)
    packed = (by_id, effects, injected)
    _LIBRARIES[sample_id] = packed
    return packed


def _reached(effects: dict[str, frozenset[str]], measurement: str, node: str) -> list[str]:
    direct = [fault for fault, columns in effects.items() if measurement in columns]
    if direct or measurement == f"ID{node}":
        return direct
    return [fault for fault, columns in effects.items() if f"ID{node}" in columns]


def _same(node: str, left: str, right: str) -> bool:
    from domains.wind_tunnel.error_graph import NODE_BY_ID
    from domains.wind_tunnel.measurement import _equivalent

    return _equivalent(NODE_BY_ID[node].answer_type, str(left or ""), str(right or ""))


def _rule_parents(node: str) -> tuple[str, ...] | None:
    from domains.wind_tunnel.contract_verifiers import input_ids, validator_of
    from domains.wind_tunnel.error_workflow import PHI

    if node in PHI:
        return PHI[node][2]
    if validator_of(node) is None:
        return None
    return input_ids(node)


def _expected(model, node: str, texts: dict[str, str]):
    """``("value"|"contradiction"|"unavailable"|"no_rule", text)``."""
    from domains.wind_tunnel.contract_executor import attitude_target, roll_target
    from domains.wind_tunnel.contract_verifiers import derive, validator_of
    from domains.wind_tunnel.error_workflow import PHI

    if node in PHI:
        kind, symbol, parents, _chain = PHI[node]
        phrases = [texts.get(parent, "") for parent in parents]
        execution = (
            attitude_target(model, symbol, *phrases, texts)
            if kind == "attitude"
            else roll_target(model, *phrases)
        )
        return _classify(execution)
    if validator_of(node) is None:
        return ("no_rule", "")
    return _classify(derive(model, node, texts))


def _classify(execution) -> tuple[str, str]:
    if execution is None:
        return ("unavailable", "")
    got = from_execution(execution)
    if got is None:
        return ("unavailable", execution.reason or "")
    if isinstance(got, str):
        return ("value", got)
    return ("contradiction", got.reason)


def _ancestor_wrong(node: str, wrong: set[str]) -> bool:
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    seen: set[str] = set()
    pending = list(NODE_BY_ID[node].depends_on)
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        if item in wrong:
            return True
        if item in NODE_BY_ID:
            pending.extend(NODE_BY_ID[item].depends_on)
    return False


def _states(model, rows: dict, gold: dict[str, str]):
    from domains.contract_cognition import answer_sheet, working_texts
    from domains.wind_tunnel.contract_cognition import recognition_ids
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    sheet = answer_sheet({node: obs.model_answer for node, obs in rows.items()})
    recognition = recognition_ids()
    base = {str(node): str(value) for node, value in gold.items()}
    for node, obs in rows.items():
        if str(obs.gold_answer or "").strip():
            base.setdefault(node, str(obs.gold_answer))
    wrong = {
        node
        for node, obs in rows.items()
        if not obs.correct and obs.parse_status not in {"unparseable", "ambiguous"} and str(obs.model_answer or "").strip()
    }
    found: dict[str, str] = {}
    recomputed: dict[str, str] = {}
    outcomes: dict[str, dict] = {}
    unscored: list[str] = []
    for node, obs in rows.items():
        if node not in NODE_BY_ID:
            continue
        raw = str(obs.model_answer or "")
        matches_gold = bool(obs.correct) and bool(raw.strip())
        if not raw.strip() or obs.parse_status in {"unparseable", "ambiguous"}:
            found[node] = "unparseable_or_unscorable"
            unscored.append(f"ID{node}")
            outcomes[node] = _outcome("unavailable", False, None)
            continue
        parents = _rule_parents(node)
        if parents == ():
            found[node] = "correct_after_wrong_ancestor" if matches_gold and _ancestor_wrong(node, wrong) else (
                "correct" if matches_gold else "wrong_and_locally_inconsistent"
            )
            outcomes[node] = _outcome("root", matches_gold, None)
            continue
        if parents and any(
            parent in rows and not str(rows[parent].model_answer or "").strip()
            for parent in parents
        ):
            found[node] = "unparseable_or_unscorable"
            unscored.append(f"ID{node}")
            outcomes[node] = _outcome("unavailable", matches_gold, None)
            continue
        texts = working_texts(base, sheet, recognition, parents or ())
        kind, text = _expected(model, node, texts)
        if kind == "no_rule":
            found[node] = "correct" if matches_gold else "wrong_without_local_rule"
            if matches_gold and _ancestor_wrong(node, wrong):
                found[node] = "correct_after_wrong_ancestor"
            outcomes[node] = _outcome("no_rule", matches_gold, None)
            continue
        if kind == "contradiction":
            found[node] = "contract_contradiction"
            gold_texts = {str(key): str(value) for key, value in gold.items()}
            for item, row in rows.items():
                if str(row.gold_answer or "").strip():
                    gold_texts[item] = str(row.gold_answer)
            gkind, gtext = _expected(model, node, gold_texts)
            gold_value = str(obs.gold_answer or gold.get(node, ""))
            source = contradiction_source(gtext if gkind == "value" else None, gkind == "value" and _same(node, gtext, gold_value))
            outcomes[node] = _outcome("contradiction", matches_gold, None, source)
            continue
        if kind != "value":
            if not matches_gold:
                found[node] = "unparseable_or_unscorable"
                unscored.append(f"ID{node}")
            else:
                found[node] = "correct_after_wrong_ancestor" if _ancestor_wrong(node, wrong) else "correct"
            outcomes[node] = _outcome("unavailable", matches_gold, None)
            continue
        matches_phi = _same(node, raw, text)
        recomputed[node] = text
        if matches_gold and matches_phi:
            found[node] = "correct_after_wrong_ancestor" if _ancestor_wrong(node, wrong) else "correct"
        elif matches_gold:
            found[node] = "unfaithful_correct"
        elif matches_phi:
            found[node] = "wrong_but_locally_consistent"
        else:
            found[node] = "wrong_and_locally_inconsistent"
        outcomes[node] = _outcome("value", matches_gold, matches_phi)
    return found, unscored, recomputed, outcomes


def _graph_parents(node: str) -> tuple[str, ...]:
    from domains.wind_tunnel.error_graph import diagnosis_parents

    return diagnosis_parents(node)


def _field_matches(node: str, measurement: str, observed: str, phi: str) -> bool:
    from domains.wind_tunnel.measurement import project

    if measurement == f"ID{node}":
        return _same(node, observed, phi)
    return project(node, observed).get(measurement, "") == project(node, phi).get(measurement, "")


def _collect(rows: dict, gold: dict[str, str], states: dict[str, str], phi: dict[str, str], effects, injected):
    from domains.conflict_diagnosis import _WRONG
    from domains.wind_tunnel.measurement import differences

    records = []
    symptoms: list[str] = []
    unexplained: list[str] = []
    for node, state in states.items():
        if state not in _WRONG:
            continue
        obs = rows[node]
        observed = str(obs.model_answer or "")
        base = str(obs.gold_answer or gold.get(node, ""))
        if state == "contract_contradiction" and obs.correct:
            continue
        found = list(differences(node, observed, base))
        if not found:
            found = [f"ID{node}"]
        local = [fault for fault, nodes in injected.items() if node in nodes]
        ancestors = _ancestors(_graph_parents(node), _graph_parents)
        phi_text = phi.get(node)
        for measurement in found:
            symptoms.append(measurement)
            if state == "contract_contradiction":
                measurement_state = state
            else:
                matches = None if phi_text is None else _field_matches(node, measurement, observed, phi_text)
                measurement_state = field_state(state, matches)
            reached = upstream_faults(_reached(effects, measurement, node), injected, ancestors)
            faults = conflict_faults(measurement_state, local, reached) or []
            if not faults:
                unexplained.append(measurement)
                continue
            records.append({
                "measurement": measurement,
                "node": node,
                "state": measurement_state,
                "faults": faults,
            })
    return records, symptoms, unexplained


def explain_observations(model, observations, gold: dict[str, str], baseline: dict[str, str] | None = None) -> dict:
    """Every model-prefix trajectory, not only the previously unresolved cases."""
    from domains.conflict_diagnosis import _WRONG, _attach_nearest, answer_sheet, factor_terminals
    from domains.contract_cognition import working_texts
    from domains.wind_tunnel.contract_cognition import recognition_ids
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    rows = {
        str(obs.node_id).removeprefix("ID"): obs
        for obs in observations
        if str(obs.node_id).removeprefix("ID") in NODE_BY_ID and obs.prefix == "model"
    }
    sample_id = str(observations[0].sample_id) if observations else ""
    condition = str(observations[0].condition) if observations else ""
    if not rows:
        return _store(sample_id, condition, finish_report(
            symptoms=[], states={}, records=[], unexplained=[], unscored=[],
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, status="gold_prefix",
        ))
    states, unscored, phi, outcomes = _states(model, rows, gold)
    from domains.wind_tunnel.t_final import final_nodes

    fallback = tuple(node for node in final_nodes() if node in rows and _graph_parents(node))
    if baseline is None:
        observed_targets = tuple(node for node in final_nodes() if node in rows)
    else:
        sinks = tuple(node for node in factor_terminals(NODE_BY_ID, _graph_parents, gold, baseline) if node in rows)
        observed_targets = sinks or fallback
    readings, _target_rows = build_readings(outcomes, _graph_parents, observed_targets)
    sheet = answer_sheet({node: obs.model_answer for node, obs in rows.items()})
    recognition = recognition_ids()
    wanted = {_label(node) for node in observed_targets}

    def becomes_defined(node: str, trial: dict) -> bool:
        texts = working_texts(gold, trial, recognition, _graph_parents(node) or ())
        kind, _text = _expected(model, node, texts)
        return kind == "value"

    def close(found: list[dict]) -> list[dict]:
        _attach_nearest(found, gold, sheet, _graph_parents, becomes_defined)
        return [item for item in found if item["node"] in wanted]

    sources = _process_sources(states, outcomes)
    actionable = any(
        state in _WRONG and not (state == "contract_contradiction" and rows[node].correct)
        for node, state in states.items()
    )
    if not actionable and not sources:
        readings, process = attach_process(readings, states, outcomes, {}, _graph_parents)
        target_rows = close(readings)
        return _store(sample_id, condition, finish_report(
            symptoms=[], states=states, records=[], unexplained=[], unscored=unscored,
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, readings=readings, targets=target_rows, **process,
        ))
    by_id, effects, injected = _library(sample_id)
    catalog = {fault_id: frozenset(node for node, _value in fault.injector) for fault_id, fault in by_id.items()}
    readings, process = attach_process(readings, states, outcomes, catalog, _graph_parents)
    target_rows = close(readings)
    if not actionable:
        return _store(sample_id, condition, finish_report(
            symptoms=[], states=states, records=[], unexplained=[], unscored=unscored,
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, readings=readings, targets=target_rows, **process,
        ))
    records, symptoms, unexplained = _collect(rows, gold, states, phi, effects, injected)
    diagnoses, truncated = minimal_hitting_sets([frozenset(record["faults"]) for record in records])
    repair_cardinality, repairs = _repair_rows(
        states,
        outcomes,
        _graph_parents,
        observed_targets,
        {node: str(obs.model_answer or "") for node, obs in rows.items()},
        gold,
        _same,
    )
    return _store(sample_id, condition, finish_report(
        symptoms=symptoms,
        states=states,
        records=records,
        unexplained=unexplained,
        unscored=unscored,
        diagnoses=diagnoses,
        repair_cardinality=repair_cardinality,
        repairs=repairs,
        truncated=truncated,
        readings=readings,
        targets=target_rows,
        **process,
    ))


def _store(sample_id: str, condition: str, report: dict) -> dict:
    assert_report(report)
    if sample_id:
        _REPORTS[(sample_id, condition)] = report
    return report
