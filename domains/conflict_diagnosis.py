# -*- coding: utf-8 -*-
"""Candidate sets on the dependency graph, split by whether φ stays defined.

Each wrong measurement collects library rows by graph reach and local φ.
A path on which every φ has a value is attributable. A path that meets φ=⊥
is associative. The repair set of a target is the local-deviation nodes in
its ancestor cone, read off that classification.

Library witness text is not replayed through this verifier. Replaying a
stored wrong answer on the gold trajectory only reruns φ on another input.
Matching the real model is a separate session on the LLM.
"""
from __future__ import annotations

from domains.contract_cognition import answer_sheet, working_texts
from domains.mbd_kernel import DomainView, evaluate, sort_key
from domains.phi_result import Contradiction

_MAX_CARDINALITY = 4
_MAX_DIAGNOSES = 48
_WRONG = frozenset({
    "wrong_but_locally_consistent",
    "wrong_and_locally_inconsistent",
    "wrong_without_local_rule",
    "contract_contradiction",
})
_STATUSES = frozenset({
    "attributed",
    "unmodeled",
    "consistent",
    "unscored",
    "gold_prefix",
})
_NOTES = {
    "attributed": "错测点已在依赖图上收成候选集，并按 φ 是否有值切开。",
    "unmodeled": "错测点没有可归入的库内故障。",
    "consistent": "没有错误测点。",
    "unscored": "只有无法评分的测点，没有把它当成故障。",
    "gold_prefix": "金标前缀不生成故障诊断。",
}
_LIBRARIES: dict[str, tuple[dict, dict, dict]] = {}


def _label(node: str) -> str:
    text = str(node)
    if text[:1].isalpha():
        return text
    return f"ID{text}"


def conflict_faults(state: str, local_ids: list[str], reached_ids: list[str]) -> list[str] | None:
    """Minimal fault candidates for one wrong measurement.

    ``reached_ids`` must already be injectors strictly upstream of this node.
    A contradiction at this node keeps upstream and local candidates, because
    there is no recomputed value to drop the local ones. An inconsistent
    measurement, or a node with no local rule, does the same. A consistent
    measurement keeps only the upstream candidates. Inheritance of a
    contradiction does not change this choice.
    """
    if state == "wrong_but_locally_consistent":
        faults = sorted(set(reached_ids))
        return faults or None
    if state in {"wrong_and_locally_inconsistent", "wrong_without_local_rule", "contract_contradiction"}:
        faults = sorted(set(local_ids) | set(reached_ids))
        return faults or None
    return None


def field_state(node_state: str, matches_phi: bool | None) -> str:
    """Consistency of one field against the recomputed node.

    ``matches_phi`` is None when this node has no recomputed value. The node
    state then stands for every one of its fields.
    """
    if node_state == "wrong_without_local_rule" or matches_phi is None:
        return node_state
    if matches_phi:
        return "wrong_but_locally_consistent"
    return "wrong_and_locally_inconsistent"


def _ancestors(seeds, parents_of) -> set[str]:
    seen: set[str] = set()
    pending = list(seeds)
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pending.extend(parents_of(item))
    return seen


def node_phi(view: DomainView, node: str, assignment: dict[str, str]):
    """φ in the model's contract cognition, plus this node's direct parents.

    ``assignment`` is the model answer sheet. Recognition answers in it
    overwrite the gold declaration even when they are not parents of ``node``.
    Derived answers overwrite the declaration only when they are direct parents.
    """
    texts = working_texts(view.gold(), assignment, view.declaration_roots, view.parents(node))
    if view.reference is not None:
        return view.reference(node, texts)
    return view.answer_under(node, texts)


def _reading_name(phi_kind: str, matches_gold: bool, matches_phi: bool | None) -> str:
    if phi_kind == "contradiction":
        return "contradiction_correct" if matches_gold else "contradiction_wrong"
    if phi_kind in {"unavailable", "unscored"}:
        return "unscored"
    if phi_kind == "no_rule":
        return "no_rule"
    if phi_kind == "root":
        return "faithful_reliable" if matches_gold else "local_deviation"
    if matches_phi and matches_gold:
        return "faithful_reliable"
    if matches_phi:
        return "faithful_wrong"
    if matches_gold:
        return "unfaithful_correct"
    return "local_deviation"


def _candidate_mode(phi_kind: str, reading: str, matches_gold: bool) -> str:
    if phi_kind == "contradiction":
        return "union"
    if reading == "faithful_wrong":
        return "upstream"
    if reading in {"local_deviation", "unfaithful_correct"}:
        return "union"
    if reading == "no_rule" and not matches_gold:
        return "union"
    return "none"


_ANSWER_ZH = {
    "faithful_reliable": "一致性正确",
    "faithful_wrong": "可归因错误",
    "local_deviation": "可归因错误",
    "unfaithful_correct": "可追溯答案遮蔽",
    "contradiction_wrong": "不可归因错误",
    "contradiction_correct": "随机性正确",
    "unscored": "无法评分",
    "no_rule": "没有局部规则",
    "unconstrained_input": "不可归因错误",
    "unreliable_correct": "可追溯答案遮蔽",
    "一致性正确": "一致性正确",
    "可追溯答案遮蔽": "可追溯答案遮蔽",
    "随机性正确": "随机性正确",
    "可归因错误": "可归因错误",
    "不可归因错误": "不可归因错误",
}


def target_note(rows: list[dict]) -> str:
    """Chinese sentence for targets whose reliability or reading is not the default."""
    flagged = [
        item for item in rows
        if item.get("reliability") != "deterministic" or item.get("answer_class") not in {"faithful_reliable", "一致性正确"}
    ]
    if not flagged:
        return ""
    parts = [
        f"{item['node']} {_ANSWER_ZH.get(item['answer_class'], item['answer_class'])}"
        for item in flagged
    ]
    return "终点 " + "，".join(parts) + "。"


def build_readings(outcomes: dict[str, dict], parents_of, targets: tuple[str, ...] = ()) -> tuple[list[dict], list[dict]]:
    """Local reading plus a reliability flag. The flag does not rewrite candidates."""
    contradicted = {node for node, row in outcomes.items() if row["phi_kind"] == "contradiction"}
    readings = []
    for node in sorted(outcomes, key=sort_key):
        row = outcomes[node]
        if row["phi_kind"] == "contradiction":
            reliability = "contradiction"
        elif _ancestors(parents_of(node), parents_of) & contradicted:
            reliability = "inherited"
        else:
            reliability = "deterministic"
        reading = _reading_name(row["phi_kind"], row["matches_gold"], row["matches_phi"])
        if reliability == "inherited" and row["phi_kind"] == "value":
            answer_class = "unconstrained_input"
        else:
            answer_class = reading
        readings.append({
            "node": _label(node),
            "reading": reading,
            "reliability": reliability,
            "answer_class": answer_class,
            "candidate_mode": _candidate_mode(row["phi_kind"], reading, row["matches_gold"]),
            "matches_gold": row["matches_gold"],
            "matches_phi": row["matches_phi"],
            "contradiction_source": row.get("contradiction_source") or "",
        })
    wanted = {_label(node) for node in targets}
    return readings, [item for item in readings if item["node"] in wanted]


def upstream_faults(reached_ids, injected: dict[str, frozenset[str]], ancestors: set[str]) -> list[str]:
    """Injectors that lie entirely in the strict ancestors and reach this measurement."""
    found = []
    for fault in reached_ids:
        points = injected.get(fault) or frozenset()
        if points and set(points) <= ancestors:
            found.append(fault)
    return found


def minimal_hitting_sets(conflicts: list[frozenset[str]]) -> tuple[list[list[str]], bool]:
    """Incremental minimal hitting sets. Supersets are dropped first.

    Returns the diagnoses and whether the cardinality or count cap truncated
    the search. An empty conflict cannot be hit, so the result is empty.
    """
    reduced: list[frozenset[str]] = []
    for raw in sorted(conflicts, key=lambda item: (len(item), sorted(item))):
        block = frozenset(raw)
        if not block:
            return [], False
        if any(other <= block for other in reduced):
            continue
        reduced = [other for other in reduced if not block < other]
        reduced.append(block)
    if not reduced:
        return [[]], False

    diagnoses = {frozenset()}
    truncated = False
    for conflict in reduced:
        expanded: set[frozenset[str]] = set()
        for diagnosis in diagnoses:
            if diagnosis & conflict:
                expanded.add(diagnosis)
                continue
            for fault in sorted(conflict):
                nxt = diagnosis | {fault}
                if len(nxt) > _MAX_CARDINALITY:
                    truncated = True
                    continue
                expanded.add(nxt)
        minimal = [
            candidate
            for candidate in expanded
            if not any(other < candidate for other in expanded)
        ]
        if len(minimal) > _MAX_DIAGNOSES:
            minimal.sort(key=lambda item: (len(item), sorted(item)))
            minimal = minimal[:_MAX_DIAGNOSES]
            truncated = True
        diagnoses = set(minimal)
        if not diagnoses:
            truncated = True
            break
    ordered = [
        sorted(item)
        for item in sorted(diagnoses, key=lambda item: (len(item), sorted(item)))
        if item
    ]
    return ordered, truncated


def remember(view: DomainView, results) -> None:
    """Keep the binary graph-reach rows. A later trajectory must not rebuild them."""
    by_id = {}
    effects: dict[str, frozenset[str]] = {}
    injected: dict[str, frozenset[str]] = {}
    for result in results:
        fault = result.instance
        by_id[fault.id] = fault
        if result.det != 1:
            continue
        effects[fault.id] = frozenset(result.entries)
        injected[fault.id] = frozenset(node for node, _value in fault.injector)
    _LIBRARIES[view.name] = (by_id, effects, injected)


def _library(view: DomainView) -> tuple[dict, dict, dict]:
    """Binary support rows, keyed by view name. The library does not depend on one trace."""
    cached = _LIBRARIES.get(view.name)
    if cached is not None:
        return cached
    remember(view, evaluate(view))
    return _LIBRARIES[view.name]


def _same(view: DomainView, node: str, left: str, right: str) -> bool:
    if str(left) == str(right):
        return True
    return view.equal(str(left), str(right), node)


def _outcome(
    phi_kind: str,
    matches_gold: bool,
    matches_phi: bool | None,
    contradiction_source: str = "",
) -> dict:
    return {
        "phi_kind": phi_kind,
        "matches_gold": matches_gold,
        "matches_phi": matches_phi,
        "contradiction_source": contradiction_source,
    }


def contradiction_source(phi_on_gold, same_as_gold: bool) -> str:
    """``parents`` when gold parents make φ defined and equal gold. Otherwise the node itself."""
    if isinstance(phi_on_gold, str) and same_as_gold:
        return "parents"
    return "local"


def _states(view: DomainView, answers: dict[str, str]):
    gold = view.gold()
    correct: dict[str, bool] = {}
    unscored: list[str] = []
    present = [node for node in view.node_ids if node in answers]
    for node in present:
        raw = "" if answers[node] is None else str(answers[node])
        if not raw.strip():
            correct[node] = False
            continue
        correct[node] = _same(view, node, raw, gold[node])

    def ancestor_wrong(node: str) -> bool:
        return bool(_ancestors(view.parents(node), view.parents) & {item for item, ok in correct.items() if not ok})

    found: dict[str, str] = {}
    recomputed: dict[str, str] = {}
    outcomes: dict[str, dict] = {}
    _sheet = answer_sheet(answers)
    for node in present:
        raw = "" if answers[node] is None else str(answers[node])
        matches_gold = bool(raw.strip()) and correct.get(node, False)
        if not raw.strip():
            found[node] = "unparseable_or_unscorable"
            unscored.append(_label(node))
            outcomes[node] = _outcome("unavailable", False, None)
            continue
        parents = view.parents(node)
        if not parents:
            found[node] = "correct" if matches_gold else "wrong_and_locally_inconsistent"
            if matches_gold and ancestor_wrong(node):
                found[node] = "correct_after_wrong_ancestor"
            outcomes[node] = _outcome("root", matches_gold, None)
            continue
        if any(parent in answers and not str(answers[parent] or "").strip() for parent in parents):
            found[node] = "unparseable_or_unscorable"
            unscored.append(_label(node))
            outcomes[node] = _outcome("unavailable", matches_gold, None)
            continue
        phi = node_phi(view, node, _sheet)
        if isinstance(phi, Contradiction):
            found[node] = "contract_contradiction"
            phi_gold = node_phi(view, node, {parent: str(gold[parent]) for parent in parents})
            source = contradiction_source(phi_gold, isinstance(phi_gold, str) and _same(view, node, phi_gold, str(gold[node])))
            outcomes[node] = _outcome("contradiction", matches_gold, None, source)
            continue
        if phi is None:
            if matches_gold:
                found[node] = "correct_after_wrong_ancestor" if ancestor_wrong(node) else "correct"
            else:
                found[node] = "unparseable_or_unscorable"
                unscored.append(_label(node))
            outcomes[node] = _outcome("unavailable", matches_gold, None)
            continue
        matches_phi = _same(view, node, raw, str(phi))
        recomputed[node] = str(phi)
        if matches_gold and matches_phi:
            found[node] = "correct_after_wrong_ancestor" if ancestor_wrong(node) else "correct"
        elif matches_gold:
            found[node] = "unfaithful_correct"
        elif matches_phi:
            found[node] = "wrong_but_locally_consistent"
        else:
            found[node] = "wrong_and_locally_inconsistent"
        outcomes[node] = _outcome("value", matches_gold, matches_phi)
    return found, unscored, recomputed, outcomes


def _diffs(view: DomainView, node: str, predicted: str, gold: str) -> list[str]:
    from domains.mbd_kernel import _project

    pred = _project(view, node, predicted)
    base = _project(view, node, gold)
    found = []
    for measurement, value in pred.items():
        if measurement == f"ID{node}":
            if not _same(view, node, predicted, gold):
                found.append(measurement)
            continue
        if value != base.get(measurement, ""):
            found.append(measurement)
    if not found and not _same(view, node, predicted, gold):
        found.append(f"ID{node}")
    return found


def _reached(effects: dict[str, frozenset[str]], measurement: str, node: str) -> list[str]:
    direct = [fault for fault, columns in effects.items() if measurement in columns]
    if direct or measurement == f"ID{node}":
        return direct
    return [fault for fault, columns in effects.items() if f"ID{node}" in columns]


def _field_matches(view: DomainView, node: str, measurement: str, observed: str, phi: str) -> bool:
    from domains.mbd_kernel import _project

    if measurement == f"ID{node}":
        return _same(view, node, observed, phi)
    return _project(view, node, observed).get(measurement, "") == _project(view, node, phi).get(measurement, "")


def _collect(view: DomainView, answers: dict[str, str], states: dict[str, str], phi: dict[str, str], effects, injected):
    gold = view.gold()
    records = []
    symptoms: list[str] = []
    unexplained: list[str] = []
    for node, state in states.items():
        if state not in _WRONG:
            continue
        predicted = "" if answers.get(node) is None else str(answers.get(node))
        if state == "contract_contradiction" and _same(view, node, predicted, str(gold[node])):
            continue
        measurements = _diffs(view, node, predicted, str(gold[node]))
        if not measurements:
            unexplained.append(f"ID{node}")
            continue
        local = [fault for fault, nodes in injected.items() if node in nodes]
        ancestors = _ancestors(view.parents(node), view.parents)
        phi_text = phi.get(node)
        for measurement in measurements:
            symptoms.append(measurement)
            if state == "contract_contradiction":
                measurement_state = state
            else:
                matches = None if phi_text is None else _field_matches(view, node, measurement, predicted, phi_text)
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


def _must_replace(state: str | None, outcome: dict | None) -> bool:
    """A node whose own answer must be overwritten before φ can restore a descendant.

    Faithful propagation is absent: gold parents already recompute it to gold.
    A contradiction caused by the parents is absent for the same reason.
    A contradiction that remains on gold parents stays, because φ cannot recompute it.
    """
    if state in {"wrong_and_locally_inconsistent", "wrong_without_local_rule"}:
        return True
    if state != "contract_contradiction" or not outcome:
        return False
    return outcome.get("contradiction_source") == "local" and not outcome.get("matches_gold")


def _process_sources(states: dict[str, str], outcomes: dict[str, dict]) -> set[str]:
    """Nodes that contribute a fault. Gold output does not remove them."""
    del outcomes
    return {
        node
        for node, state in states.items()
        if state in {
            "wrong_and_locally_inconsistent",
            "wrong_without_local_rule",
            "unfaithful_correct",
            "contract_contradiction",
        }
    }


def _fault_sites(fault: str, points: frozenset[str]) -> set[str]:
    """Injector nodes plus the node named by the instance itself."""
    sites = set(points)
    body = fault.split(":", 1)[-1]
    if "<-" in body:
        sites.add(body.split("<-", 1)[0])
    else:
        sites.add(body)
    return sites


def conservative_faults(node: str, sources: set[str], injected: dict[str, frozenset[str]], parents_of, *, include_self: bool) -> list[str]:
    """Ancestor faults that stay in force. An intermediate gold answer does not drop them."""
    cone = set(_ancestors(parents_of(node), parents_of))
    if include_self:
        cone.add(node)
    found = []
    for fault, points in injected.items():
        sites = _fault_sites(fault, points)
        if not sites & sources or not sites & cone:
            continue
        if points and not set(points) <= cone:
            continue
        found.append(fault)
    return sorted(found)


def _five_class(item: dict) -> str:
    """Answer class: correctness × attributability, plus the faithful baseline.

    Attributable means the chain never meets φ=⊥. Whether the written answer
    equals φ does not choose the class.
    """
    reading = item.get("reading")
    if reading == "unscored":
        return "无法评分"
    if reading == "no_rule":
        return "没有局部规则"
    strength = item.get("chain_strength") or ""
    gold = bool(item.get("matches_gold"))
    if strength == "未解开":
        return "未解开"
    if strength == "关联" or reading in {"contradiction_correct", "contradiction_wrong"}:
        return "随机性正确" if gold else "不可归因错误"
    if gold and not item.get("process_faults"):
        return "一致性正确"
    if gold:
        return "可追溯答案遮蔽"
    return "可归因错误"


def _chain_class(strength: str, outcome: dict | None) -> str:
    """Five answer classes. Attributable means φ stays defined on the chain.

    Faithfulness (whether the written answer equals φ) is not this label.
    A fully correct chain is assigned where no fault remains.
    """
    matches_gold = bool((outcome or {}).get("matches_gold"))
    if strength == "未解开":
        return "未解开"
    if strength == "关联":
        return "随机性正确" if matches_gold else "不可归因错误"
    if matches_gold:
        return "可追溯答案遮蔽"
    return "可归因错误"


def _child_map(nodes: set[str], parents_of) -> dict[str, list[str]]:
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for node in nodes:
        for parent in parents_of(node):
            children.setdefault(parent, []).append(node)
    return children


def _point_band(outcome: dict | None) -> str:
    """ok when φ has a value, break when φ=⊥, open when φ did not run."""
    if not outcome:
        return "open"
    kind = outcome.get("phi_kind")
    if kind == "contradiction":
        return "break"
    if kind in {"root", "value"}:
        return "ok"
    return "open"


def _self_band(outcome: dict | None) -> str:
    """A node is associative only when its own φ is ⊥. A value, matched or not, is causal."""
    if not outcome:
        return "unresolved"
    kind = outcome.get("phi_kind")
    if kind == "contradiction":
        return "associative"
    if kind in {"root", "value"}:
        return "causal"
    return "unresolved"


def _path_nodes(source: str, target: str, children: dict[str, list[str]], ancestors: set[str]) -> set[str] | None:
    """Nodes on the inclusive path. None when target is not downstream of source."""
    if source == target:
        return {source}
    allowed = set(ancestors)
    allowed.add(target)
    seen: set[str] = set()
    pending = [node for node in children.get(source, ()) if node in allowed]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(child for child in children.get(node, ()) if child in allowed)
    if target not in seen:
        return None
    seen.add(source)
    return seen


def _path_band(source: str, target: str, outcomes: dict[str, dict], children: dict[str, list[str]], ancestors: set[str]) -> str:
    """causal when every φ on the path has a value. associative only at φ=⊥."""
    if source == target:
        return _self_band(outcomes.get(source))
    seen = _path_nodes(source, target, children, ancestors)
    if seen is None:
        return "unresolved"
    marks = [_point_band(outcomes.get(node)) for node in seen]
    if "break" in marks:
        return "associative"
    if "open" in marks:
        return "unresolved"
    return "causal"


def _chain_strength(source: str, target: str, outcomes: dict[str, dict], children: dict[str, list[str]], ancestors: set[str]) -> str:
    band = _path_band(source, target, outcomes, children, ancestors)
    return {"causal": "因果", "associative": "关联", "unresolved": "未解开"}[band]


def partition_candidates(by_node: dict[str, list[str]], outcomes: dict[str, dict], injected: dict[str, frozenset[str]], parents_of) -> list[dict]:
    """Split each node's candidate set. This set is not a conflict.

    Causal means every φ from the injector to the node has a value, including
    a local mismatch. Associative means some node on that path has φ=⊥.
    Unresolved means a rule did not run and no contradiction was seen.
    """
    nodes = set(outcomes) | set(by_node)
    children = _child_map(nodes, parents_of)
    rows = []
    for target in sorted(by_node, key=sort_key):
        faults = list(by_node.get(target) or [])
        if not faults:
            continue
        ancestors = _ancestors(parents_of(target), parents_of)
        groups = {"causal": [], "associative": [], "unresolved": []}
        for fault in faults:
            points = set(injected.get(fault) or ()) or _fault_sites(fault, frozenset())
            relevant = [point for point in points if point == target or point in ancestors]
            if not relevant:
                groups["unresolved"].append(fault)
                continue
            bands = [_path_band(point, target, outcomes, children, ancestors) for point in relevant]
            if "associative" in bands:
                band = "associative"
            elif "unresolved" in bands:
                band = "unresolved"
            else:
                band = "causal"
            groups[band].append(fault)
        rows.append({
            "target": target,
            "candidates": faults,
            "causal": groups["causal"],
            "associative": groups["associative"],
            "unresolved": groups["unresolved"],
        })
    return rows


def chain_attributions(states: dict[str, str], outcomes: dict[str, dict], sources: set[str], parents_of) -> list[dict]:
    """Label each source-to-node pair by whether φ stays defined on the path."""
    nodes = set(states) | set(outcomes)
    children = _child_map(nodes, parents_of)
    rows = []
    for target in sorted(nodes, key=sort_key):
        ancestors = _ancestors(parents_of(target), parents_of)
        candidates = [source for source in sources if source == target or source in ancestors]
        outcome = outcomes.get(target)
        matches_gold = bool((outcome or {}).get("matches_gold"))
        for source in sorted(candidates, key=sort_key):
            strength = _chain_strength(source, target, outcomes, children, ancestors)
            rows.append({
                "target": target,
                "source": source,
                "strength": strength,
                "chain_class": _chain_class(strength, outcome),
                "target_correct": matches_gold,
            })
    return rows


def _apply_chains(readings: list[dict], rows: list[dict]) -> None:
    """The explaining path sets the class. A side path does not replace it.

    A target whose own φ is ⊥ stays unattributable. A target whose own φ has a
    value takes a causal path when one exists. Associative and unresolved
    paths are kept on ``side_paths``.
    """
    by_target: dict[str, list[dict]] = {}
    for row in rows:
        by_target.setdefault(_label(row["target"]), []).append({
            "source": _label(row["source"]),
            "strength": row["strength"],
            "chain_class": row["chain_class"],
        })
    for item in readings:
        chains = by_target.get(item["node"], [])
        item["chains"] = chains
        own_break = item.get("reading") in {"contradiction_wrong", "contradiction_correct"}
        if not chains:
            item["chain_strength"] = ""
            item["chain_class"] = "一致性正确" if item.get("matches_gold") and not item.get("process_faults") else ""
            item["side_paths"] = []
            continue
        if own_break:
            chosen = next((chain for chain in chains if chain["strength"] == "关联"), chains[0])
            item["chain_strength"] = "关联"
            item["chain_class"] = "随机性正确" if item.get("matches_gold") else "不可归因错误"
        else:
            chosen = next((chain for chain in chains if chain["strength"] == "因果"), None)
            if chosen is None:
                chosen = next((chain for chain in chains if chain["strength"] == "未解开"), chains[0])
            item["chain_strength"] = chosen["strength"]
            item["chain_class"] = chosen["chain_class"]
        item["side_paths"] = [
            {"source": chain["source"], "strength": chain["strength"]}
            for chain in chains
            if chain["strength"] != item["chain_strength"]
        ]


def _process_text(state: str | None, outcome: dict | None) -> str:
    kind = (outcome or {}).get("phi_kind")
    if kind == "contradiction" or state == "contract_contradiction":
        return "答案正确，局部规则矛盾，上游故障仍保留。"
    if state == "unfaithful_correct":
        return "答案正确，但不符合局部规则，上游故障仍保留。"
    return "答案正确，不能证明因果关系已恢复，上游故障仍保留。"


def attach_process(readings: list[dict], states: dict[str, str], outcomes: dict[str, dict], injected: dict[str, frozenset[str]], parents_of):
    """Fault sets for correct answers whose process is still unreliable."""
    sources = _process_sources(states, outcomes)
    by_node: dict[str, list[str]] = {}
    rows = []
    for node in sorted(set(states) | set(outcomes), key=sort_key):
        faults = conservative_faults(
            node, sources, injected, parents_of, include_self=node in sources,
        )
        by_node[node] = faults
        outcome = outcomes.get(node) or {}
        if outcome.get("matches_gold") and faults:
            rows.append({
                "node": node,
                "state": "unreliable_correct",
                "faults": faults,
                "attribution": _process_text(states.get(node), outcome),
            })
    diagnoses, truncated = minimal_hitting_sets([frozenset(row["faults"]) for row in rows]) if rows else ([], False)
    chains = chain_attributions(states, outcomes, sources, parents_of)
    by_target_chains: dict[str, list[dict]] = {}
    for chain in chains:
        by_target_chains.setdefault(chain["target"], []).append(chain)
    for row in rows:
        row["chains"] = [
            {"source": chain["source"], "strength": chain["strength"], "chain_class": chain["chain_class"]}
            for chain in by_target_chains.get(row["node"], [])
        ]
    attributed = {_label(row["node"]): row["attribution"] for row in rows}
    listed = {_label(node): faults for node, faults in by_node.items()}
    partitions = partition_candidates(by_node, outcomes, injected, parents_of)
    by_partition = {_label(row["target"]): row for row in partitions}
    for item in readings:
        faults = listed.get(item["node"], [])
        item["process_faults"] = faults
        item["process_attribution"] = attributed.get(item["node"], "")
        part = by_partition.get(item["node"]) or {}
        item["candidates"] = list(part.get("candidates") or faults)
        item["causal_candidates"] = list(part.get("causal") or [])
        item["associative_candidates"] = list(part.get("associative") or [])
        item["unresolved_candidates"] = list(part.get("unresolved") or [])
    _apply_chains(readings, chains)
    for item in readings:
        item["answer_class"] = _five_class(item)
    return readings, {
        "process_records": rows,
        "process_conflicts": [row["faults"] for row in rows],
        "process_diagnoses": diagnoses,
        "process_truncated": truncated,
        "process_by_node": by_node,
        "chain_attributions": chains,
        "candidate_partitions": partitions,
    }


def factor_terminals(nodes, parents_of, gold: dict, baseline: dict) -> tuple[str, ...]:
    """Changed blanks with no changed descendant. These are the factor's sinks."""
    changed = {
        str(node)
        for node in nodes
        if str(baseline.get(node, "")) != str(gold.get(node, ""))
    }
    if not changed:
        return ()
    children: dict[str, list[str]] = {}
    for node in nodes:
        for parent in parents_of(str(node)):
            children.setdefault(str(parent), []).append(str(node))

    def reaches_changed(node: str) -> bool:
        stack = list(children.get(node, ()))
        seen: set[str] = set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            if current in changed:
                return True
            stack.extend(children.get(current, ()))
            seen.add(current)
        return False

    sinks = [node for node in changed if not reaches_changed(node)]
    return tuple(sorted(sinks, key=sort_key))


def nearest_restoring_slots(node: str, gold: dict, sheet: dict, parents_of, becomes_defined) -> list[str]:
    """Closest ancestors whose gold text makes φ a string again."""
    if becomes_defined(sheet):
        return []
    distance: dict[str, int] = {}
    pending = [(str(parent), 1) for parent in parents_of(str(node))]
    while pending:
        item, dist = pending.pop(0)
        if item in distance:
            continue
        distance[item] = dist
        pending.extend((str(parent), dist + 1) for parent in parents_of(item) if str(parent) not in distance)
    found: list[str] = []
    best: int | None = None
    for item, dist in sorted(distance.items(), key=lambda pair: (pair[1], sort_key(pair[0]))):
        if best is not None and dist > best:
            break
        observed = str(sheet.get(item, "") or "")
        gold_text = str(gold.get(item, "") or "")
        if not observed.strip() or observed == gold_text:
            continue
        trial = dict(sheet)
        trial[item] = gold_text
        if not becomes_defined(trial):
            continue
        found.append(item)
        best = dist
    return found


def _attach_nearest(readings: list[dict], gold: dict, sheet: dict, parents_of, becomes_defined) -> None:
    for item in readings:
        if item.get("reading") not in {"contradiction_wrong", "contradiction_correct"}:
            item.setdefault("nearest_slots", [])
            continue
        bare = str(item["node"]).removeprefix("ID") if str(item["node"]).startswith("ID") else str(item["node"])
        slots = nearest_restoring_slots(
            bare, gold, sheet, parents_of, lambda trial, node=bare: becomes_defined(node, trial),
        )
        item["nearest_slots"] = [_label(slot) for slot in slots]


def replacement_nodes(states: dict[str, str], outcomes: dict[str, dict], parents_of, target: str) -> list[str]:
    """Local deviations in the ancestor cone, including the target when it is one."""
    cone = set(_ancestors(parents_of(target), parents_of))
    cone.add(target)
    return [node for node in sorted(cone, key=sort_key) if _must_replace(states.get(node), outcomes.get(node))]


def _repair_rows(states, outcomes, parents_of, targets, answers, gold, same) -> tuple[int | None, list[dict]]:
    """Read K(t) from the local classification. No gold-replay search."""
    rows = []
    cards = []
    for target in targets:
        if target not in answers:
            continue
        observed = "" if answers[target] is None else str(answers[target])
        if observed.strip() and same(target, observed, str(gold[target])):
            continue
        nodes = replacement_nodes(states, outcomes, parents_of, target)
        card = len(nodes) if nodes else None
        rows.append({
            "target": _label(target),
            "cardinality": card,
            "grade": "局部偏差" if nodes else "无法从局部重算读出",
            "sets": [nodes] if nodes else [],
            "direct": target in nodes,
        })
        cards.append(card)
    if not rows or any(card is None for card in cards):
        return None, rows
    return max(cards), rows


def finish_report(
    *,
    symptoms: list[str],
    states: dict[str, str],
    records: list[dict],
    unexplained: list[str],
    unscored: list[str],
    diagnoses: list[list[str]],
    repair_cardinality: int | None,
    repairs: list[dict],
    truncated: bool,
    note: str = "",
    status: str | None = None,
    readings: list[dict] | None = None,
    targets: list[dict] | None = None,
    process_records: list[dict] | None = None,
    process_conflicts: list[list[str]] | None = None,
    process_diagnoses: list[list[str]] | None = None,
    process_truncated: bool = False,
    process_by_node: dict[str, list[str]] | None = None,
    chain_attributions: list[dict] | None = None,
    candidate_partitions: list[dict] | None = None,
) -> dict:
    if status is None:
        if not symptoms and not unexplained:
            status = "unscored" if unscored else "consistent"
        elif records:
            status = "attributed"
        else:
            status = "unmodeled"
    text = note or _NOTES[status]
    if truncated:
        text = text + "全卷碰集在基数 4 内没有覆盖。空的极小碰集是附录，不表示没有故障。终点诊断见 terminal_diagnoses。"
    terminal_diagnoses = [
        {
            "node": item.get("node"),
            "reading": item.get("reading"),
            "chain_class": item.get("chain_class"),
            "matches_gold": item.get("matches_gold"),
            "causal": list(item.get("causal_candidates") or []),
            "nearest_slots": list(item.get("nearest_slots") or []),
            "side_paths": list(item.get("side_paths") or []),
        }
        for item in (targets or [])
    ]
    return {
        "status": status,
        "symptoms": list(symptoms),
        "states": {_label(node): state for node, state in sorted(states.items(), key=lambda item: sort_key(item[0])) if state != "correct"},
        "correct_count": sum(state == "correct" for state in states.values()),
        "conflicts": [list(record["faults"]) for record in records],
        "conflict_records": records,
        "minimal_diagnoses": diagnoses,
        "validated_diagnoses": [],
        "validation": [],
        "diagnosis_cardinality": None,
        "repair_cardinality": repair_cardinality,
        "repairs": repairs,
        "unexplained_symptoms": unexplained,
        "unscored": unscored,
        "ambiguity_groups": [],
        "truncated": truncated,
        "note": text,
        "readings": list(readings or []),
        "targets": list(targets or []),
        "process_records": list(process_records or []),
        "process_conflicts": list(process_conflicts or []),
        "process_diagnoses": list(process_diagnoses or []),
        "process_truncated": process_truncated,
        "process_by_node": dict(process_by_node or {}),
        "chain_attributions": list(chain_attributions or []),
        "candidate_partitions": list(candidate_partitions or []),
        "terminal_diagnoses": terminal_diagnoses,
    }


def _prepare_targets(view: DomainView, targets: tuple[str, ...], baseline: dict | None) -> tuple[str, ...]:
    if baseline is None:
        return targets
    sinks = factor_terminals(view.node_ids, view.parents, view.gold(), baseline)
    if sinks:
        return sinks
    return tuple(node for node in targets if tuple(view.parents(node)))


def explain(
    view: DomainView,
    answers: dict[str, str],
    targets: tuple[str, ...] = (),
    *,
    prefix: str = "model",
    baseline: dict | None = None,
) -> dict:
    """One trajectory. Local φ, graph candidates, then the attributable split."""
    targets = _prepare_targets(view, targets, baseline)
    if prefix != "model":
        return finish_report(
            symptoms=[], states={}, records=[], unexplained=[], unscored=[],
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, status="gold_prefix",
        )
    states, unscored, phi, outcomes = _states(view, answers)
    readings, _target_rows = build_readings(outcomes, view.parents, targets)
    gold = view.gold()
    sheet = answer_sheet(answers)
    wanted = {_label(node) for node in targets}

    def becomes_defined(node: str, trial: dict) -> bool:
        return isinstance(node_phi(view, node, trial), str)

    def close(found: list[dict]) -> list[dict]:
        _attach_nearest(found, gold, sheet, view.parents, becomes_defined)
        return [item for item in found if item["node"] in wanted]

    sources = _process_sources(states, outcomes)
    actionable = any(
        state in _WRONG and not (
            state == "contract_contradiction"
            and _same(view, node, str(answers.get(node) or ""), str(gold[node]))
        )
        for node, state in states.items()
    )
    if not actionable and not sources:
        readings, process = attach_process(readings, states, outcomes, {}, view.parents)
        target_rows = close(readings)
        return finish_report(
            symptoms=[], states=states, records=[], unexplained=[], unscored=unscored,
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, readings=readings, targets=target_rows, **process,
        )
    by_id, effects, injected = _library(view)
    catalog = {fault_id: frozenset(node for node, _value in fault.injector) for fault_id, fault in by_id.items()}
    readings, process = attach_process(readings, states, outcomes, catalog, view.parents)
    target_rows = close(readings)
    if not actionable:
        return finish_report(
            symptoms=[], states=states, records=[], unexplained=[], unscored=unscored,
            diagnoses=[], repair_cardinality=None, repairs=[],
            truncated=False, readings=readings, targets=target_rows, **process,
        )
    records, symptoms, unexplained = _collect(view, answers, states, phi, effects, injected)
    diagnoses, truncated = minimal_hitting_sets([frozenset(record["faults"]) for record in records])
    repair_cardinality, repairs = _repair_rows(
        states, outcomes, view.parents, targets, answers, view.gold(),
        lambda node, left, right: _same(view, node, left, right),
    )
    return finish_report(
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
    )


def attach(cases, report: dict):
    """Copy each node's candidate rows onto the local case. Classes stay as they were."""
    from dataclasses import replace

    by_node: dict[str, list[tuple[str, ...]]] = {}
    for record in report.get("conflict_records") or []:
        by_node.setdefault(record["node"], []).append(tuple(record["faults"]))
    process = report.get("process_by_node") or {}
    causal = {
        row["target"]: tuple(row.get("causal") or ())
        for row in report.get("candidate_partitions") or []
    }
    updated = []
    for case in cases:
        groups = by_node.get(case.node)
        process_faults = tuple(process.get(case.node) or ())
        attributable = causal.get(case.node) or ()
        if not groups and not process_faults and not attributable:
            updated.append(case)
            continue
        updated.append(replace(
            case,
            conflict_sets=tuple(groups or ()),
            covering_diagnoses=(attributable,) if attributable else (),
            process_faults=process_faults,
        ))
    return tuple(updated)


def assert_consistent_upstream(view: DomainView, report: dict) -> None:
    """A measurement that follows φ does not list a fault injected at its own node."""
    _by_id, _effects, injected = _library(view)
    for record in report.get("conflict_records") or []:
        if record["state"] != "wrong_but_locally_consistent":
            continue
        node = record["node"]
        for fault in record["faults"]:
            if node in (injected.get(fault) or ()):
                raise RuntimeError(f"{view.name} { _label(node) } 与重算一致，冲突集仍含本节点注入 {fault}")


def assert_report(report: dict) -> None:
    status = report["status"]
    if status not in _STATUSES:
        raise RuntimeError(f"诊断状态无法识别：{status}")
    conflicts = [set(item) for item in report["conflicts"]]
    if any(not item for item in conflicts):
        raise RuntimeError("冲突集是空的")
    minimal = [list(item) for item in report["minimal_diagnoses"]]
    for diagnosis in minimal:
        block = set(diagnosis)
        if conflicts and not all(block & item for item in conflicts):
            raise RuntimeError("极小碰集没有击中全部冲突集")
    if report["validation"] or report["validated_diagnoses"]:
        raise RuntimeError("建模系统不再重放见证文")
    if report["diagnosis_cardinality"] is not None:
        raise RuntimeError("诊断基数来自已删除的见证文重放")
    if status == "attributed" and not report["conflict_records"]:
        raise RuntimeError("已归因状态没有候选记录")
    if status == "unmodeled" and report["conflict_records"]:
        raise RuntimeError("未建模状态仍有候选记录")
    if status in {"consistent", "unscored", "gold_prefix"} and report["symptoms"]:
        raise RuntimeError(f"{status} 仍带有症状")
    for row in report.get("candidate_partitions") or []:
        parts = [set(row["causal"]), set(row["associative"]), set(row["unresolved"])]
        if set(row["candidates"]) != set().union(*parts):
            raise RuntimeError(f"{row['target']} 的候选集没有被切完")
        if parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2]:
            raise RuntimeError(f"{row['target']} 的候选子集重叠")


def assert_root_explained(view: DomainView, answers: dict[str, str], report: dict, root: str) -> None:
    """A library fault that writes the injected root stays in the causal candidates."""
    assert_report(report)
    by_id, effects, _injected = _library(view)
    observed = "" if answers.get(root) is None else str(answers[root])
    matching = []
    for fault_id, fault in by_id.items():
        if fault_id not in effects:
            continue
        writes = dict(fault.injector)
        if root not in writes or not _same(view, root, writes[root], observed):
            continue
        if any(
            node != root and not _same(view, node, value, str(answers.get(node, view.gold()[node])))
            for node, value in writes.items()
        ):
            continue
        matching.append(fault_id)
    if not matching:
        return
    if report["status"] != "attributed":
        raise RuntimeError(f"{view.name} 注入了库内故障，候选集却是空的：{report['status']}")
    causal = [
        fault
        for row in report.get("candidate_partitions") or []
        for fault in row.get("causal") or []
    ]
    if not set(causal) & set(matching):
        raise RuntimeError(f"{view.name} 的可归因候选没有包含写入 {_label(root)} 的故障实例")
