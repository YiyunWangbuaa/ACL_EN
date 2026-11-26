# -*- coding: utf-8 -*-
"""Model-trajectory attribution for domains that share DomainView.

diagnose classifies each blank from the gold residual and φ on the model's
contract cognition plus the observed parents. One parent subset is propagated;
several stay unresolved.
annotate_trajectory then attaches the candidate-set report from
conflict_diagnosis. That report does not replace the local class.
observation_fixes reads the local-deviation blanks in each target's ancestor
cone. Restoring a terminal is domains.repair_search, not this readout.

Wind-tunnel trajectories use domains.wind_tunnel.conflict_diagnosis for the
same candidate report. Their sheet loader stays in the wind-tunnel package
because the contract executor is not a DomainView.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from domains.conflict_diagnosis import node_phi, target_note
from domains.contract_cognition import answer_sheet
from domains.phi_result import Contradiction
from domains.mbd_kernel import DomainView, canon_format, sort_key


@dataclass(frozen=True)
class Case:
    node: str
    origin: str
    final_class: str
    minimal_fault_set: tuple[str, ...] = ()
    candidate_fault_sets: tuple[tuple[str, ...], ...] = ()
    propagated_from: tuple[str, ...] = ()
    upstream_wrong: tuple[str, ...] = ()
    mixed: bool = False
    phi: str = ""
    prefix: str = "model"
    note: str = ""
    conflict_sets: tuple[tuple[str, ...], ...] = ()
    covering_diagnoses: tuple[tuple[str, ...], ...] = ()
    process_faults: tuple[str, ...] = ()


@dataclass(frozen=True)
class TargetFix:
    target: str
    direct: bool
    upstream: tuple[str, ...]
    single: tuple[tuple[str, str], ...]
    grade: str
    minimal: tuple[tuple[str, ...], ...]


def load_answers(path: Path | str) -> dict[str, str]:
    """One model trajectory. Accepts prediction or model_answer rows."""
    found: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("branch") == "probe":
            continue
        raw_id = row.get("node_id", row.get("blank_id", ""))
        node = str(raw_id).removeprefix("ID")
        if not node:
            continue
        if "prediction" in row:
            value = row.get("prediction")
        elif "model_answer" in row:
            value = row.get("model_answer")
        else:
            continue
        found[node] = "" if value is None else str(value)
    if not found:
        raise RuntimeError(f"{path} has no answer rows")
    return found


def diagnose(view: DomainView, answers: dict[str, str], *, prefix: str = "model") -> tuple[Case, ...]:
    """Classify one trajectory and attach parent-subset fault sets."""
    if prefix not in {"model", "gold"}:
        raise RuntimeError(f"Prefix must be model or gold, got {prefix!r}")
    gold = view.gold()
    present = [node for node in view.node_ids if node in answers]
    correct: dict[str, bool] = {}
    blank: set[str] = set()
    for node in present:
        raw = "" if answers[node] is None else str(answers[node])
        if not raw.strip():
            blank.add(node)
            correct[node] = False
            continue
        correct[node] = _same(view, node, raw, gold[node])
    cases: list[Case] = []
    _sheet = answer_sheet(answers)
    for node in present:
        raw = "" if answers[node] is None else str(answers[node])
        parents = view.parents(node)
        if node in blank:
            cases.append(Case(
                node, "wrong_answer", "UNPARSEABLE_PENDING_REVIEW",
                prefix=prefix, note="empty answer",
            ))
            continue
        if prefix != "model":
            if not correct[node]:
                cases.append(Case(node, "wrong_answer", "CONDITIONAL_FAILURE", prefix=prefix))
            continue
        wrong = tuple(parent for parent in parents if parent in correct and not correct[parent])
        if not parents:
            if correct[node]:
                continue
            kind = "ROOT_ERROR" if node in view.declaration_roots else "NEW_LOCAL_ERROR"
            cases.append(Case(
                node, "wrong_answer", kind, minimal_fault_set=(node,), prefix=prefix,
            ))
            continue
        if correct[node] and not wrong:
            phi = node_phi(view, node, _sheet)
            if isinstance(phi, Contradiction):
                cases.append(Case(
                    node, "latent", "UNRELIABLE_CORRECT",
                    prefix=prefix, note="chance_correct",
                ))
            elif any(not correct.get(item, True) for item in _ancestors(view, node)):
                cases.append(Case(
                    node, "latent", "UNRELIABLE_CORRECT",
                    prefix=prefix, note="traceable_masking",
                ))
            continue
        phi = node_phi(view, node, _sheet)
        if isinstance(phi, Contradiction):
            if correct[node]:
                cases.append(Case(
                    node, "latent", "UNRELIABLE_CORRECT",
                    upstream_wrong=wrong, prefix=prefix, note="chance_correct",
                ))
            else:
                cases.append(Case(
                    node, "wrong_answer", "UNRESOLVED",
                    candidate_fault_sets=_contradiction_sets(wrong, node),
                    upstream_wrong=wrong, prefix=prefix, note="local contradiction",
                ))
            continue
        if phi is None:
            if not correct[node]:
                cases.append(Case(
                    node, "wrong_answer", "UNRESOLVED",
                    upstream_wrong=wrong, prefix=prefix, note="calculator refused",
                ))
            continue
        matches = _same(view, node, raw, phi)
        phi_text = str(phi)
        if matches and not correct[node]:
            if not wrong:
                cases.append(Case(
                    node, "wrong_answer", "UNRESOLVED",
                    phi=phi_text, prefix=prefix, note="no wrong parent",
                ))
                continue
            observed_parents = {
                parent: str(answers[parent]).strip()
                if parent in answers and str(answers[parent] or "").strip()
                else str(gold[parent])
                for parent in parents
            }
            kernels = _kernels(view, node, parents, wrong, raw, observed_parents, _sheet)
            cases.append(_propagation(node, wrong, kernels, phi_text, prefix))
            continue
        if matches and correct[node]:
            cases.append(Case(
                node, "latent", "RESULT_EQUIVALENCE",
                minimal_fault_set=wrong, upstream_wrong=wrong,
                mixed=len(wrong) > 1, phi=phi_text, prefix=prefix,
            ))
            continue
        if correct[node]:
            cases.append(Case(
                node, "latent", "CONTRACT_TARGET_DISCONNECT",
                upstream_wrong=wrong, phi=phi_text, prefix=prefix,
            ))
            continue
        cases.append(Case(
            node, "wrong_answer", "NEW_LOCAL_ERROR",
            minimal_fault_set=(node,), upstream_wrong=wrong,
            phi=phi_text, prefix=prefix,
        ))
    return tuple(cases)


def annotate_trajectory(
    view: DomainView,
    answers: dict[str, str],
    targets: tuple[str, ...],
    *,
    prefix: str = "model",
    baseline: dict | None = None,
) -> tuple[Case, ...]:
    """Local classes for every node, then the candidate-set report.

    The report is attached to each wrong node it mentions. It does not replace
    a local propagation label.
    """
    cases = diagnose(view, answers, prefix=prefix)
    if prefix != "model":
        return cases
    from domains.conflict_diagnosis import attach, explain

    return attach(cases, explain(view, answers, targets, prefix=prefix, baseline=baseline))


def observation_fixes(
    view: DomainView,
    answers: dict[str, str],
    targets: tuple[str, ...],
    *,
    prefix: str = "model",
) -> tuple[TargetFix, ...]:
    """Local-deviation nodes in each wrong target's ancestor cone."""
    if prefix != "model":
        return ()
    from domains.conflict_diagnosis import _states, replacement_nodes

    states, _unscored, _phi, outcomes = _states(view, answers)
    gold = view.gold()
    found: list[TargetFix] = []
    for target in targets:
        if target not in answers:
            continue
        observed = "" if answers[target] is None else str(answers[target])
        if observed.strip() and _same(view, target, observed, str(gold[target])):
            continue
        nodes = tuple(replacement_nodes(states, outcomes, view.parents, target))
        upstream = tuple(node for node in nodes if node != target)
        found.append(TargetFix(
            target,
            target in nodes,
            upstream,
            (),
            "local deviation" if nodes else "not readable from local recomputation",
            (nodes,) if nodes else (),
        ))
    return tuple(found)


def audit_control(view: DomainView, targets: tuple[str, ...]) -> dict:
    """Control trajectories. No model call.

    One declaration is replaced and every derived blank follows φ.
    One derived blank is replaced while its parents stay at gold.
    """
    root, faithful, cases, fixes = _faithful_root(view, targets)
    local_node, local_targets, local_answers = _assert_local(view, targets)
    from domains.conflict_diagnosis import (
        assert_consistent_upstream,
        assert_report,
        assert_root_explained,
        attach,
        explain,
    )

    report = explain(view, faithful, targets, prefix="model")
    assert_report(report)
    assert_root_explained(view, faithful, report, root)
    assert_consistent_upstream(view, report)
    if report["repair_cardinality"] != 1:
        raise RuntimeError(f"{view.name} faithful trajectory repair cardinality is not 1: {report['repair_cardinality']}")
    classes = {case.node: case.final_class for case in cases}
    for case in attach(cases, report):
        if classes[case.node] != case.final_class:
            raise RuntimeError(f"{view.name} {_tag(case.node)} local class was overwritten by the candidate report")
    gold_report = explain(view, faithful, targets, prefix="gold")
    assert_report(gold_report)
    if gold_report["status"] != "gold_prefix":
        raise RuntimeError(f"{view.name} gold prefix produced a diagnosis: {gold_report['status']}")
    clean = explain(view, dict(view.gold()), targets)
    assert_report(clean)
    if clean["status"] != "consistent":
        raise RuntimeError(f"{view.name} all-gold trajectory is not consistent: {clean['status']}")
    local_report = explain(view, local_answers, targets, prefix="model")
    assert_report(local_report)
    assert_consistent_upstream(view, local_report)
    summary = _summary(cases, fixes)
    summary["root"] = root
    summary["local"] = local_node
    summary["conflict_status"] = report["status"]
    summary["diagnosis_cardinality"] = report["diagnosis_cardinality"]
    summary["repair_cardinality"] = report["repair_cardinality"]
    summary["targets"] = report["targets"]
    flagged = target_note(report["targets"])
    print(
        f"Observation attribution: {view.name} faithful trajectory {_tag(root)}, "
        f"propagated {summary['propagated']}, latent {summary['latent']}, "
        f"unresolved {summary['unresolved']}, terminals {_fix_text(fixes)}. "
        f"Local trajectory {_tag(local_node)} is a local error"
        + (f", repaired {_join(local_targets)}" if local_targets else ", and the terminal is that node")
        + f". Stage 5 candidates {report['status']}, "
        f"repair cardinality {report['repair_cardinality']}, read from local deviations."
        + (f" {flagged}" if flagged else "")
        + " A gold prefix does not produce a diagnosis.",
        flush=True,
    )
    return summary


def _contradiction_sets(wrong: tuple[str, ...], node: str) -> tuple[tuple[str, ...], ...]:
    """Propagation from the wrong parents, together with a local EXE at this node."""
    found = []
    if wrong:
        found.append(tuple(wrong))
    found.append((node,))
    return tuple(found)


def _propagation(
    node: str,
    wrong: tuple[str, ...],
    kernels: tuple[tuple[str, ...], ...] | None,
    phi: str,
    prefix: str,
) -> Case:
    if kernels is None:
        return Case(
            node, "wrong_answer", "UNRESOLVED",
            upstream_wrong=wrong, phi=phi, prefix=prefix, note="too many parents",
        )
    if len(kernels) == 1:
        only = kernels[0]
        return Case(
            node, "wrong_answer", "PROPAGATED_ERROR",
            minimal_fault_set=only, propagated_from=only, upstream_wrong=wrong,
            mixed=len(only) > 1, phi=phi, prefix=prefix,
        )
    if not kernels:
        return Case(
            node, "wrong_answer", "UNRESOLVED",
            upstream_wrong=wrong, phi=phi, prefix=prefix, note="empty kernel",
        )
    return Case(
        node, "wrong_answer", "UNRESOLVED",
        candidate_fault_sets=kernels, upstream_wrong=wrong,
        phi=phi, prefix=prefix,
    )


def _kernels(
    view: DomainView,
    node: str,
    parents: tuple[str, ...],
    wrong: tuple[str, ...],
    observed: str,
    observed_parents: dict[str, str],
    sheet: dict[str, str],
) -> tuple[tuple[str, ...], ...] | None:
    if len(wrong) > 8:
        return None
    gold = view.gold()

    def run(active: set[str]) -> str | None:
        trial = dict(sheet)
        for parent in parents:
            trial[parent] = observed_parents[parent] if parent in active else str(gold[parent])
        return view.answer_under(node, trial)

    def matches(active: set[str]) -> bool:
        predicted = run(active)
        if predicted is None:
            return False
        return _same(view, node, observed, predicted)

    if not matches(set(wrong)):
        return ()
    sufficient: list[set[str]] = []
    for size in range(len(wrong) + 1):
        for combo in combinations(wrong, size):
            if matches(set(combo)):
                sufficient.append(set(combo))
        if sufficient:
            break
    minimal = [
        tuple(sorted(item, key=sort_key))
        for item in sufficient
        if not any(other < item for other in sufficient)
    ]
    return tuple(minimal)


def _faithful_root(view: DomainView, targets: tuple[str, ...]):
    gold = view.gold()
    last_error = f"{view.name} has no declaration root that changes a terminal"
    for node, alt in _root_alts(view):
        answers = dict(gold)
        answers[node] = alt
        _follow(view, answers, dirty={node}, frozen={node})
        if not any(not _same(view, target, answers[target], gold[target]) for target in targets if target in answers):
            continue
        cases = diagnose(view, answers, prefix="model")
        try:
            _assert_faithful(view, answers, cases, node)
            _assert_gold_prefix(answers, view, diagnose(view, answers, prefix="gold"))
            fixes = observation_fixes(view, answers, targets)
            _assert_root_repairs(view, answers, fixes, node, targets)
            _assert_disconnect(view, answers, cases)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        return node, answers, cases, fixes
    raise RuntimeError(last_error)


def _assert_faithful(view: DomainView, answers: dict[str, str], cases: tuple[Case, ...], root: str) -> None:
    gold = view.gold()
    wrong = [
        node for node in view.node_ids
        if node in answers and not _same(view, node, answers[node], gold[node])
    ]
    wrong_cases = [case for case in cases if case.origin == "wrong_answer"]
    if {case.node for case in wrong_cases} != set(wrong):
        missing = set(wrong) - {case.node for case in wrong_cases}
        extra = {case.node for case in wrong_cases} - set(wrong)
        raise RuntimeError(f"{view.name} wrong blanks and cases disagree: missing {sorted(missing)} extra {sorted(extra)}")
    if len({case.node for case in cases}) != len(cases):
        raise RuntimeError(f"{view.name} has more than one observation case for the same node")
    roots = [case for case in wrong_cases if case.final_class == "ROOT_ERROR"]
    if len(roots) != 1 or roots[0].node != root:
        raise RuntimeError(f"{view.name} faithful trajectory root is not {_tag(root)}")
    if roots[0].minimal_fault_set != (root,):
        raise RuntimeError(f"{view.name} {_tag(root)} minimal set is not itself")
    locals_ = [case.node for case in wrong_cases if case.final_class == "NEW_LOCAL_ERROR"]
    if locals_:
        raise RuntimeError(f"{view.name} faithful trajectory has a local error: {locals_}")
    for case in wrong_cases:
        if case.final_class == "PROPAGATED_ERROR" and not case.propagated_from:
            raise RuntimeError(f"{view.name} {_tag(case.node)} is marked propagated but has no parent")
        if case.final_class == "UNRESOLVED" and not case.candidate_fault_sets and case.note not in {"calculator refused", "local contradiction", "too many parents", "no wrong parent", "empty kernel"}:
            raise RuntimeError(f"{view.name} {_tag(case.node)} is unresolved but has no candidate kernel")
        if case.final_class not in {"ROOT_ERROR", "PROPAGATED_ERROR", "UNRESOLVED"}:
            raise RuntimeError(f"{view.name} {_tag(case.node)} class is {case.final_class}")
    for case in cases:
        if case.origin == "latent" and case.final_class not in {"RESULT_EQUIVALENCE", "CONTRACT_TARGET_DISCONNECT", "UNRELIABLE_CORRECT"}:
            raise RuntimeError(f"{view.name} {_tag(case.node)} latent class is {case.final_class}")


def _assert_gold_prefix(answers: dict[str, str], view: DomainView, cases: tuple[Case, ...]) -> None:
    gold = view.gold()
    wrong = [
        node for node in view.node_ids
        if node in answers and not _same(view, node, answers[node], gold[node])
    ]
    if {case.node for case in cases} != set(wrong):
        raise RuntimeError(f"{view.name} gold prefix does not cover the wrong blanks")
    bad = [case.node for case in cases if case.final_class != "CONDITIONAL_FAILURE"]
    if bad:
        raise RuntimeError(f"{view.name} gold prefix marked {_tag(bad[0])} as propagated or root")


def _assert_root_repairs(
    view: DomainView,
    answers: dict[str, str],
    fixes: tuple[TargetFix, ...],
    root: str,
    targets: tuple[str, ...],
) -> None:
    gold = view.gold()
    children = _children(view)
    reached = set(_descendants(children, root))
    checked = 0
    for fix in fixes:
        if fix.target not in targets or fix.target not in reached:
            continue
        if _same(view, fix.target, answers[fix.target], gold[fix.target]):
            continue
        checked += 1
        if (root,) not in fix.minimal:
            raise RuntimeError(f"{view.name} repairing {_tag(root)} did not restore {_tag(fix.target)} ({fix.grade})")
    if checked == 0:
        raise RuntimeError(f"{view.name} injecting {_tag(root)} left no terminal to repair")


def _assert_disconnect(view: DomainView, answers: dict[str, str], cases: tuple[Case, ...]) -> None:
    gold = view.gold()
    propagated = next(
        (case.node for case in cases if case.final_class == "PROPAGATED_ERROR" and case.phi and not _same(view, case.node, case.phi, gold[case.node])),
        "",
    )
    if not propagated:
        return
    tweaked = dict(answers)
    tweaked[propagated] = gold[propagated]
    found = next(case for case in diagnose(view, tweaked) if case.node == propagated)
    if found.origin != "latent" or found.final_class != "CONTRACT_TARGET_DISCONNECT":
        raise RuntimeError(f"{view.name} {_tag(propagated)} after the answer returns to gold is not disconnected; it is {found.final_class}")


def _assert_local(view: DomainView, targets: tuple[str, ...]) -> tuple[str, tuple[str, ...], dict[str, str]]:
    gold = view.gold()
    derived = [node for node in view.node_ids if view.parents(node)]
    for node in sorted(derived, key=sort_key):
        alt = _alt(view, node)
        if alt is None:
            continue
        answers = dict(gold)
        answers[node] = alt
        _follow(view, answers, dirty={node}, frozen={node})
        cases = diagnose(view, answers)
        own = next((case for case in cases if case.node == node), None)
        if own is None or own.final_class != "NEW_LOCAL_ERROR" or own.minimal_fault_set != (node,):
            continue
        moved = tuple(
            target for target in targets
            if target != node and target in answers and not _same(view, target, answers[target], gold[target])
        )
        if not moved:
            continue
        fixes = {fix.target: fix for fix in observation_fixes(view, answers, targets)}
        if all((node,) in fixes.get(target, TargetFix(target, False, (), (), "", ())).minimal for target in moved):
            return node, moved, answers
    for node in targets:
        if not view.parents(node):
            continue
        alt = _alt(view, node)
        if alt is None:
            continue
        answers = dict(gold)
        answers[node] = alt
        cases = diagnose(view, answers)
        own = next(case for case in cases if case.node == node)
        if own.final_class != "NEW_LOCAL_ERROR":
            continue
        fixes = observation_fixes(view, answers, targets)
        own_fix = next(fix for fix in fixes if fix.target == node)
        if own_fix.direct and own_fix.minimal == ((node,),):
            return node, (), answers
    raise RuntimeError(f"{view.name} has no local execution error that can be attributed on its own")


def _root_alts(view: DomainView) -> tuple[tuple[str, str], ...]:
    derived_ancestors: set[str] = set()
    for node in view.node_ids:
        if view.parents(node):
            derived_ancestors.update(_ancestors(view, node))
    found = []
    for node in sorted(derived_ancestors, key=sort_key):
        if node not in view.declaration_roots or view.parents(node):
            continue
        alt = _alt(view, node)
        if alt is not None:
            found.append((node, alt))
    return tuple(found)


def _follow(view: DomainView, answers: dict[str, str], *, dirty: set[str], frozen: set[str]) -> None:
    """Rewrite derived blanks that follow a dirty parent. Frozen blanks stay."""
    gold = view.gold()
    derived = [node for node in view.node_ids if view.parents(node) and node not in frozen]
    pending = set(dirty)
    for node in _topo(view, derived):
        if not (set(view.parents(node)) & pending):
            continue
        predicted = view.answer_under(node, answers)
        if predicted is None:
            continue
        answers[node] = str(predicted)
        if _same(view, node, answers[node], gold[node]):
            pending.discard(node)
        else:
            pending.add(node)


def _summary(cases: tuple[Case, ...], fixes: tuple[TargetFix, ...]) -> dict:
    return {
        "root_errors": sum(case.final_class == "ROOT_ERROR" for case in cases),
        "propagated": sum(case.final_class == "PROPAGATED_ERROR" for case in cases),
        "local": sum(case.final_class == "NEW_LOCAL_ERROR" for case in cases),
        "latent": sum(case.origin == "latent" for case in cases),
        "unresolved": sum(case.final_class == "UNRESOLVED" for case in cases),
        "conditional": sum(case.final_class == "CONDITIONAL_FAILURE" for case in cases),
        "repairs": tuple((fix.target, fix.grade, fix.minimal) for fix in fixes),
    }


def _alt(view: DomainView, node: str) -> str | None:
    gold = str(view.gold()[node])
    options = [str(item) for item in (view.question(node).get("options") or []) if str(item) not in {"", "none", "\u65e0"}]
    for option in options:
        if not _same(view, node, option, gold):
            return option
    if options:
        return None
    fmt = canon_format(str(view.question(node).get("format") or ""))
    slipped = _slip(gold, fmt)
    if _same(view, node, slipped, gold):
        return None
    return slipped


def _slip(gold: str, fmt: str) -> str:
    text = gold.strip()
    if fmt == "matrix" and text.startswith("[["):
        return text.replace("1", "0", 1) if "1" in text else text[:-2] + ", 0]]"
    if fmt == "vector" and text.startswith("["):
        return "[0, 0, 0]" if text != "[0, 0, 0]" else "[1, 0, 0]"
    if text.startswith("-"):
        return text[1:].strip() or "0"
    return f"-({text})" if text else "0"


def _same(view: DomainView, node: str, left: str, right: str) -> bool:
    if str(left) == str(right):
        return True
    return view.equal(str(left), str(right), node)


def _children(view: DomainView) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for node in view.node_ids:
        for parent in view.parents(node):
            grouped[parent].append(node)
    return {parent: tuple(nodes) for parent, nodes in grouped.items()}


def _ancestors(view: DomainView, node: str) -> set[str]:
    seen: set[str] = set()
    pending = list(view.parents(node))
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pending.extend(view.parents(item))
    return seen


def _descendants(children: dict[str, tuple[str, ...]], node: str) -> set[str]:
    seen: set[str] = set()
    pending = list(children.get(node, ()))
    while pending:
        item = pending.pop()
        if item in seen:
            continue
        seen.add(item)
        pending.extend(children.get(item, ()))
    return seen


def _topo(view: DomainView, nodes: list[str]) -> tuple[str, ...]:
    pending = {node: set(view.parents(node)) & set(nodes) for node in nodes}
    ordered: list[str] = []
    while pending:
        ready = sorted([node for node, deps in pending.items() if not deps], key=sort_key)
        if not ready:
            ready = sorted(pending, key=sort_key)[:1]
        for node in ready:
            ordered.append(node)
            pending.pop(node, None)
            for deps in pending.values():
                deps.discard(node)
    return tuple(ordered)


def _tag(node: str) -> str:
    text = str(node)
    if text[:1].isalpha():
        return text
    return f"ID{text}"


def _join(nodes: tuple[str, ...]) -> str:
    return ", ".join(_tag(node) for node in nodes)


def _fix_text(fixes: tuple[TargetFix, ...]) -> str:
    if not fixes:
        return "none"
    parts = []
    for fix in fixes:
        parts.append(f"{_tag(fix.target)} {fix.grade}")
    return ", ".join(parts)
