# -*- coding: utf-8 -*-
"""Overlapping attribution. The question layer sets the local fault mode.

Fill every input with the prefix the model actually saw and compute that
conditional output. A match means this blank followed its prefix, so it adds
no local fault. A mismatch is scored only in this question's class: DIR on
direct recognition, RSN on a route the contract implies, EXE on execution,
ALG on calculation. Answer shape does not move a question between EXE and
ALG. Upstream factors stay, each in its own class.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

from cloze_score import sympy_equal
from domains.contract_cognition import working_texts
from domains.wind_tunnel.contract_cognition import recognition_ids
from domains.wind_tunnel.contract_verifiers import derive, input_ids, validator_of
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.error_workflow import PHI, _phi
from domains.wind_tunnel.eval_scope import scope_of

ROOT = Path(__file__).resolve().parents[2]
OBS = ROOT / "results" / "contract_consistency" / "error_workflow"
_TOO_MANY = 8


def _bare(value) -> str:
    return str(value).removeprefix("ID")


def _same(left: str, right: str) -> bool:
    if str(left).strip() == str(right).strip():
        return True
    return sympy_equal(left, right)


def _parents(node: str) -> tuple[str, ...]:
    if node in PHI:
        return PHI[node][2]
    registered = input_ids(node)
    if registered:
        return registered
    return ()


def _has_calculator(node: str) -> bool:
    return node in PHI or validator_of(node) is not None


@dataclass(frozen=True)
class Factor:
    node: str
    modes: tuple[str, ...]

    def text(self) -> str:
        return f"{'+'.join(self.modes)} ID{self.node}"


@dataclass(frozen=True)
class AnswerFaults:
    node: str
    role: str
    phi_status: str
    local: tuple[str, ...]
    cause_sets: tuple[tuple[Factor, ...], ...]
    direct: tuple[tuple[str, ...], ...]
    note: str

    def modes(self) -> tuple[str, ...]:
        found = []
        for group in self.cause_sets:
            for factor in group:
                for mode in factor.modes:
                    if mode not in found:
                        found.append(mode)
        return tuple(found)


class _Context:
    def __init__(self, model, gold: dict[str, str], observed: dict[str, dict], prefix: str):
        self.model = model
        self.gold = gold
        self.observed = observed
        self.prefix = prefix
        self.cache: dict[str, AnswerFaults] = {}
        self.stack: set[str] = set()
        self.calc_cache: dict[tuple[str, tuple[str, ...]], tuple[bool, str]] = {}

    def stated(self, node: str) -> str:
        if self.prefix == "gold":
            return self.gold[node]
        item = self.observed.get(node)
        if item is None or not str(item.get("model_answer") or "").strip():
            return self.gold[node]
        return str(item["model_answer"])

    def written(self, node: str) -> str:
        item = self.observed.get(node)
        if item is None:
            return ""
        return str(item.get("model_answer") or "")

    def calculate(self, node: str, active: tuple[str, ...]) -> tuple[bool, str]:
        key = (node, active)
        cached = self.calc_cache.get(key)
        if cached is not None:
            return cached
        recognition = recognition_ids()
        owned = PHI[node][2] if node in PHI else _parents(node)
        sheet = {slot: self.stated(slot) for slot in recognition}
        for parent in owned:
            sheet[parent] = self.stated(parent) if parent in active else self.gold[parent]
        texts = working_texts(self.gold, sheet, recognition, owned)
        try:
            if node in PHI:
                derived = _phi(self.model, node, [texts[parent] for parent in PHI[node][2]], texts)
            else:
                derived = derive(self.model, node, texts)
        except Exception as exc:
            result = (False, str(exc))
            self.calc_cache[key] = result
            return result
        if derived is None or not derived.ok:
            reason = "" if derived is None else derived.reason
            result = (False, reason)
        else:
            result = (True, derived.matrix)
        self.calc_cache[key] = result
        return result


def _local_modes(node: str, matched: bool) -> tuple[str, ...]:
    """One layer per question. A matched conditional output clears it."""
    if matched:
        return ()
    return (scope_of(node),)


def _minimal(wrong: list[str], holds) -> tuple[list[tuple[str, ...]], list[str]]:
    singles = [item for item in wrong if holds((item,))]
    rest = [item for item in wrong if item not in singles]
    found = [{item} for item in singles]
    if len(rest) > _TOO_MANY:
        return [tuple(sorted(item, key=int)) for item in found], rest
    for size in range(1, len(rest) + 1):
        for combo in combinations(rest, size):
            if holds(combo):
                found.append(set(combo))
    minimal = [
        tuple(sorted(item, key=int))
        for item in found
        if not any(other < item for other in found)
    ]
    return minimal, []


def _unique(sets: list[tuple[Factor, ...]]) -> tuple[tuple[Factor, ...], ...]:
    keyed = []
    for group in sets:
        key = tuple((factor.node, factor.modes) for factor in group)
        if key not in keyed:
            keyed.append(key)
    ordered = [tuple(Factor(node, modes) for node, modes in key) for key in keyed]
    kept = []
    for group in ordered:
        nodes = {factor.node for factor in group}
        if any(other < nodes for other in ({factor.node for factor in item} for item in ordered)):
            continue
        kept.append(group)
    return tuple(kept)


def _expand(ctx: _Context, groups: list[tuple[str, ...]], through_rejection: bool) -> list[tuple[Factor, ...]]:
    """Replace each input by its own causes.

    A downstream calculator that succeeded used the written text. A rejected
    input then stays on that blank: its inputs did not produce the text.
    """
    expanded = []
    for group in groups:
        parts = []
        for parent in group:
            child = explain(ctx, parent)
            if child.phi_status == "rejected" and not through_rejection:
                parts.append(((Factor(parent, ("undecided",)),),))
            else:
                parts.append(child.cause_sets or ((Factor(parent, child.local or ("undecided",)),),))
        for combo in product(*parts):
            bag: dict[str, tuple[str, ...]] = {}
            for piece in combo:
                for factor in piece:
                    bag[factor.node] = factor.modes
            expanded.append(tuple(Factor(item, bag[item]) for item in sorted(bag, key=int)))
    return expanded


def explain(ctx: _Context, node: str) -> AnswerFaults:
    cached = ctx.cache.get(node)
    if cached is not None:
        return cached
    role = NODE_BY_ID[node].role if node in NODE_BY_ID else ""
    if node in ctx.stack:
        found = AnswerFaults(node, role, "cycle", ("undecided",), (), (), "the dependency cycles")
        ctx.cache[node] = found
        return found
    ctx.stack.add(node)
    parents = _parents(node)
    readable = bool(parents)
    wrong = [parent for parent in parents if not _same(ctx.stated(parent), ctx.gold[parent])]
    active = tuple(wrong)
    if not _has_calculator(node):
        modes = _local_modes(node, False)
        found = AnswerFaults(
            node, role, "no-calculator", modes,
            ((Factor(node, modes),),) if modes else (),
            (),
            "This blank has no conditional output and is recorded by question layer as " + "+".join(modes),
        )
        ctx.stack.remove(node)
        ctx.cache[node] = found
        return found
    ok, phi = ctx.calculate(node, active)
    written = ctx.written(node)
    if not ok:
        gold_ok, _gold_reason = ctx.calculate(node, ())
        reason = f"the calculator rejected under a wrong prefix: {phi}" if phi else "the calculator rejected under a wrong prefix"
        if not gold_ok or not wrong:
            found = AnswerFaults(node, role, "rejected", ("undecided",), ((Factor(node, ("undecided",)),),), (), reason)
        else:
            def rejects(combo: tuple[str, ...]) -> bool:
                good, _reason = ctx.calculate(node, tuple(sorted(combo, key=int)))
                return not good

            direct_sets, unresolved = _minimal(wrong, rejects)
            note = reason.rstrip(".") + ". This question has no conditional output to compare, so it is not scored as execution or algebra."
            if unresolved:
                note += " Upstream still unresolved: " + ", ".join(f"ID{item}" for item in unresolved)
            found = AnswerFaults(
                node, role, "rejected", (), _unique(_expand(ctx, list(direct_sets), True)), tuple(direct_sets), note,
            )
        ctx.stack.remove(node)
        ctx.cache[node] = found
        return found
    matched = bool(written) and _same(written, phi)
    modes = _local_modes(node, matched)
    direct: list[tuple[str, ...]] = []
    unresolved: list[str] = []
    if readable and not _same(phi, ctx.gold[node]) and wrong:
        def holds(combo: tuple[str, ...]) -> bool:
            good, value = ctx.calculate(node, tuple(sorted(combo, key=int)))
            return good and _same(value, phi)

        direct_sets, unresolved = _minimal(wrong, holds)
        direct = list(direct_sets)
    expanded = _expand(ctx, direct, False)
    if modes:
        own = Factor(node, modes)
        expanded = [(own, *group) for group in expanded] or [(own,)]
    note = ""
    if unresolved:
        note = "Upstream still unresolved: " + ", ".join(f"ID{item}" for item in unresolved)
    if matched and not _same(phi, ctx.gold[node]):
        note = "The answer equals the conditional output under the wrong prefix. " + note
    elif not matched and wrong and _same(phi, ctx.gold[node]):
        note = "The conditional output under the wrong prefix is still the gold, so those upstream blanks did not cause this failure. " + note
    elif not matched and wrong:
        note = "The answer also does not equal the conditional output under the wrong prefix. " + note
    found = AnswerFaults(node, role, "same" if matched else "different", modes, _unique(expanded), tuple(direct), note.strip())
    ctx.stack.remove(node)
    ctx.cache[node] = found
    return found


def _load(sample: str, condition: str) -> dict[str, dict]:
    found = {}
    path = OBS / f"{sample}_{condition}" / "observations.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            found[_bare(row["node_id"])] = row
    return found


def attribute_run(model, gold: dict[str, str], sample: str, condition: str) -> list[AnswerFaults]:
    observed = _load(sample, condition)
    prefix = "gold" if condition == "gold-trunc" else "model"
    ctx = _Context(model, gold, observed, prefix)
    wrong = [
        node for node, row in observed.items()
        if row.get("asked", True) and not row.get("correct")
    ]
    wrong.sort(key=int)
    return [explain(ctx, node) for node in wrong]


def _sets(item: AnswerFaults) -> str:
    if not item.cause_sets:
        return "—"
    return " or ".join(", ".join(factor.text() for factor in group) for group in item.cause_sets)


def render(sample: str, condition: str, rows: list[AnswerFaults]) -> str:
    lines = [
        f"# Overlap attribution — {sample} · {condition}",
        "",
        "A local fault uses only the question layer: direct recognition DIR, reasoning recognition RSN, execution EXE, algebra ALG.",
        "This layer is recorded only when the answer disagrees with the conditional output under the wrong prefix. If they agree, the factor is upstream. Shape does not change the layer.",
        "",
    ]
    counts = Counter()
    for item in rows:
        local = "+".join(item.local) or "none"
        counts[local] += 1
    lines.append("Factors on this question: " + ", ".join(f"{name} {count}" for name, count in counts.most_common()))
    mixed = sum(1 for item in rows if len(item.modes()) > 1)
    several = sum(1 for item in rows if any(len(group) > 1 for group in item.cause_sets))
    lines.append(f"Questions whose factor set has more than one fault: {mixed}. Questions whose factor set has more than one factor: {several}.")
    lines.append("")
    lines.append("| Id | Question | Versus conditional output | This question | Actual factors | Direct upstream |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in rows:
        direct = " or ".join(", ".join(f"ID{node}" for node in group) for group in item.direct) or "—"
        lines.append(
            f"| ID{item.node} | {item.role} | {item.phi_status} | "
            f"{'+'.join(item.local) or 'none'} | {_sets(item)} | {direct} |"
        )
        if item.note:
            lines.append(f"|  |  |  |  | {item.note} |  |")
    return "\n".join(lines)


def _check(rows: list[AnswerFaults]) -> None:
    by_id = {item.node: item for item in rows}
    matrix = by_id["132"]
    if EXE not in matrix.local or matrix.direct:
        raise RuntimeError(f"ID132 should be EXE with no upstream: {matrix}")
    total = by_id["55"]
    if EXE in total.local or not total.direct:
        raise RuntimeError(f"The answer of ID55 equals the conditional output, so it should not have a local EXE: {total}")
    side = by_id["81"]
    if side.local != (ALG,) or not side.direct:
        raise RuntimeError(f"ID81 is in the algebra layer and should have a local ALG plus upstream: {side}")
    moment = by_id["197"]
    if moment.local != (ALG,) or not any(factor.modes == (EXE,) for group in moment.cause_sets for factor in group):
        raise RuntimeError(f"ID197 should be algebra ALG, with the sign factor left in execution: {moment}")
    if by_id["4"].local != (DIR,):
        raise RuntimeError(f"ID4 is an axial blank with no calculator and should be recorded as DIR: {by_id['4']}")
    if EXE not in by_id["98"].local:
        raise RuntimeError(f"ID98 disagrees with the fixed product code and should be recorded as EXE: {by_id['98']}")
    frame = by_id["24"]
    if EXE in frame.local or not {DIR, RSN} <= set(frame.modes()):
        raise RuntimeError(f"The calculator of ID24 was rejected by an upstream blank, so the factors should include both DIR and RSN: {frame}")
    for item in rows:
        if item.phi_status == "same" and item.local:
            raise RuntimeError(f"ID{item.node} matches the conditional output but still recorded a local fault {item.local}.")
        if item.phi_status == "different" and item.local != (scope_of(item.node),):
            raise RuntimeError(f"The local fault of ID{item.node} should be {scope_of(item.node)}, got {item.local}.")


if __name__ == "__main__":
    import json as _json

    from domains.wind_tunnel.contract_executor import _model, generated_answers

    data = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
    specs = {}
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = _json.loads(line)
        if row["id"] in {"body-rule", "roll-rule"}:
            specs[row["id"]] = row["spec"]
    for sample, spec in specs.items():
        model = _model(spec)
        gold = generated_answers(spec)
        for condition in ("self-trunc", "gold-trunc"):
            rows = attribute_run(model, gold, sample, condition)
            if sample == "body-rule" and condition == "self-trunc":
                _check(rows)
                print("self-check ok", flush=True)
            print(render(sample, condition, rows))
            print(flush=True)
