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
            if child.phi_status == "拒绝" and not through_rejection:
                parts.append(((Factor(parent, ("未判定",)),),))
            else:
                parts.append(child.cause_sets or ((Factor(parent, child.local or ("未判定",)),),))
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
        found = AnswerFaults(node, role, "循环", ("未判定",), (), (), "依赖成环")
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
            node, role, "无计算器", modes,
            ((Factor(node, modes),),) if modes else (),
            (),
            "这个空没有条件输出，按题目层级记 " + "+".join(modes),
        )
        ctx.stack.remove(node)
        ctx.cache[node] = found
        return found
    ok, phi = ctx.calculate(node, active)
    written = ctx.written(node)
    if not ok:
        gold_ok, _gold_reason = ctx.calculate(node, ())
        reason = f"错误前置下计算器拒绝：{phi}" if phi else "错误前置下计算器拒绝"
        if not gold_ok or not wrong:
            found = AnswerFaults(node, role, "拒绝", ("未判定",), ((Factor(node, ("未判定",)),),), (), reason)
        else:
            def rejects(combo: tuple[str, ...]) -> bool:
                good, _reason = ctx.calculate(node, tuple(sorted(combo, key=int)))
                return not good

            direct_sets, unresolved = _minimal(wrong, rejects)
            note = reason.rstrip("。") + "。本题没有可比较的条件输出，不在执行或数理记故障。"
            if unresolved:
                note += "还有未拆开的前置：" + "、".join(f"ID{item}" for item in unresolved)
            found = AnswerFaults(
                node, role, "拒绝", (), _unique(_expand(ctx, list(direct_sets), True)), tuple(direct_sets), note,
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
        note = "还有未拆开的前置：" + "、".join(f"ID{item}" for item in unresolved)
    if matched and not _same(phi, ctx.gold[node]):
        note = "答案等于错误前置的条件输出。" + note
    elif not matched and wrong and _same(phi, ctx.gold[node]):
        note = "错误前置的条件输出仍是金标，这些前置没有造成这道题的失败。" + note
    elif not matched and wrong:
        note = "答案也不等于错误前置的条件输出。" + note
    found = AnswerFaults(node, role, "相同" if matched else "不同", modes, _unique(expanded), tuple(direct), note.strip())
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
    return " 或 ".join("、".join(factor.text() for factor in group) for group in item.cause_sets)


def render(sample: str, condition: str, rows: list[AnswerFaults]) -> str:
    lines = [
        f"# 重叠归因 — {sample} · {condition}",
        "",
        "本地故障只看题目层级：直接识别 DIR，推理识别 RSN，执行 EXE，数理 ALG。",
        "答案与错误前置的条件输出不一致，才记这一层。一致则因素在上游。形态不改变层级。",
        "",
    ]
    counts = Counter()
    for item in rows:
        local = "+".join(item.local) or "无"
        counts[local] += 1
    lines.append("本题因素：" + "，".join(f"{name} {count}" for name, count in counts.most_common()))
    mixed = sum(1 for item in rows if len(item.modes()) > 1)
    several = sum(1 for item in rows if any(len(group) > 1 for group in item.cause_sets))
    lines.append(f"同一因素集里有多种故障的题：{mixed}。同一因素集里有多个因素的题：{several}。")
    lines.append("")
    lines.append("| 题号 | 题目 | 与条件输出 | 本题 | 实际因素 | 直接前置 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in rows:
        direct = " 或 ".join("、".join(f"ID{node}" for node in group) for group in item.direct) or "—"
        lines.append(
            f"| ID{item.node} | {item.role} | {item.phi_status} | "
            f"{'+'.join(item.local) or '无'} | {_sets(item)} | {direct} |"
        )
        if item.note:
            lines.append(f"|  |  |  |  | {item.note} |  |")
    return "\n".join(lines)


def _check(rows: list[AnswerFaults]) -> None:
    by_id = {item.node: item for item in rows}
    matrix = by_id["132"]
    if EXE not in matrix.local or matrix.direct:
        raise RuntimeError(f"ID132 应是没有上游的 EXE：{matrix}")
    total = by_id["55"]
    if EXE in total.local or not total.direct:
        raise RuntimeError(f"ID55 的答案等于条件输出，不应有本题 EXE：{total}")
    side = by_id["81"]
    if side.local != (ALG,) or not side.direct:
        raise RuntimeError(f"ID81 在数理层，应有本题 ALG 和上游：{side}")
    moment = by_id["197"]
    if moment.local != (ALG,) or not any(factor.modes == (EXE,) for group in moment.cause_sets for factor in group):
        raise RuntimeError(f"ID197 应是数理 ALG，符号因素留在执行：{moment}")
    if by_id["4"].local != (DIR,):
        raise RuntimeError(f"ID4 是没有计算器的轴向，应记 DIR：{by_id['4']}")
    if EXE not in by_id["98"].local:
        raise RuntimeError(f"ID98 与固定连乘码不一致，应记 EXE：{by_id['98']}")
    frame = by_id["24"]
    if EXE in frame.local or not {DIR, RSN} <= set(frame.modes()):
        raise RuntimeError(f"ID24 的计算器被前置拒绝，因素应同时有 DIR 和 RSN：{frame}")
    for item in rows:
        if item.phi_status == "相同" and item.local:
            raise RuntimeError(f"ID{item.node} 与条件输出相同，仍记了本题故障 {item.local}。")
        if item.phi_status == "不同" and item.local != (scope_of(item.node),):
            raise RuntimeError(f"ID{item.node} 的本题故障应是 {scope_of(item.node)}，实际 {item.local}。")


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
