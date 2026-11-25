# -*- coding: utf-8 -*-
"""从答案里提取错误特征。特征只描述形态。

故障类由题目层级决定，不由形态决定：直接识别 DIR，推理识别 RSN，执行 EXE，数理 ALG。
执行和数理先看模型前置下的计算器：
有条件输出且与答案相同，记为传播；
有条件输出且与答案不同，按层级记 EXE 或 ALG；
计算器拒绝、没有条件输出，并且拒绝来自上游空，也记为传播，不和标准答案比较。
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RSN

import itertools
import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import sympy as sp

from cloze_score import identified_option, judge, normalize
from domains.wind_tunnel.answer_form import parse_answer
from domains.wind_tunnel.error_graph import NODE_BY_ID
from domains.wind_tunnel.eval_scope import scope_of

_ROOT = Path(__file__).resolve().parents[2]
_OBS = _ROOT / "results" / "contract_consistency" / "error_workflow"
_CONTEXTS: dict[tuple[str, str], object] = {}
_FORCE = re.compile(r"(?<![A-Za-z])(CA|CZ|CN)(?![A-Za-z])")
_ANGLE = ("alpha", "beta", "gamma", "α", "β", "γ")
_TRIG = (
    (sp.sin, sp.cos, "sin", "cos"),
    (sp.cos, sp.sin, "cos", "sin"),
    (sp.tan, sp.sin, "tan", "sin"),
    (sp.sin, sp.tan, "sin", "tan"),
    (sp.asin, sp.atan, "asin", "atan"),
    (sp.atan, sp.asin, "atan", "asin"),
    (sp.sin, sp.asin, "sin", "asin"),
    (sp.asin, sp.sin, "asin", "sin"),
    (sp.cos, sp.acos, "cos", "acos"),
    (sp.acos, sp.cos, "acos", "cos"),
    (sp.tan, sp.atan, "tan", "atan"),
    (sp.atan, sp.tan, "atan", "tan"),
)
_FAULT = {
    DIR: "DIR 直接识别",
    RSN: "RSN 推理识别",
    EXE: "EXE 执行",
    ALG: "ALG 数理",
    "传播": "传播",
}


@dataclass(frozen=True)
class AnswerFeatures:
    parsed: bool
    kind: str
    transposed: bool = False
    negated: bool = False
    outer_minus_dropped: bool = False
    cells_flipped: int = 0
    cells_other: int = 0
    trig_swap: str = ""
    force_gold: str = ""
    force_model: str = ""
    force_sign_dropped: bool = False
    double_angle: bool = False
    gold_identity: bool = False
    model_has_angle: bool = False
    planar: bool = False
    zero: bool = False
    zero_angle: bool = False
    angles_dropped: tuple[str, ...] = ()
    term_sign: str = ""


@dataclass(frozen=True)
class PatternVerdict:
    node: str
    surface: str
    fault: str
    score: float
    features: tuple[str, ...]
    detail: str

    @property
    def fault_name(self) -> str:
        return _FAULT.get(self.fault, self.fault)


@dataclass(frozen=True)
class Slot:
    node: str
    process: str
    kind: str
    gold: str
    model: str
    options: tuple[str, ...]
    correct: bool
    format_code: str
    phi: str
    phi_ok: bool
    propagated: tuple[str, ...]
    origin: str
    refused: bool = False
    reject_sets: tuple[tuple[str, ...], ...] = ()


def _bare(value) -> str:
    return str(value).removeprefix("ID")


def _options(options) -> tuple[str, ...]:
    return tuple(str(item) for item in (options or []) if str(item) not in {"", "无"})


def _same(left, right) -> bool:
    try:
        diff = left - right
    except Exception:
        return False
    try:
        diff = sp.simplify(diff)
    except Exception:
        return False
    shape = getattr(diff, "shape", None)
    if shape:
        return all(_atom_zero(cell) for cell in diff)
    return _atom_zero(diff)


def _atom_zero(expr) -> bool:
    try:
        return bool(sp.simplify(expr) == 0)
    except Exception:
        return False


def _angle_symbols(expr) -> tuple:
    found = [
        symbol for symbol in getattr(expr, "free_symbols", ())
        if any(key in symbol.name for key in _ANGLE)
    ]
    return tuple(sorted(found, key=lambda symbol: symbol.name))


def _contains(expr, func) -> bool:
    return any(getattr(node, "func", None) is func for node in sp.preorder_traversal(expr))


def _double_angle(expr) -> bool:
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp == 2:
            if getattr(node.base, "func", None) in (sp.sin, sp.cos, sp.tan):
                return True
        func = getattr(node, "func", None)
        if func in (sp.sin, sp.cos, sp.tan) and node.args:
            coeff, _rest = node.args[0].as_coeff_mul()
            if getattr(coeff, "is_number", False) and abs(coeff) >= 2:
                return True
        if func is sp.Mul:
            funcs = {getattr(arg, "func", None) for arg in node.args}
            if sp.sin in funcs and sp.cos in funcs:
                return True
    return False


def _planar(matrix) -> bool:
    if getattr(matrix, "shape", None) != (3, 3):
        return False
    for axis in range(3):
        others = [index for index in range(3) if index != axis]
        if not all(_atom_zero(matrix[axis, index]) for index in others):
            continue
        if not all(_atom_zero(matrix[index, axis]) for index in others):
            continue
        diag = matrix[axis, axis]
        if not (_atom_zero(diag - 1) or _atom_zero(diag + 1)):
            continue
        first, second = others
        cosine, sine = matrix[first, first], matrix[first, second]
        sine_other, cosine_other = matrix[second, first], matrix[second, second]
        if _atom_zero(sine) and _atom_zero(cosine - 1) and _atom_zero(cosine_other - 1):
            continue
        if _atom_zero(sine) and _atom_zero(cosine):
            continue
        diag_ok = _atom_zero(cosine - cosine_other) or _atom_zero(cosine + cosine_other)
        skew_ok = _atom_zero(sine + sine_other) or _atom_zero(sine - sine_other)
        circle = _atom_zero(sp.simplify(cosine ** 2 + sine ** 2 - 1))
        if diag_ok and skew_ok and circle:
            return True
    return False


def _sign_of(expr) -> int:
    if getattr(expr, "func", None) is sp.Add:
        return 0
    try:
        coeff, _factors = expr.as_coeff_mul()
    except Exception:
        return 0
    if getattr(coeff, "is_number", False) and coeff < 0:
        return -1
    if getattr(coeff, "is_number", False) and coeff > 0:
        return 1
    return 0


def _force_body(text: str):
    raw = normalize(text)
    found = _FORCE.findall(raw)
    if len(set(found)) != 1:
        return None
    parsed = parse_answer(_FORCE.sub("F", raw), "expression")
    if parsed.status != "ok":
        return None
    expr = parsed.value
    try:
        coeff, factors = expr.as_coeff_mul()
    except Exception:
        return None
    sign = -1 if getattr(coeff, "is_number", False) and coeff < 0 else 1
    body = sp.Mul(*factors) if factors else expr
    return found[0], body, sign


def _cell_signs(pred, gold) -> tuple[int, int]:
    if pred.kind not in {"matrix", "vector"}:
        return 0, 0
    flipped = other = 0
    for (_index, left), (_other, right) in zip(pred.cells, gold.cells):
        if _same(left, right):
            continue
        if _same(left, -right):
            flipped += 1
        else:
            other += 1
    return flipped, other


def _zero_angle(gold, model) -> tuple[bool, tuple[str, ...]]:
    gold_angles = _angle_symbols(gold)
    model_names = {symbol.name for symbol in _angle_symbols(model)}
    dropped = tuple(symbol for symbol in gold_angles if symbol.name not in model_names)
    if not dropped:
        return False, ()
    try:
        reduced = sp.simplify(gold.subs({symbol: 0 for symbol in dropped}))
    except Exception:
        return False, tuple(symbol.name for symbol in dropped)
    return _same(reduced, model), tuple(symbol.name for symbol in dropped)


def _split_term(term) -> tuple[int, object]:
    coeff, factors = term.as_coeff_mul()
    sign = 1
    if getattr(coeff, "is_number", False) and coeff < 0:
        sign = -1
        coeff = -coeff
    body = sp.Mul(coeff, *factors) if factors else coeff
    return sign, body


def _term_sign(gold, model) -> str:
    """和式里仅符号不同。一项为 drop 或 flip，多项为 partial。对不上则空。"""
    if getattr(gold, "func", None) is not sp.Add or getattr(model, "func", None) is not sp.Add:
        return ""
    gold_terms = list(gold.args)
    model_terms = list(model.args)
    if len(gold_terms) != len(model_terms) or len(gold_terms) < 2:
        return ""
    used: set[int] = set()
    flipped = dropped = 0
    for term in gold_terms:
        sign, body = _split_term(term)
        match = None
        for index, other in enumerate(model_terms):
            if index in used:
                continue
            other_sign, other_body = _split_term(other)
            if _same(body, other_body):
                match = (index, other_sign)
                break
        if match is None:
            return ""
        used.add(match[0])
        if sign != match[1]:
            flipped += 1
            if sign < 0 < match[1]:
                dropped += 1
    if flipped == 0 or len(used) != len(model_terms):
        return ""
    if flipped == len(gold_terms):
        return ""
    if flipped == 1:
        return "drop" if dropped == 1 else "flip"
    return "partial"


def _trig_swap(gold, model) -> str:
    found = []
    for source, target, source_name, target_name in _TRIG:
        if not _contains(gold, source):
            continue
        try:
            changed = gold.replace(source, lambda *args, _target=target: _target(*args))
        except Exception:
            continue
        if _same(changed, model):
            found.append(f"{source_name}→{target_name}")
    if len(found) == 1:
        return found[0]
    return ""


def extract_features(model: str, gold: str, kind: str, options=None) -> AnswerFeatures:
    """抽出相对金标的符号、三角、角度、力系数和矩阵形状。"""
    if kind not in {"matrix", "vector", "expression"}:
        return AnswerFeatures(parsed=False, kind=kind)
    pred = parse_answer(model, kind, options)
    ref = parse_answer(gold, kind, options)
    if pred.status != "ok" or ref.status != "ok":
        return AnswerFeatures(parsed=False, kind=kind)
    left, right = pred.value, ref.value
    negated = _same(left, -right)
    dropped_hit, dropped = _zero_angle(right, left)
    force = _force_body(model)
    force_gold = _force_body(gold)
    force_ok = (
        force is not None
        and force_gold is not None
        and force[0] != force_gold[0]
        and _same(force[1], force_gold[1])
    )
    flipped, other = _cell_signs(pred, ref)
    transposed = kind == "matrix" and _same(left, right.T)
    return AnswerFeatures(
        parsed=True,
        kind=kind,
        transposed=transposed,
        negated=negated,
        outer_minus_dropped=kind == "expression" and negated and _sign_of(right) < 0 and _sign_of(left) > 0,
        cells_flipped=flipped,
        cells_other=other,
        trig_swap="" if dropped else _trig_swap(right, left),
        force_gold=force_gold[0] if force_ok else "",
        force_model=force[0] if force_ok else "",
        force_sign_dropped=bool(force_ok and force_gold[2] < 0 and force[2] > 0),
        double_angle=_double_angle(left) and not _double_angle(right),
        gold_identity=kind == "matrix" and _same(right, sp.eye(3)),
        model_has_angle=bool(_angle_symbols(left)),
        planar=kind == "matrix" and _planar(left),
        zero=_same(left, sp.zeros(*left.shape) if getattr(left, "shape", None) else 0),
        zero_angle=dropped_hit and not _same(left, 0 if kind == "expression" else sp.zeros(*left.shape)),
        angles_dropped=dropped,
        term_sign=_term_sign(right, left) if kind == "expression" else "",
    )


def _local_choice_fault(node: str) -> str:
    return scope_of(node)


def _phi_same(slot: Slot) -> bool:
    if not slot.phi_ok or not slot.phi:
        return False
    try:
        return bool(judge(slot.model, slot.phi, slot.format_code, list(slot.options))["normalized"])
    except Exception:
        return False


def _along(groups: tuple[tuple[str, ...], ...]) -> str:
    return " 或 ".join("、".join(f"ID{node}" for node in group) for group in groups)


def _verdict(
    slot: Slot,
    surface: str,
    fault: str,
    features: tuple[str, ...],
    detail: str,
) -> PatternVerdict:
    """``fault`` is unused. The calculator on the model prefix assigns EXE and ALG."""
    del fault
    layer = scope_of(slot.node)
    if _phi_same(slot):
        fault_code = "传播"
        sources = "、".join(f"ID{item}" for item in slot.propagated) or "上游"
        detail = f"{detail}；与 Φ 相同，沿 {sources}"
    elif slot.refused and layer in {EXE, ALG} and slot.reject_sets:
        fault_code = "传播"
        detail = f"{detail}；计算器拒绝，没有条件输出，沿 {_along(slot.reject_sets)}"
    elif slot.refused and layer in {EXE, ALG}:
        fault_code = "未认定"
        detail = f"{detail}；计算器拒绝，没有可沿的上游，不按标准答案记 {layer}"
    else:
        fault_code = layer
    return PatternVerdict(slot.node, surface, fault_code, 1.0, features, detail)


def _structural(slot: Slot) -> PatternVerdict | None:
    feat = extract_features(slot.model, slot.gold, slot.kind, slot.options)
    if not feat.parsed:
        return None
    if feat.transposed:
        return _verdict(slot, "转置", EXE, ("转置",), "矩阵是金标的转置")
    if feat.outer_minus_dropped:
        return _verdict(slot, "外侧符号缺失", EXE, ("外侧负号未写", "符号相反"), "金标的外侧负号没有写上")
    if feat.negated:
        return _verdict(slot, "整体反号", EXE, ("符号相反",), "整体与金标相差一个负号")
    if feat.cells_other == 0 and feat.cells_flipped == 1:
        return _verdict(slot, "局部符号写反", ALG, ("符号写反",), "1 处符号与金标相反")
    if feat.cells_other == 0 and feat.cells_flipped > 1:
        return _verdict(
            slot, "局部符号写反", EXE, ("符号写反",),
            f"{feat.cells_flipped} 处符号与金标相反",
        )
    if feat.force_gold:
        features = [f"力系数{feat.force_gold}→{feat.force_model}"]
        detail = f"{feat.force_gold} 写成了 {feat.force_model}"
        if feat.force_sign_dropped:
            features.append("外侧负号未写")
            detail += "，外侧负号没有写上"
        return _verdict(slot, "力系数替换", EXE, tuple(features), detail)
    if feat.double_angle:
        return _verdict(slot, "倍角展开", EXE, ("倍角展开",), "写成了倍角展开，金标是一次式")
    if feat.gold_identity and feat.model_has_angle and feat.planar:
        return _verdict(slot, "写成含角", EXE, ("写成含角",), "单位阵写成了含角的旋转")
    if feat.zero:
        return _verdict(slot, "写成零", ALG, ("写成零",), "整式写成 0")
    if feat.term_sign == "drop":
        return _verdict(slot, "一项负号未写", ALG, ("一项负号未写",), "有一项的负号没有写上")
    if feat.term_sign == "flip":
        return _verdict(slot, "一项符号写反", ALG, ("符号写反",), "有一项的符号写反了")
    if feat.term_sign == "partial":
        return _verdict(slot, "局部符号写反", EXE, ("符号写反",), "不止一项的符号与金标相反")
    if feat.zero_angle:
        names = "、".join(feat.angles_dropped)
        return _verdict(
            slot, "角度丢弃", EXE, (f"角度丢弃:{names}",),
            f"去掉 {names} 后，与这些角为 0 时的金标相同",
        )
    if feat.planar and feat.angles_dropped:
        names = "、".join(feat.angles_dropped)
        return _verdict(
            slot, "单角旋转块", EXE, ("单角旋转块",),
            f"只留下一个转角；把 {names} 代成 0 对不上",
        )
    if feat.trig_swap:
        return _verdict(
            slot, "三角函数替换", EXE, (f"三角函数{feat.trig_swap}",),
            f"三角函数 {feat.trig_swap}",
        )
    return None


def _picked(text: str, options: tuple[str, ...]) -> str:
    found = identified_option(text, list(options))
    if found is not None:
        return found
    return text.strip()


def _choice_hits(slots: list[Slot]) -> dict[str, PatternVerdict]:
    groups: dict[tuple, list[Slot]] = {}
    for slot in slots:
        if slot.kind != "choice" or not slot.options:
            continue
        groups.setdefault((slot.process, slot.options), []).append(slot)
    found: dict[str, PatternVerdict] = {}
    for group in groups.values():
        options = group[0].options
        gold = {slot.node: _picked(slot.gold, options) for slot in group}
        model = {slot.node: _picked(slot.model, options) for slot in group}
        wrongs = [slot for slot in group if not slot.correct]
        paired: set[str] = set()
        for left, right in itertools.combinations(wrongs, 2):
            one, other = left.node, right.node
            if one in paired or other in paired:
                continue
            if model[one] == gold[other] and model[other] == gold[one] and model[one] != gold[one]:
                paired.update((one, other))
                found[one] = _verdict(
                    left, "成对对调", _local_choice_fault(one), ("成对对调",),
                    f"与 ID{other} 互换，写成了{model[one]}",
                )
                found[other] = _verdict(
                    right, "成对对调", _local_choice_fault(other), ("成对对调",),
                    f"与 ID{one} 互换，写成了{model[other]}",
                )
        counts = Counter(model.values())
        duplicated = [name for name, count in counts.items() if count > 1]
        if duplicated and set(gold.values()) == set(options):
            missing = [name for name in dict.fromkeys(gold.values()) if counts[name] == 0]
            detail = "、".join(duplicated) + " 写了多次"
            if missing:
                detail += "，缺 " + "、".join(missing)
            for slot in wrongs:
                if slot.node in found:
                    continue
                found[slot.node] = _verdict(
                    slot, "代号重复", _local_choice_fault(slot.node), ("代号重复",), detail,
                )
        option_set = set(options)
        for slot in wrongs:
            if slot.node in found:
                continue
            picked = model[slot.node]
            if picked in option_set and picked != gold[slot.node]:
                found[slot.node] = _verdict(
                    slot, "选项替换", _local_choice_fault(slot.node), ("选项替换",),
                    f"写成了{picked}",
                )
    return found


def _sign_hit(slot: Slot) -> PatternVerdict | None:
    if slot.kind != "sign" or slot.correct:
        return None
    pair = {slot.model.strip(), slot.gold.strip()}
    if pair == {"正", "负"}:
        return _verdict(slot, "符号写反", EXE, ("符号写反",), "正负写反")
    return None


def verdicts_for(slots: list[Slot]) -> dict[str, PatternVerdict]:
    choice = _choice_hits(slots)
    found = {}
    for slot in slots:
        if slot.correct:
            continue
        if slot.kind == "choice":
            hit = choice.get(slot.node)
        elif slot.kind == "sign":
            hit = _sign_hit(slot)
        else:
            hit = _structural(slot)
        if hit is not None:
            found[slot.node] = hit
    return found


def _refusal_context(sample_id: str, condition: str):
    key = (sample_id, condition)
    cached = _CONTEXTS.get(key)
    if cached is not None:
        return cached
    from domains.wind_tunnel.contract_executor import _model, generated_answers
    from domains.wind_tunnel.overlap_faults import _Context, _load

    spec = None
    data = _ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("id") == sample_id:
            spec = row["spec"]
            break
    if spec is None:
        raise RuntimeError(f"没有 {sample_id} 的题面，无法在拒绝分支上重算条件输出。")
    prefix = "gold" if condition == "gold-trunc" else "model"
    ctx = _Context(_model(spec), generated_answers(spec), _load(sample_id, condition), prefix)
    _CONTEXTS[key] = ctx
    return ctx


def _attach_refusals(sample_id: str, condition: str, slots: list[Slot]) -> list[Slot]:
    """Name the upstream blanks that make an application or calculation calculator refuse."""
    pending = [slot.node for slot in slots if slot.refused and scope_of(slot.node) in {EXE, ALG}]
    if not pending:
        return slots
    from domains.wind_tunnel.overlap_faults import explain

    ctx = _refusal_context(sample_id, condition)
    found = {}
    for node in pending:
        item = explain(ctx, node)
        if item.phi_status == "拒绝" and item.direct:
            found[node] = item.direct
    if not found:
        return slots
    return [replace(slot, reject_sets=found[slot.node]) if slot.node in found else slot for slot in slots]


def _slots(sample_id: str, condition: str) -> list[Slot]:
    folder = _OBS / f"{sample_id}_{condition}"
    cases = {}
    for line in (folder / "error_cases.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("case_origin") != "wrong_answer":
            continue
        cases[_bare(row["node_id"])] = row
    slots = []
    for line in (folder / "observations.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("asked", True):
            continue
        node = _bare(row["node_id"])
        if node not in NODE_BY_ID:
            continue
        case = cases.get(node, {})
        slots.append(Slot(
            node=node,
            process=str(row.get("process") or ""),
            kind=NODE_BY_ID[node].answer_type,
            gold=str(row.get("gold_answer") or ""),
            model=str(row.get("model_answer") or ""),
            options=_options(row.get("options")),
            correct=bool(row.get("correct")),
            format_code=str(row.get("format_code") or ""),
            phi=str(case.get("phi_matrix") or ""),
            phi_ok=bool(case.get("phi_verified")),
            propagated=tuple(_bare(item) for item in (case.get("propagated_from") or [])),
            origin=str(case.get("final_class") or ""),
            refused=(
                str(case.get("propagation_status") or "") == "contract_contradiction"
                and not bool(case.get("phi_verified"))
            ),
        ))
    return _attach_refusals(sample_id, condition, slots)


def classify_wrongs(sample_id: str, condition: str) -> dict[str, PatternVerdict]:
    """对一份观测里的错题做模式约束。键是不带 ID 前缀的题号。"""
    return verdicts_for(_slots(sample_id, condition))


def render_report(sample_id: str, condition: str) -> str:
    slots = {slot.node: slot for slot in _slots(sample_id, condition)}
    verdicts = verdicts_for(list(slots.values()))
    wrongs = [node for node, slot in slots.items() if not slot.correct]
    wrongs.sort(key=lambda node: int(node))
    lines = [
        f"# 模式约束 — {sample_id} · {condition}",
        "",
        "只保留和约束完全对齐的答案，匹配度都是 1。对不上的不认定。",
        "执行和数理先看模型前置下的计算器。",
        "条件输出与答案相同，记为传播。条件输出与答案不同，按层级记 EXE 或 ALG。",
        "计算器拒绝且拒绝来自上游空，也记为传播，不和标准答案比较。",
        "",
        "| 题号 | 题目 | 约束 | 故障 | 特征 | 说明 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    counts: Counter[str] = Counter()
    for node in wrongs:
        hit = verdicts.get(node)
        title = NODE_BY_ID[node].role
        if hit is None:
            counts["未认定"] += 1
            lines.append(f"| ID{node} | {title} | — | — | — | — |")
            continue
        counts[hit.fault_name] += 1
        features = "、".join(hit.features)
        lines.append(
            f"| ID{node} | {title} | {hit.surface} | {hit.fault_name} | {features} | {hit.detail} |"
        )
    lines.extend(["", "分布："])
    for name in ("DIR 直接识别", "RSN 推理识别", "EXE 执行", "ALG 数理", "传播", "未认定"):
        lines.append(f"- {name}：{counts.get(name, 0)}")
    return "\n".join(lines)


def _must(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _slot(
    node: str,
    kind: str,
    gold: str,
    model: str,
    *,
    options: tuple[str, ...] = (),
    correct: bool = False,
    format_code: str = "表达式",
    phi: str = "",
    phi_ok: bool = False,
    propagated: tuple[str, ...] = (),
    process: str = "wind_to_body",
    refused: bool = False,
    reject_sets: tuple[tuple[str, ...], ...] = (),
) -> Slot:
    return Slot(
        node, process, kind, gold, model, options, correct, format_code,
        phi, phi_ok, propagated, "", refused, reject_sets,
    )


def _self_check() -> None:
    sign = _structural(_slot("85", "expression", "-sin(alpha_M)", "sin(alpha_M)"))
    _must(sign is not None and sign.surface == "外侧符号缺失" and sign.fault == EXE, f"外侧负号：{sign}")
    flipped = _structural(_slot(
        "1", "matrix",
        "[[0, -1, 0], [1, 0, 0], [0, 0, 1]]",
        "[[0, 1, 0], [-1, 0, 0], [0, 0, -1]]",
        format_code="矩阵",
    ))
    _must(flipped is not None and flipped.surface == "整体反号", f"整体反号：{flipped}")
    trig = _structural(_slot("1", "expression", "sin(alpha_M)", "cos(alpha_M)"))
    _must(trig is not None and trig.surface == "三角函数替换" and "sin→cos" in trig.detail, f"三角函数：{trig}")
    force = _structural(_slot("197", "expression", "-CA*Yg/L_RT", "CZ*Yg/L_RT", format_code="力臂"))
    _must(
        force is not None and force.surface == "力系数替换" and "外侧负号未写" in force.features,
        f"力系数：{force}",
    )
    dropped = _structural(_slot("59", "expression", "cos(alpha_M)*cos(beta_M)", "cos(alpha_M)"))
    _must(dropped is not None and dropped.surface == "角度丢弃", f"角度丢弃：{dropped}")
    one_term = _structural(_slot(
        "111", "expression",
        "CN_TP*cos(gamma_M - gamma_tp) + CZ_TP*sin(gamma_M - gamma_tp)",
        "cos(gamma_M-gamma_tp)*CN_TP - sin(gamma_M-gamma_tp)*CZ_TP",
    ))
    _must(one_term is not None and one_term.surface == "一项符号写反" and one_term.fault == ALG, f"一项符号：{one_term}")
    missing_term = _structural(_slot(
        "113", "expression",
        "CN_TP*sin(gamma_M - gamma_tp) - CZ_TP*cos(gamma_M - gamma_tp)",
        "sin(gamma_M-gamma_tp)*CN_TP + cos(gamma_M-gamma_tp)*CZ_TP",
    ))
    _must(
        missing_term is not None and missing_term.surface == "一项负号未写" and missing_term.fault == ALG,
        f"一项负号：{missing_term}",
    )
    zero = _structural(_slot("61", "expression", "-sin(alpha_M)", "0"))
    _must(zero is not None and zero.surface == "写成零" and zero.fault == ALG, f"写成零：{zero}")
    carried = _structural(_slot(
        "59", "expression", "cos(alpha_M)*cos(beta_M)", "cos(alpha_M)",
        phi="cos(alpha_M)", phi_ok=True, propagated=("55",),
    ))
    _must(carried is not None and carried.fault == "传播" and carried.surface == "角度丢弃", f"传播：{carried}")
    identity = _structural(_slot(
        "126", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, cos(gamma_tp), -sin(gamma_tp)], [0, sin(gamma_tp), cos(gamma_tp)]]",
        format_code="矩阵",
    ))
    _must(identity is not None and identity.surface == "写成含角", f"写成含角：{identity}")
    transposed = _structural(_slot(
        "132", "matrix",
        "[[1, 0, 0], [0, cos(gamma_tp), sin(gamma_tp)], [0, -sin(gamma_tp), cos(gamma_tp)]]",
        "[[1, 0, 0], [0, cos(gamma_tp), -sin(gamma_tp)], [0, sin(gamma_tp), cos(gamma_tp)]]",
        format_code="矩阵",
    ))
    _must(transposed is not None and transposed.surface == "转置", f"转置：{transposed}")
    applied = _structural(_slot(
        "13", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]",
        format_code="矩阵",
    ))
    _must(applied is not None and applied.surface == "局部符号写反" and applied.fault == EXE, f"执行单格：{applied}")
    calculated = _structural(_slot(
        "55", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]",
        format_code="矩阵",
    ))
    _must(
        calculated is not None and calculated.surface == "局部符号写反" and calculated.fault == ALG,
        f"数理单格：{calculated}",
    )
    total = (
        "[[cos(alpha_M)*cos(beta_M), sin(alpha_M)*cos(beta_M), sin(beta_M)], "
        "[-sin(alpha_M)*cos(gamma_M) + sin(beta_M)*sin(gamma_M)*cos(alpha_M), "
        "sin(alpha_M)*sin(beta_M)*sin(gamma_M) + cos(alpha_M)*cos(gamma_M), "
        "-sin(gamma_M)*cos(beta_M)], "
        "[-sin(alpha_M)*sin(gamma_M) - sin(beta_M)*cos(alpha_M)*cos(gamma_M), "
        "-sin(alpha_M)*sin(beta_M)*cos(gamma_M) + sin(gamma_M)*cos(alpha_M), "
        "cos(beta_M)*cos(gamma_M)]]"
    )
    collapsed = "[[cos(alpha_M), sin(alpha_M), 0], [-sin(alpha_M), cos(alpha_M), 0], [0, 0, -1]]"
    block = _structural(_slot("55", "matrix", total, collapsed, format_code="矩阵"))
    _must(block is not None and block.surface == "单角旋转块", f"单角旋转块：{block}")
    frames = ("体轴系", "天平系", "风轴系")
    pair = verdicts_for([
        _slot("9", "choice", "体轴系", "风轴系", options=frames, format_code="无"),
        _slot("10", "choice", "风轴系", "体轴系", options=frames, format_code="无"),
        _slot("14", "choice", "体轴系", "体轴系", options=frames, format_code="无", correct=True),
    ])
    _must(pair["9"].surface == "成对对调" and pair["9"].fault == DIR, f"对调：{pair['9']}")
    _must(pair["10"].fault == DIR and "ID9" in pair["10"].detail, f"对调对方：{pair['10']}")
    _must("14" not in pair, "答对的空不应进入约束")
    load = ("R_(γ_M-γ_tp)", "RTP_TZ1", "RTP_TZ2")
    repeated = verdicts_for([
        _slot("98", "choice", "RTP_TZ2", "RTP_TZ1", options=load, format_code="无", process="balance_to_body"),
        _slot("99", "choice", "RTP_TZ1", "R_(γ_M-γ_tp)", options=load, format_code="无", process="balance_to_body"),
        _slot("100", "choice", "R_(γ_M-γ_tp)", "RTP_TZ1", options=load, format_code="无", process="balance_to_body"),
    ])
    _must(repeated["98"].surface == "代号重复" and repeated["98"].fault == EXE, f"代号重复：{repeated['98']}")
    _must("RTP_TZ2" in repeated["98"].detail, f"缺号：{repeated['98'].detail}")
    _must(repeated["99"].surface == "成对对调" and repeated["100"].surface == "成对对调", f"载荷对调：{repeated}")
    opened = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    closed = "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]"
    readable = _structural(_slot("104", "matrix", opened, closed, format_code="矩阵"))
    _must(readable is not None and readable.fault == ALG and readable.surface == "局部符号写反", f"可计算时 ID104 仍按数理：{readable}")
    blocked = _structural(_slot(
        "104", "matrix", opened, closed, format_code="矩阵",
        refused=True, reject_sets=(("98",), ("99",), ("100",)),
    ))
    _must(
        blocked is not None and blocked.fault == "传播" and blocked.surface == "局部符号写反"
        and "拒绝" in blocked.detail and "ID98" in blocked.detail
        and "ID99" in blocked.detail and "ID100" in blocked.detail,
        f"计算器拒绝时 ID104 应沿连乘码传播：{blocked}",
    )
    nameless = _structural(_slot("104", "matrix", opened, closed, format_code="矩阵", refused=True))
    _must(nameless is not None and nameless.fault == "未认定", f"拒绝但没有上游时不应落回 ALG：{nameless}")


_EXPECT = {
    "9": ("成对对调", DIR),
    "10": ("成对对调", DIR),
    "23": ("选项替换", RSN),
    "47": ("成对对调", EXE),
    "48": ("成对对调", EXE),
    "50": ("成对对调", EXE),
    "51": ("成对对调", EXE),
    "55": ("单角旋转块", "传播"),
    "59": ("角度丢弃", "传播"),
    "60": ("角度丢弃", "传播"),
    "61": ("写成零", "传播"),
    "80": ("角度丢弃", "传播"),
    "81": ("角度丢弃", ALG),
    "98": ("代号重复", EXE),
    "99": ("成对对调", EXE),
    "100": ("成对对调", EXE),
    "104": ("局部符号写反", "传播"),
    "126": ("写成含角", EXE),
    "132": ("转置", EXE),
    "147": ("倍角展开", "传播"),
    "148": ("倍角展开", "传播"),
    "176": ("倍角展开", "传播"),
    "177": ("倍角展开", "传播"),
    "111": ("一项符号写反", "传播"),
    "113": ("一项负号未写", "传播"),
    "114": ("一项符号写反", "传播"),
    "116": ("一项负号未写", "传播"),
    "197": ("力系数替换", ALG),
    "199": ("力系数替换", ALG),
    "200": ("力系数替换", ALG),
}


def _audit_live() -> None:
    found = classify_wrongs("body-rule", "self-trunc")
    mismatches = []
    for node, (surface, fault) in _EXPECT.items():
        hit = found.get(node)
        if hit is None or hit.surface != surface or hit.fault != fault:
            got = "无" if hit is None else f"{hit.surface}/{hit.fault} {hit.detail}"
            mismatches.append(f"ID{node} 期望 {surface}/{fault}，实际 {got}")
    blocked = found.get("104")
    if blocked is None or not {"ID98", "ID99", "ID100"} <= set(blocked.detail.replace("或", " ").split()):
        got = "无" if blocked is None else blocked.detail
        mismatches.append(f"ID104 应沿 ID98 或 ID99 或 ID100，实际 {got}")
    frame = found.get("24")
    if frame is None or frame.fault != DIR:
        got = "无" if frame is None else f"{frame.surface}/{frame.fault}"
        mismatches.append(f"ID24 是直接识别上的拒绝，应仍为 DIR，实际 {got}")
    if mismatches:
        raise RuntimeError("；".join(mismatches))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按答案模式认定故障")
    parser.add_argument("--sample", default="body-rule")
    parser.add_argument("--condition", default="self-trunc")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    _self_check()
    if args.sample == "body-rule" and args.condition == "self-trunc":
        _audit_live()
    text = render_report(args.sample, args.condition)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
