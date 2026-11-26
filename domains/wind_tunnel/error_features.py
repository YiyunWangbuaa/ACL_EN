# -*- coding: utf-8 -*-
"""Extract error features from an answer. A feature only describes the shape.

The fault class comes from the question layer, not from the shape: direct recognition DIR, reasoning recognition RSN, execution EXE, algebra ALG.
Execution and algebra first consult the calculator under the model prefix:
a conditional output that matches the answer is propagated;
a conditional output that differs from the answer is EXE or ALG by layer;
a calculator rejection with no conditional output, when the rejection comes from an upstream blank, is also propagated and is not compared with the gold answer.
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
    DIR: "DIR direct recognition",
    RSN: "RSN reasoning recognition",
    EXE: "EXE execution",
    ALG: "ALG algebra",
    "propagated": "propagated",
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
    return tuple(str(item) for item in (options or []) if str(item) not in {"", "none"})


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
    """Only the signs differ inside a sum. One term is drop or flip, several terms are partial. Otherwise empty."""
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
    """Pull out the sign, trigonometry, angles, force coefficient, and matrix shape relative to the gold."""
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
    return " or ".join(", ".join(f"ID{node}" for node in group) for group in groups)


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
        fault_code = "propagated"
        sources = ", ".join(f"ID{item}" for item in slot.propagated) or "upstream"
        detail = f"{detail}; same as Phi, along {sources}"
    elif slot.refused and layer in {EXE, ALG} and slot.reject_sets:
        fault_code = "propagated"
        detail = f"{detail}; the calculator rejected and there is no conditional output, along {_along(slot.reject_sets)}"
    elif slot.refused and layer in {EXE, ALG}:
        fault_code = "unresolved"
        detail = f"{detail}; the calculator rejected and there is no upstream to follow, so {layer} is not scored against the gold"
    else:
        fault_code = layer
    return PatternVerdict(slot.node, surface, fault_code, 1.0, features, detail)


def _structural(slot: Slot) -> PatternVerdict | None:
    feat = extract_features(slot.model, slot.gold, slot.kind, slot.options)
    if not feat.parsed:
        return None
    if feat.transposed:
        return _verdict(slot, "transpose", EXE, ("transpose",), "the matrix is the transpose of the gold")
    if feat.outer_minus_dropped:
        return _verdict(slot, "missing outer sign", EXE, ("outer minus not written", "opposite sign"), "the outer minus of the gold was not written")
    if feat.negated:
        return _verdict(slot, "whole sign flip", EXE, ("opposite sign",), "the whole expression differs from the gold by a minus sign")
    if feat.cells_other == 0 and feat.cells_flipped == 1:
        return _verdict(slot, "local sign flip", ALG, ("sign flipped",), "1 sign is opposite the gold")
    if feat.cells_other == 0 and feat.cells_flipped > 1:
        return _verdict(
            slot, "local sign flip", EXE, ("sign flipped",),
            f"{feat.cells_flipped} signs are opposite the gold",
        )
    if feat.force_gold:
        features = [f"force coefficient {feat.force_gold}->{feat.force_model}"]
        detail = f"{feat.force_gold} was written as {feat.force_model}"
        if feat.force_sign_dropped:
            features.append("outer minus not written")
            detail += "; the outer minus was not written"
        return _verdict(slot, "force coefficient replaced", EXE, tuple(features), detail)
    if feat.double_angle:
        return _verdict(slot, "double-angle expansion", EXE, ("double-angle expansion",), "written as a double-angle expansion; the gold is first order")
    if feat.gold_identity and feat.model_has_angle and feat.planar:
        return _verdict(slot, "written with an angle", EXE, ("written with an angle",), "the identity was written as a rotation that still contains an angle")
    if feat.zero:
        return _verdict(slot, "written as zero", ALG, ("written as zero",), "the whole expression was written as 0")
    if feat.term_sign == "drop":
        return _verdict(slot, "one term missing its minus", ALG, ("one term missing its minus",), "one term is missing its minus sign")
    if feat.term_sign == "flip":
        return _verdict(slot, "one term sign flipped", ALG, ("sign flipped",), "one term has its sign flipped")
    if feat.term_sign == "partial":
        return _verdict(slot, "local sign flip", EXE, ("sign flipped",), "more than one term has a sign opposite the gold")
    if feat.zero_angle:
        names = ", ".join(feat.angles_dropped)
        return _verdict(
            slot, "angle dropped", EXE, (f"angle dropped:{names}",),
            f"after dropping {names}, it matches the gold with those angles set to 0",
        )
    if feat.planar and feat.angles_dropped:
        names = ", ".join(feat.angles_dropped)
        return _verdict(
            slot, "single-angle rotation block", EXE, ("single-angle rotation block",),
            f"only one rotation angle remains; substituting 0 for {names} does not match",
        )
    if feat.trig_swap:
        return _verdict(
            slot, "trigonometric function replaced", EXE, (f"trigonometric function {feat.trig_swap}",),
            f"trigonometric function {feat.trig_swap}",
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
                    left, "pairwise swap", _local_choice_fault(one), ("pairwise swap",),
                    f"swapped with ID{other} and written as {model[one]}",
                )
                found[other] = _verdict(
                    right, "pairwise swap", _local_choice_fault(other), ("pairwise swap",),
                    f"swapped with ID{one} and written as {model[other]}",
                )
        counts = Counter(model.values())
        duplicated = [name for name, count in counts.items() if count > 1]
        if duplicated and set(gold.values()) == set(options):
            missing = [name for name in dict.fromkeys(gold.values()) if counts[name] == 0]
            detail = ", ".join(duplicated) + " written more than once"
            if missing:
                detail += "; missing " + ", ".join(missing)
            for slot in wrongs:
                if slot.node in found:
                    continue
                found[slot.node] = _verdict(
                    slot, "code repeated", _local_choice_fault(slot.node), ("code repeated",), detail,
                )
        option_set = set(options)
        for slot in wrongs:
            if slot.node in found:
                continue
            picked = model[slot.node]
            if picked in option_set and picked != gold[slot.node]:
                found[slot.node] = _verdict(
                    slot, "option replaced", _local_choice_fault(slot.node), ("option replaced",),
                    f"written as {picked}",
                )
    return found


def _sign_hit(slot: Slot) -> PatternVerdict | None:
    if slot.kind != "sign" or slot.correct:
        return None
    pair = {slot.model.strip(), slot.gold.strip()}
    if pair == {"positive", "negative"}:
        return _verdict(slot, "sign flipped", EXE, ("sign flipped",), "positive and negative were swapped")
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
        raise RuntimeError(f"No manuscript for {sample_id}, so the conditional output cannot be recomputed on the rejection branch.")
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
        if item.phi_status == "rejected" and item.direct:
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
    """Constrain the wrong answers in one observation. Keys are question ids without the ID prefix."""
    return verdicts_for(_slots(sample_id, condition))


def render_report(sample_id: str, condition: str) -> str:
    slots = {slot.node: slot for slot in _slots(sample_id, condition)}
    verdicts = verdicts_for(list(slots.values()))
    wrongs = [node for node, slot in slots.items() if not slot.correct]
    wrongs.sort(key=lambda node: int(node))
    lines = [
        f"# Pattern constraint — {sample_id} · {condition}",
        "",
        "Only answers that match a constraint exactly are kept, and every match score is 1. A mismatch stays unresolved.",
        "Execution and algebra first consult the calculator under the model prefix.",
        "A conditional output that matches the answer is propagated. A conditional output that differs is EXE or ALG by layer.",
        "A calculator rejection that comes from an upstream blank is also propagated and is not compared with the gold answer.",
        "",
        "| Id | Question | Constraint | Fault | Features | Note |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    counts: Counter[str] = Counter()
    for node in wrongs:
        hit = verdicts.get(node)
        title = NODE_BY_ID[node].role
        if hit is None:
            counts["unresolved"] += 1
            lines.append(f"| ID{node} | {title} | — | — | — | — |")
            continue
        counts[hit.fault_name] += 1
        features = ", ".join(hit.features)
        lines.append(
            f"| ID{node} | {title} | {hit.surface} | {hit.fault_name} | {features} | {hit.detail} |"
        )
    lines.extend(["", "Distribution:"])
    for name in ("DIR direct recognition", "RSN reasoning recognition", "EXE execution", "ALG algebra", "propagated", "unresolved"):
        lines.append(f"- {name}: {counts.get(name, 0)}")
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
    format_code: str = "expression",
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
    _must(sign is not None and sign.surface == "missing outer sign" and sign.fault == EXE, f"outer minus: {sign}")
    flipped = _structural(_slot(
        "1", "matrix",
        "[[0, -1, 0], [1, 0, 0], [0, 0, 1]]",
        "[[0, 1, 0], [-1, 0, 0], [0, 0, -1]]",
        format_code="matrix",
    ))
    _must(flipped is not None and flipped.surface == "whole sign flip", f"whole sign flip: {flipped}")
    trig = _structural(_slot("1", "expression", "sin(alpha_M)", "cos(alpha_M)"))
    _must(trig is not None and trig.surface == "trigonometric function replaced" and "sin→cos" in trig.detail, f"trigonometric function: {trig}")
    force = _structural(_slot("197", "expression", "-CA*Yg/L_RT", "CZ*Yg/L_RT", format_code="moment-arm"))
    _must(
        force is not None and force.surface == "force coefficient replaced" and "outer minus not written" in force.features,
        f"force coefficient: {force}",
    )
    dropped = _structural(_slot("59", "expression", "cos(alpha_M)*cos(beta_M)", "cos(alpha_M)"))
    _must(dropped is not None and dropped.surface == "angle dropped", f"angle dropped: {dropped}")
    one_term = _structural(_slot(
        "111", "expression",
        "CN_TP*cos(gamma_M - gamma_tp) + CZ_TP*sin(gamma_M - gamma_tp)",
        "cos(gamma_M-gamma_tp)*CN_TP - sin(gamma_M-gamma_tp)*CZ_TP",
    ))
    _must(one_term is not None and one_term.surface == "one term sign flipped" and one_term.fault == ALG, f"one term sign: {one_term}")
    missing_term = _structural(_slot(
        "113", "expression",
        "CN_TP*sin(gamma_M - gamma_tp) - CZ_TP*cos(gamma_M - gamma_tp)",
        "sin(gamma_M-gamma_tp)*CN_TP + cos(gamma_M-gamma_tp)*CZ_TP",
    ))
    _must(
        missing_term is not None and missing_term.surface == "one term missing its minus" and missing_term.fault == ALG,
        f"one missing minus: {missing_term}",
    )
    zero = _structural(_slot("61", "expression", "-sin(alpha_M)", "0"))
    _must(zero is not None and zero.surface == "written as zero" and zero.fault == ALG, f"written as zero: {zero}")
    carried = _structural(_slot(
        "59", "expression", "cos(alpha_M)*cos(beta_M)", "cos(alpha_M)",
        phi="cos(alpha_M)", phi_ok=True, propagated=("55",),
    ))
    _must(carried is not None and carried.fault == "propagated" and carried.surface == "angle dropped", f"propagated: {carried}")
    identity = _structural(_slot(
        "126", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, cos(gamma_tp), -sin(gamma_tp)], [0, sin(gamma_tp), cos(gamma_tp)]]",
        format_code="matrix",
    ))
    _must(identity is not None and identity.surface == "written with an angle", f"written with an angle: {identity}")
    transposed = _structural(_slot(
        "132", "matrix",
        "[[1, 0, 0], [0, cos(gamma_tp), sin(gamma_tp)], [0, -sin(gamma_tp), cos(gamma_tp)]]",
        "[[1, 0, 0], [0, cos(gamma_tp), -sin(gamma_tp)], [0, sin(gamma_tp), cos(gamma_tp)]]",
        format_code="matrix",
    ))
    _must(transposed is not None and transposed.surface == "transpose", f"transpose: {transposed}")
    applied = _structural(_slot(
        "13", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]",
        format_code="matrix",
    ))
    _must(applied is not None and applied.surface == "local sign flip" and applied.fault == EXE, f"execution cell: {applied}")
    calculated = _structural(_slot(
        "55", "matrix",
        "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]",
        "[[1, 0, 0], [0, 1, 0], [0, 0, -1]]",
        format_code="matrix",
    ))
    _must(
        calculated is not None and calculated.surface == "local sign flip" and calculated.fault == ALG,
        f"algebra cell: {calculated}",
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
    block = _structural(_slot("55", "matrix", total, collapsed, format_code="matrix"))
    _must(block is not None and block.surface == "single-angle rotation block", f"single-angle rotation block: {block}")
    frames = ("body-axis frame", "balance frame", "wind frame")
    pair = verdicts_for([
        _slot("9", "choice", "body-axis frame", "wind frame", options=frames, format_code="choice"),
        _slot("10", "choice", "wind frame", "body-axis frame", options=frames, format_code="choice"),
        _slot("14", "choice", "body-axis frame", "body-axis frame", options=frames, format_code="choice", correct=True),
    ])
    _must(pair["9"].surface == "pairwise swap" and pair["9"].fault == DIR, f"swap: {pair['9']}")
    _must(pair["10"].fault == DIR and "ID9" in pair["10"].detail, f"swap partner: {pair['10']}")
    _must("14" not in pair, "A correct blank should not enter the constraint")
    load = ("R_(γ_M-γ_tp)", "RTP_TZ1", "RTP_TZ2")
    repeated = verdicts_for([
        _slot("98", "choice", "RTP_TZ2", "RTP_TZ1", options=load, format_code="choice", process="balance_to_body"),
        _slot("99", "choice", "RTP_TZ1", "R_(γ_M-γ_tp)", options=load, format_code="choice", process="balance_to_body"),
        _slot("100", "choice", "R_(γ_M-γ_tp)", "RTP_TZ1", options=load, format_code="choice", process="balance_to_body"),
    ])
    _must(repeated["98"].surface == "code repeated" and repeated["98"].fault == EXE, f"code repeated: {repeated['98']}")
    _must("RTP_TZ2" in repeated["98"].detail, f"missing code: {repeated['98'].detail}")
    _must(repeated["99"].surface == "pairwise swap" and repeated["100"].surface == "pairwise swap", f"load swap: {repeated}")
    opened = "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"
    closed = "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]"
    readable = _structural(_slot("104", "matrix", opened, closed, format_code="matrix"))
    _must(readable is not None and readable.fault == ALG and readable.surface == "local sign flip", f"ID104 stays algebra when it can be calculated: {readable}")
    blocked = _structural(_slot(
        "104", "matrix", opened, closed, format_code="matrix",
        refused=True, reject_sets=(("98",), ("99",), ("100",)),
    ))
    _must(
        blocked is not None and blocked.fault == "propagated" and blocked.surface == "local sign flip"
        and "rejected" in blocked.detail and "ID98" in blocked.detail
        and "ID99" in blocked.detail and "ID100" in blocked.detail,
        f"when the calculator rejected, ID104 should propagate along the product codes: {blocked}",
    )
    nameless = _structural(_slot("104", "matrix", opened, closed, format_code="matrix", refused=True))
    _must(nameless is not None and nameless.fault == "unresolved", f"a rejection with no upstream should not fall back to ALG: {nameless}")


_EXPECT = {
    "9": ("pairwise swap", DIR),
    "10": ("pairwise swap", DIR),
    "23": ("option replaced", RSN),
    "47": ("pairwise swap", EXE),
    "48": ("pairwise swap", EXE),
    "50": ("pairwise swap", EXE),
    "51": ("pairwise swap", EXE),
    "55": ("single-angle rotation block", "propagated"),
    "59": ("angle dropped", "propagated"),
    "60": ("angle dropped", "propagated"),
    "61": ("written as zero", "propagated"),
    "80": ("angle dropped", "propagated"),
    "81": ("angle dropped", ALG),
    "98": ("code repeated", EXE),
    "99": ("pairwise swap", EXE),
    "100": ("pairwise swap", EXE),
    "104": ("local sign flip", "propagated"),
    "126": ("written with an angle", EXE),
    "132": ("transpose", EXE),
    "147": ("double-angle expansion", "propagated"),
    "148": ("double-angle expansion", "propagated"),
    "176": ("double-angle expansion", "propagated"),
    "177": ("double-angle expansion", "propagated"),
    "111": ("one term sign flipped", "propagated"),
    "113": ("one term missing its minus", "propagated"),
    "114": ("one term sign flipped", "propagated"),
    "116": ("one term missing its minus", "propagated"),
    "197": ("force coefficient replaced", ALG),
    "199": ("force coefficient replaced", ALG),
    "200": ("force coefficient replaced", ALG),
}


def _audit_live() -> None:
    found = classify_wrongs("body-rule", "self-trunc")
    mismatches = []
    for node, (surface, fault) in _EXPECT.items():
        hit = found.get(node)
        if hit is None or hit.surface != surface or hit.fault != fault:
            got = "none" if hit is None else f"{hit.surface}/{hit.fault} {hit.detail}"
            mismatches.append(f"ID{node} expected {surface}/{fault}, got {got}")
    blocked = found.get("104")
    if blocked is None or not {"ID98", "ID99", "ID100"} <= set(blocked.detail.replace(" or ", " ").split()):
        got = "none" if blocked is None else blocked.detail
        mismatches.append(f"ID104 should follow ID98 or ID99 or ID100, got {got}")
    frame = found.get("24")
    if frame is None or frame.fault != DIR:
        got = "none" if frame is None else f"{frame.surface}/{frame.fault}"
        mismatches.append(f"ID24 is a rejection on direct recognition and should stay DIR, got {got}")
    if mismatches:
        raise RuntimeError("; ".join(mismatches))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Assign faults from answer patterns")
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
