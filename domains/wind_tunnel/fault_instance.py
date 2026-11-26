# -*- coding: utf-8 -*-
"""Injectable fault instances. RES stays outside the row set.

DIR is direct recognition: one row per non-latent contract that states a
mapping. The injector writes another option of that blank, never this
manuscript's gold. A blank with no such option is left out.

RSN is reasoning recognition: one row per route the contract implies, in the
process that holds those slots. EXE is execution: one row per parent blank,
named as the judgment that failed. The witness differs from this manuscript's
gold and moves at least one calculator. A child that only reads that parent
is propagation. ALG is one algebraic slip per matrix, vector, or formula
blank. RES has no injector.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RES, RSN

from dataclasses import dataclass, replace
from functools import lru_cache

import sympy as sp

from cloze_score import normalize
from domains.wind_tunnel.answer_form import match_predicate, parse_answer
from domains.wind_tunnel.contract_graph import _authored, _recognition_index, _statuses
from domains.wind_tunnel.error_graph import NODE_BY_ID, diagnosis_descendants
from domains.wind_tunnel.measurement import by_node
from domains.wind_tunnel.multi_slot import _BRANCH_MODALITY, _READS, branch_table
from domains.wind_tunnel.slot_signature import (
    _manuscript,
    _options,
    _outcome,
    calculator_nodes,
    slots,
)
from domains.wind_tunnel.tz_fd_generate import _matrix_text, _text, _vector_text

_ROW_CLASSES = frozenset({DIR, RSN, EXE, ALG})
_MATH_KINDS = frozenset({"matrix", "vector", "expression"})
# These expressions are only read to judge a component sign. The sign question
# does not cite a method or a calculation, so they do not get an ALG row.
SIGN_COMPONENTS = frozenset({"59", "60", "61", "146", "147", "148"})


@dataclass(frozen=True)
class FaultInstance:
    id: str
    klass: str
    scope: str
    injector: tuple[tuple[str, str], ...]
    direct: frozenset[str]
    potential: frozenset[str]


def outside() -> tuple[tuple[str, str], ...]:
    """Classes that cannot be rows. A prediction needs an injector."""
    return ((RES, "residual has no injector"),)


@lru_cache(maxsize=None)
def e1_instances(sample_id: str = "standard-default") -> tuple[FaultInstance, ...]:
    """Recognition rows. Each blank is written as another of its options."""
    return _e1(sample_id)


@lru_cache(maxsize=None)
def e1_census(sample_id: str = "standard-default") -> dict[str, int]:
    """How the recognition rows sit in the contract graph."""
    recognition = _recognition_index()
    statuses = _statuses()
    observed = 0
    derived = 0
    for item in e1_instances(sample_id):
        iid = item.id.removeprefix(f"{DIR}:")
        status = statuses.get(iid, "")
        if status == "observed":
            observed += 1
        elif status == "derived":
            derived += 1
        if iid not in recognition and status == "observed":
            raise RuntimeError(f"{item.id} is marked observed, but it has no recognition blank.")
    return {"rows": len(e1_instances(sample_id)), "observed": observed, "derived": derived}


@lru_cache(maxsize=None)
def instances(sample_id: str) -> tuple[FaultInstance, ...]:
    """Rows of one manuscript. EXE and ALG read that manuscript's gold."""
    gold = _manuscript(sample_id)
    rows = tuple(
        item
        for item in (
            _drop_unchanged(row, gold)
            for row in e1_instances(sample_id) + _e2(sample_id) + _e3(sample_id) + _e4(sample_id)
        )
        if item is not None
    )
    rows = _resolve_duplicate_site(rows)
    bad = sorted({item.klass for item in rows} - _ROW_CLASSES)
    if bad:
        raise RuntimeError(f"A class outside the library entered a row: {bad}")
    ids = [item.id for item in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("A fault-instance id is repeated.")
    return rows


def _resolve_duplicate_site(rows: tuple[FaultInstance, ...]) -> tuple[FaultInstance, ...]:
    """One singleton site keeps the class that blank actually is.

    The comparison is the set of blanks, not the written text. A direct
    recognition row (DIR) owns the site, so the EXE copy and any one-blank RSN
    are dropped. A site with no DIR is an execution parent: the EXE stays and
    the one-blank RSN is dropped. A reasoning joint writes at least two blanks,
    so it does not share a set with a parent EXE. An ALG on the same blank is
    not compared.
    """
    grouped: dict[frozenset[str], list[FaultInstance]] = {}
    rest: list[FaultInstance] = []
    for item in rows:
        if item.klass in {DIR, RSN, EXE}:
            grouped.setdefault(frozenset(blank for blank, _value in item.injector), []).append(item)
        else:
            rest.append(item)
    kept = list(rest)
    for sites, items in grouped.items():
        if len(sites) != 1:
            kept.extend(items)
            continue
        if any(item.klass == DIR for item in items):
            kept.extend(item for item in items if item.klass == DIR)
        elif any(item.klass == EXE for item in items):
            kept.extend(item for item in items if item.klass == EXE)
        else:
            kept.extend(items)
    order = {item.id: index for index, item in enumerate(rows)}
    kept.sort(key=lambda item: order[item.id])
    return tuple(kept)


def _drop_unchanged(item: FaultInstance, gold: dict[str, str]) -> FaultInstance | None:
    """Drop a row that would write the manuscript back unchanged."""
    kept = tuple(
        (blank, value)
        for blank, value in item.injector
        if normalize(value) != normalize(gold.get(blank, ""))
    )
    if not kept:
        return None
    if len(kept) == len(item.injector):
        return item
    direct, potential = _reach(tuple(blank for blank, _value in kept))
    return replace(item, injector=kept, direct=direct, potential=potential)


def _wrong_option(blank: str, gold_text: str) -> str | None:
    """Another choice for this blank. None when every option is the gold."""
    gold_key = normalize(gold_text)
    wrong = [option for option in _options().get(blank, ()) if normalize(option) != gold_key]
    if not wrong:
        return None
    return min(wrong, key=normalize)


def _e1(sample_id: str) -> tuple[FaultInstance, ...]:
    recognition = _recognition_index()
    authored, _edges = _authored()
    statuses = _statuses()
    reads = {slot for group in _READS.values() for slot in group}
    gold = _manuscript(sample_id)
    found = []
    for iid in sorted(authored):
        if statuses.get(iid) == "latent":
            continue
        blanks = set(recognition.get(iid, ()))
        blanks |= {target for target in authored[iid] if target in reads}
        if not blanks:
            continue
        injector = tuple(
            (blank, alt)
            for blank in sorted(blanks, key=int)
            if (alt := _wrong_option(blank, gold.get(blank, ""))) is not None
        )
        if not injector:
            continue
        direct, potential = _reach(tuple(blank for blank, _value in injector))
        _process, scope = _scope_of(iid)
        found.append(FaultInstance(f"{DIR}:{iid}", DIR, scope, injector, direct, potential))
    if not found:
        raise RuntimeError(f"{sample_id} has no DIR instance.")
    return tuple(found)


def _e2(sample_id: str) -> tuple[FaultInstance, ...]:
    """One counterfactual pole per branch, in the process that holds the slots."""
    found = []
    for name, assignment in branch_table(sample_id).items():
        by_process: dict[str, list[tuple[str, str]]] = {}
        for slot, value in assignment.slots:
            by_process.setdefault(NODE_BY_ID[slot].process, []).append((slot, value))
        modality = _BRANCH_MODALITY.get(name, "explicit")
        for process, pairs in by_process.items():
            ordered = tuple(sorted(pairs, key=lambda item: int(item[0])))
            direct, potential = _reach(tuple(slot for slot, _value in ordered))
            found.append(FaultInstance(
                f"{RSN}:{name}@{process}",
                RSN,
                f"{process}/{modality}",
                ordered,
                direct,
                potential,
            ))
    return tuple(found)


def _e3(sample_id: str) -> tuple[FaultInstance, ...]:
    """One concept-level fault per parent blank.

    The id names the failed judgment. It does not name the option that was
    written. The witness is a text other than this manuscript's gold, and it
    must move at least one calculator. Ties take the lexicographically
    smallest text. A parent with no such text is not a row. A child that
    reads the parent is propagation.
    """
    gold = _manuscript(sample_id)
    moved: dict[str, dict[str, int]] = {}
    for node_id in calculator_nodes():
        if node_id in {"36", "38"}:
            continue
        seen: set[tuple[str, str]] = set()
        for slot in slots(node_id, sample_id):
            if normalize(slot.alt) == normalize(gold.get(slot.node, "")):
                continue
            answer, _reason = _outcome(sample_id, node_id, slot.node, slot.alt)
            if not answer or normalize(answer) == normalize(gold.get(node_id, "")):
                continue
            pair = (slot.node, slot.alt)
            if pair in seen:
                continue
            seen.add(pair)
            changed = moved.setdefault(slot.node, {})
            changed[slot.alt] = changed.get(slot.alt, 0) + 1
    found = []
    for parent in sorted(moved, key=int):
        counts = moved[parent]
        alt = min(counts, key=lambda text: (-counts[text], normalize(text)))
        direct, potential = _reach((parent,))
        concept = _concept(parent)
        found.append(FaultInstance(
            f"{EXE}:{concept}",
            EXE,
            NODE_BY_ID[parent].process,
            ((parent, alt),),
            direct,
            potential,
        ))
    if not found:
        raise RuntimeError(f"{sample_id} has no EXE instance.")
    return tuple(found)


def _e4(sample_id: str) -> tuple[FaultInstance, ...]:
    """One algebra error per blank that asks for a matrix, vector, or formula."""
    gold = _manuscript(sample_id)
    found = []
    for node_id in sorted(NODE_BY_ID, key=int):
        node = NODE_BY_ID[node_id]
        if node.answer_type not in _MATH_KINDS or node_id in SIGN_COMPONENTS:
            continue
        text = gold.get(node_id, "")
        if not text:
            raise RuntimeError(f"ID{node_id} of {sample_id} has no gold, so an algebra error cannot be written.")
        rewritten, _form = _algebraic_slip(text, node.answer_type, f"ID{node_id}")
        direct, potential = _reach((node_id,))
        found.append(FaultInstance(
            f"{ALG}:ID{node_id}",
            ALG,
            f"{node.role} algebra error",
            ((node_id, rewritten),),
            direct,
            potential,
        ))
    if not found:
        raise RuntimeError(f"{sample_id} has no ALG instance.")
    return tuple(found)


def _algebraic_slip(text: str, kind: str, node_id: str) -> tuple[str, str]:
    """A local writing error. Transpose and full negation stay with EXE."""
    form = parse_answer(text, kind)
    if form.status != "ok":
        raise RuntimeError(f"The gold of {node_id} cannot parse: {form.status} {form.note}")
    if kind == "matrix":
        rewritten, name = _slip_matrix(form.value)
    elif kind == "vector":
        rewritten, name = _slip_vector(form.value)
    else:
        rewritten, name = _slip_expression(form.value)
    got = parse_answer(rewritten, kind)
    if got.status != "ok":
        raise RuntimeError(f"The algebra error of {node_id} cannot parse: {got.status} {got.note}")
    hits = match_predicate(got, form)
    if hits != ("P7",):
        raise RuntimeError(f"The algebra error of {node_id} hit {hits}, so it is not one structural difference.")
    if kind == "expression" and sp.simplify(got.value) == 0:
        raise RuntimeError(f"The algebra error of {node_id} simplified to 0.")
    return rewritten, name


def _slip_matrix(matrix) -> tuple[str, str]:
    grid = sp.Matrix(matrix)
    for row in range(3):
        for column in range(row + 1, 3):
            upper = sp.simplify(grid[row, column])
            lower = sp.simplify(grid[column, row])
            if upper == 0 or sp.simplify(upper + lower) != 0:
                continue
            grid[row, column] = -grid[row, column]
            return _matrix_text(grid), f"the sine sign in row {row + 1} column {column + 1} is flipped"
    for row in range(3):
        for column in range(3):
            if row == column or sp.simplify(grid[row, column]) == 0:
                continue
            grid[row, column] = -grid[row, column]
            return _matrix_text(grid), f"the sign in row {row + 1} column {column + 1} is flipped"
    grid[0, 1] = grid[0, 1] + 1
    return _matrix_text(grid), "the zero in row 1 column 2 was written as 1"


def _slip_vector(matrix) -> tuple[str, str]:
    parsed = sp.Matrix(matrix)
    if parsed.shape == (1, 3):
        values = [parsed[0, index] for index in range(3)]
    else:
        values = [parsed[index, 0] for index in range(3)]
    nonzero = [index for index, value in enumerate(values) if sp.simplify(value) != 0]
    if len(nonzero) >= 2:
        index = nonzero[0]
        values[index] = -values[index]
        return _vector_text(values), f"the sign of component {index + 1} is flipped"
    index = ((nonzero[0] + 1) % 3) if nonzero else 1
    values[index] = values[index] + 1
    return _vector_text(values), f"zero component {index + 1} was written as 1"


def _show_expr(expr) -> str:
    rendered = _text(expr)
    return rendered.replace("asin", "arcsin").replace("atan", "arctan")


def _slip_expression(expr) -> tuple[str, str]:
    inverted = _slip_inverse(expr)
    if inverted is not None:
        return _show_expr(inverted[0]), inverted[1]
    if expr.is_Add and len(expr.args) >= 2:
        args = list(expr.args)
        args[-1] = -args[-1]
        return _show_expr(sp.Add(*args)), "the sign of one term is flipped"
    numer, denom = sp.fraction(sp.together(expr))
    if sp.simplify(denom - 1) != 0:
        return _show_expr(numer * denom), "a quotient was written as a product"
    summed = _product_to_sum(expr)
    if summed is not None:
        return _show_expr(summed), "a product was written as a sum"
    angle = sp.cos(sp.Symbol("gamma_tp"))
    return _show_expr(expr * angle), "one term was also multiplied by cos(γ_tp)"


def _slip_inverse(expr):
    sign = 1
    body = expr
    inverse = {sp.atan, sp.asin}
    if expr.is_Mul and expr.args and expr.args[0] == -1 and getattr(expr.args[-1], "func", None) in inverse:
        sign = -1
        body = sp.Mul(*expr.args[1:])
    if getattr(body, "func", None) not in inverse:
        return None
    arg = body.args[0]
    numer, denom = sp.fraction(sp.together(arg))
    if sp.simplify(denom - 1) != 0:
        new_arg = numer * denom
        name = "the quotient inside the inverse function was written as a product"
    elif arg.is_Add and len(arg.args) >= 2:
        args = list(arg.args)
        args[-1] = -args[-1]
        new_arg = sp.Add(*args)
        name = "the sign of one term inside the inverse function is flipped"
    else:
        new_arg = arg + 1
        name = "1 was added to the argument of the inverse function"
    return sign * body.func(new_arg), name


def _product_to_sum(expr):
    if not expr.is_Mul:
        return None
    sign = sp.Integer(1)
    factors = []
    for arg in expr.as_ordered_factors():
        if arg.is_number:
            sign *= arg
        else:
            factors.append(arg)
    if len(factors) < 2:
        return None
    return sign * sp.Add(*factors)


def _reach(injected: tuple[str, ...], extra: tuple[str, ...] = ()) -> tuple[frozenset[str], frozenset[str]]:
    """Direct columns are the injected blanks. Potential adds descendant columns."""
    closure = set(injected) | set(extra)
    pending = list(closure)
    for node_id in pending:
        if node_id not in NODE_BY_ID:
            raise RuntimeError(f"Injection point ID{node_id} is not a public node.")
        closure.update(diagnosis_descendants(node_id))
    columns = by_node()
    direct = frozenset(column.id for node_id in injected for column in columns[node_id])
    potential = frozenset(column.id for node_id in closure for column in columns[node_id])
    return direct, potential


# These two blanks share one role string but ask different judgments.
_ROLE_QUALIFIER = {
    "119": " positive direction",
    "125": " axis correspondence",
}


def _concept(node_id: str) -> str:
    """Judgment that failed. The chosen option is not part of the name."""
    node = NODE_BY_ID[node_id]
    if _axis_judgment(node):
        kind = "axial judgment failed"
    elif _angle_judgment(node):
        kind = "angle judgment failed"
    else:
        kind = "judgment failed"
    role = node.role + _ROLE_QUALIFIER.get(node_id, "")
    return f"{node.process}·{role}·{kind}"


def _axis_judgment(node) -> bool:
    """The blank chooses which axis or axis component is in force."""
    if node.answer_type != "choice":
        return False
    key = node.target_definition
    if key.endswith("_axis") or key.endswith("_component"):
        return True
    return node.role.endswith(("normal", "axial", "lateral", "coordinate axis", "absolute axis"))


def _angle_judgment(node) -> bool:
    """The blank judges the sense of an angle, not which value was written."""
    if node.answer_type not in {"choice", "sign"} or _axis_judgment(node):
        return False
    return any(token in node.role for token in ("angle of attack", "sideslip", "roll", "right-hand", "installation angle", "elastic", "turning sense"))


def _scope_of(instance_id: str) -> tuple[str, str]:
    _global, _, rest = instance_id.partition("@")
    process, _, _slot = rest.partition(":")
    return process, process

