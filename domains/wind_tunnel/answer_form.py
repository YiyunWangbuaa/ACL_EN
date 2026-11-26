# -*- coding: utf-8 -*-
"""Parse one answer and name its formal relation to the gold answer.

The parser does not decide a fault mode. It returns an ``AnswerForm``, the
cell differences, and the P1–P7 hits. P4 and P5 stay unmatched until T4
passes the factor table. An empty match tuple means the relation does not
apply: the two answers are equivalent, or the type has no structural
predicate. P6 is a wrong shape and routes to ALG. Unparsed, empty, and
out-of-option answers route to R6. ``cell_delta`` is ``None`` when it
cannot be computed, and ``{("const",)}`` when an expression differs by a
constant. P7 is any structural difference P1–P6 miss; ``is_local`` says
whether that difference is a proper subset of the cells.
"""
from __future__ import annotations

import itertools
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cloze_score import (
    choice_letter,
    extract_payload,
    identified_option,
    normalize,
    sympy_equal,
)
from cloze_score import _parse_expr, _sympy_modules

_ANSWER_TYPES = ("matrix", "vector", "expression", "choice", "sign")
_STRUCTURAL = ("matrix", "vector", "expression")
_STATUSES = ("ok", "unparseable", "wrong_shape", "empty", "out_of_options")
_PROTOCOL = {"unparseable", "empty", "out_of_options"}
_SIGN_CANON = {
    "positive": "positive",
    "positive sign": "positive",
    "+": "positive",
    "＋": "positive",
    "negative": "negative",
    "negative sign": "negative",
    "-": "negative",
    "－": "negative",
    "−": "negative",
    "indeterminate": "indeterminate",
}
_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_OBSERVATIONS = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "contract_consistency"
    / "error_workflow"
)


@dataclass(frozen=True)
class AnswerForm:
    raw: str
    kind: str
    status: str
    value: object | None
    cells: tuple
    note: str = ""


def clean_display(text: str) -> str:
    """Display cleanup only. Algebraic equivalence stays with sympy."""
    value = str(text or "")
    value = value.replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("arcsin", "asin").replace("arctan", "atan")
    value = value.replace("·", "*").replace("×", "*").replace("⋅", "*")
    value = value.replace("（", "(").replace("）", ")")
    while True:
        updated = _FRAC.sub(r"(\1)/(\2)", value)
        if updated == value:
            break
        value = updated
    return re.sub(r"\s+", " ", value).strip()


def parse_answer(
    raw: str,
    answer_type: str,
    options: Sequence[str] | None = None,
) -> AnswerForm:
    """Parse one answer. A bad string returns a status instead of raising."""
    kind = answer_type if answer_type in _ANSWER_TYPES else "none"
    source = str(raw or "")
    text = extract_payload(source)
    if not text.strip():
        return _form(source, kind, "empty", note="the answer is empty")
    try:
        if kind in {"choice", "sign"}:
            return _parse_word(source, text, kind, options)
        if kind in _STRUCTURAL:
            return _parse_structural(source, text, kind)
    except Exception as exc:
        return _form(source, kind, "unparseable", note=f"parse failed: {exc}")
    return _form(source, "none", "unparseable", note=f"unknown answer_type: {answer_type}")


def cell_delta(pred: AnswerForm, gold: AnswerForm) -> set | None:
    """Indices where the two parsed values differ.

    ``None`` means the difference cannot be computed. An empty set means
    the values match. Matrix indices are ``(i, j)``, vector indices are
    ``(i,)``, and an expression contributes the symbol names left in the
    difference. A constant expression difference is ``{("const",)}``.
    """
    if pred.status != "ok" or gold.status != "ok" or pred.kind != gold.kind:
        return None
    if pred.kind in {"matrix", "vector"}:
        return {
            index
            for (index, left), (_, right) in zip(pred.cells, gold.cells)
            if not _exprs_equal(left, right)
        }
    if pred.kind == "expression":
        if _exprs_equal(pred.value, gold.value):
            return set()
        return set(_difference_names(pred.value, gold.value)) or {("const",)}
    return set()


def is_local(pred: AnswerForm, gold: AnswerForm) -> bool | None:
    """True when the difference is a nonempty proper subset of the gold cells.

    P7 itself is any structural difference. This flag is the narrower
    reading. ``None`` follows ``cell_delta``.
    """
    delta = cell_delta(pred, gold)
    if delta is None:
        return None
    if not delta:
        return False
    return delta < _cell_universe(gold)


def match_predicate(
    pred: AnswerForm,
    gold: AnswerForm,
    factors: Sequence[tuple[str, object]] | None = None,
) -> tuple[str, ...]:
    """P1–P7 hits, or ``("R6",)`` for a protocol failure.

    P6 is only ``wrong_shape`` and routes to ALG. Empty, unparsed, and
    out-of-option answers are R6. Several structural hits can fire
    together; P7 is the structural remainder, not necessarily a local one.
    ``factors`` is the T4 table ``(name, matrix)``. ``None`` leaves P4 and
    P5 unmatched. The identity permutation is not a P5 hit.
    """
    if pred.status == "wrong_shape":
        return ("P6",)
    if pred.status in _PROTOCOL or gold.status != "ok" or pred.kind != gold.kind:
        return ("R6",)
    if _same_form(pred, gold):
        return ()
    hits: list[str] = []
    if pred.kind == "matrix" and _exprs_equal(pred.value, gold.value.T):
        hits.append("P1")
    if pred.kind in _STRUCTURAL and _exprs_equal(pred.value, -gold.value):
        hits.append("P2")
    if pred.kind == "matrix" and _is_identity(pred.value) and not _is_identity(gold.value):
        hits.append("P3")
    if pred.kind == "matrix" and factors is not None and _matches_factor(pred.value, factors):
        hits.append("P4")
    if pred.kind == "matrix" and factors is not None and _matches_permutation(pred.value, factors):
        hits.append("P5")
    if hits:
        return tuple(hits)
    if pred.kind in _STRUCTURAL:
        return ("P7",)
    return ()


def _form(raw: str, kind: str, status: str, value=None, cells=(), note: str = "") -> AnswerForm:
    if status not in _STATUSES:
        raise RuntimeError(f"Unknown parse status: {status}")
    return AnswerForm(raw=raw, kind=kind, status=status, value=value, cells=tuple(cells), note=note)


def _options(options: Sequence[str] | None) -> list[str]:
    return [str(item) for item in (options or []) if str(item) not in {"", "none"}]


def _mapping_note(options: list[str]) -> str:
    if not options:
        return ""
    return "map=" + "|".join(f"{choice_letter(index)}:{item}" for index, item in enumerate(options))


def _unique_prefix(text: str, options: list[str]) -> str | None:
    key = normalize(text)
    if not key:
        return None
    owners = [item for item in options if normalize(item).startswith(key) and normalize(item) != key]
    if len(owners) == 1:
        return owners[0]
    return None


def _canon_sign(text: str) -> str | None:
    return _SIGN_CANON.get(text.strip())


def _parse_word(raw: str, text: str, kind: str, options: Sequence[str] | None) -> AnswerForm:
    choices = _options(options)
    note = _mapping_note(choices)
    selected = identified_option(text, choices) if choices else None
    via = "option"
    if selected is None and choices:
        selected = _unique_prefix(text, choices)
        via = "prefix"
    if selected is None and kind == "sign":
        canon = _canon_sign(text)
        if canon is not None and (not choices or canon in choices):
            selected = canon
            via = "canon"
    if selected is None and not choices and kind == "choice":
        value = normalize(text)
        if not value:
            return _form(raw, kind, "empty", note="the answer is empty")
        return _form(raw, kind, "ok", value=value, note="via=text")
    if selected is None:
        if choices:
            return _form(raw, kind, "out_of_options", note="not among the options")
        return _form(raw, kind, "unparseable", note="not positive, negative, or indeterminate")
    value = _canon_sign(selected) or selected if kind == "sign" else selected
    extra = f"via={via}"
    note = extra if not note else f"{note};{extra}"
    return _form(raw, kind, "ok", value=value, note=note)


def _known_symbols(expr) -> bool:
    """A bare word such as nope must not become a symbol. Axis codes may."""
    for symbol in expr.free_symbols:
        name = symbol.name
        if not name.isascii() or not name.replace("_", "").isalnum():
            return False
    return True


def _parse_structural(raw: str, text: str, kind: str) -> AnswerForm:
    parsed = _parse_expr(clean_display(text), *_sympy_modules())
    sympy, _locals = _sympy_modules()
    if not _known_symbols(parsed):
        return _form(raw, kind, "unparseable", note="contains an undeclared symbol")
    if isinstance(parsed, sympy.MatrixBase):
        if kind == "matrix":
            if parsed.shape != (3, 3):
                return _form(raw, kind, "wrong_shape", value=parsed, note=f"shape is {parsed.shape}; a matrix must be 3×3")
            cells = tuple(
                ((row, column), parsed[row, column])
                for row in range(3)
                for column in range(3)
            )
            return _form(raw, kind, "ok", value=parsed, cells=cells)
        if kind == "vector":
            entries = _vector_entries(parsed)
            if entries is None:
                return _form(raw, kind, "wrong_shape", value=parsed, note=f"shape is {parsed.shape}; a vector needs 3 components")
            cells = tuple((((index,), entry) for index, entry in enumerate(entries)))
            return _form(raw, kind, "ok", value=parsed, cells=cells)
        return _form(raw, kind, "wrong_shape", value=parsed, note=f"the expression parsed as a matrix {parsed.shape}")
    if kind != "expression":
        return _form(raw, kind, "wrong_shape", value=parsed, note=f"{kind} parsed as a scalar")
    return _form(raw, kind, "ok", value=parsed, cells=_symbol_names(parsed))


def _vector_entries(parsed):
    if parsed.shape == (1, 3):
        return tuple(parsed[0, index] for index in range(3))
    if parsed.shape == (3, 1):
        return tuple(parsed[index, 0] for index in range(3))
    return None


def _symbol_names(expr) -> tuple[str, ...]:
    return tuple(sorted(symbol.name for symbol in expr.free_symbols))


def _difference_names(left, right) -> tuple[str, ...]:
    sympy, _locals = _sympy_modules()
    try:
        diff = sympy.simplify(left - right)
    except Exception:
        diff = left - right
    return _symbol_names(diff)


def _render(expr) -> str:
    sympy, _locals = _sympy_modules()
    if isinstance(expr, sympy.MatrixBase):
        rows = []
        for row in range(expr.rows):
            cells = ", ".join(_render(expr[row, column]) for column in range(expr.cols))
            rows.append(f"[{cells}]")
        return "[" + ", ".join(rows) + "]"
    return str(expr)


@lru_cache(maxsize=4096)
def _cached_sympy_equal(left: str, right: str) -> bool:
    return sympy_equal(left, right)


def _exprs_equal(left, right) -> bool:
    """Same comparison ``cloze_score.sympy_equal`` uses, on rendered forms."""
    return _cached_sympy_equal(_render(left), _render(right))


def _cell_universe(form: AnswerForm) -> set:
    if form.kind in {"matrix", "vector"}:
        return {index for index, _value in form.cells}
    if form.kind == "expression":
        return set(form.cells)
    return set()


def _same_form(pred: AnswerForm, gold: AnswerForm) -> bool:
    if pred.kind in _STRUCTURAL:
        return _exprs_equal(pred.value, gold.value)
    return pred.value == gold.value


def _is_identity(expr) -> bool:
    sympy, _locals = _sympy_modules()
    return _exprs_equal(expr, sympy.eye(3))


def _matrix_factors(factors: Sequence[tuple[str, object]]) -> list | None:
    sympy, _locals = _sympy_modules()
    matrices = []
    for _name, value in factors:
        if not isinstance(value, sympy.MatrixBase) or value.shape != (3, 3):
            return None
        matrices.append(value)
    return matrices


def _product(matrices: Sequence):
    found = matrices[0]
    for item in matrices[1:]:
        found = found * item
    return found


def _matches_factor(pred, factors: Sequence[tuple[str, object]]) -> bool:
    matrices = _matrix_factors(factors)
    if not matrices:
        return False
    return any(_exprs_equal(pred, item) for item in matrices)


def _matches_permutation(pred, factors: Sequence[tuple[str, object]]) -> bool:
    matrices = _matrix_factors(factors)
    if matrices is None or len(matrices) < 2 or len(matrices) > 7:
        return False
    try:
        original = _product(matrices)
    except Exception:
        return False
    if _exprs_equal(pred, original):
        return False
    for order in itertools.permutations(range(len(matrices))):
        if order == tuple(range(len(matrices))):
            continue
        try:
            produced = _product([matrices[index] for index in order])
        except Exception:
            return False
        if _exprs_equal(pred, produced):
            return True
    return False


def _load_observations() -> list[dict]:
    files = sorted(path for path in _OBSERVATIONS.glob("*/observations.jsonl") if path.is_file())
    if len(files) != 4:
        raise RuntimeError(f"Expected 4 observation files, found {len(files)}.")
    rows = []
    for path in files:
        found = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_file"] = path.parent.name
                found.append(row)
        if len(found) != 176:
            raise RuntimeError(f"{path.parent.name} should have 176 rows, found {len(found)}.")
        rows.extend(found)
    return rows


def _check_display() -> None:
    if clean_display("2cosθ + 1") != "2cosθ + 1":
        raise RuntimeError("Cleaning must not reorder terms.")
    if clean_display("1 + 2cosθ") != "1 + 2cosθ":
        raise RuntimeError("Cleaning must not reorder terms.")
    if not sympy_equal("2cosθ + 1", "1 + 2cosθ"):
        raise RuntimeError("An equivalent expression with a different term order was not given to sympy.")
    if clean_display("arctan(x)") != "atan(x)" or clean_display("arcsin(x)") != "asin(x)":
        raise RuntimeError("Inverse trig functions were not normalized to atan/asin.")
    if clean_display("（a·b）×c") != "(a*b)*c":
        raise RuntimeError(clean_display("（a·b）×c"))
    if clean_display(r"\left(\frac{a}{b}\right)") != "((a)/(b))":
        raise RuntimeError(clean_display(r"\left(\frac{a}{b}\right)"))
    matrix = "[[1, 0, 0],\n  [0, 1, 0],\n  [0, 0, 1]]"
    if clean_display(matrix) != "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]":
        raise RuntimeError(clean_display(matrix))


def _must(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_predicates() -> None:
    gold = parse_answer("[[0, -1, 0], [1, 0, 0], [0, 0, 1]]", "matrix")
    transposed = parse_answer("[[0, 1, 0], [-1, 0, 0], [0, 0, 1]]", "matrix")
    negated = parse_answer("[[0, 1, 0], [-1, 0, 0], [0, 0, -1]]", "matrix")
    identity = parse_answer("[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "matrix")
    one_cell = parse_answer("[[0, -1, 0], [1, 0, 0], [0, 0, -1]]", "matrix")
    _must(match_predicate(transposed, gold) == ("P1",), "P1 should hit a transpose")
    _must(match_predicate(gold, gold) == (), "P1 should not hit the gold itself")
    _must(match_predicate(negated, gold) == ("P2",), "P2 should hit a negated matrix")
    _must(match_predicate(identity, gold) == ("P3",), "P3 should hit the identity")
    _must("P3" not in match_predicate(gold, identity), "The P3 counterexample is a model that did not write the identity")
    _must(match_predicate(gold, identity) == ("P7",), "Writing the identity as a rotation matrix is still a structural difference")
    _must(match_predicate(one_cell, gold) == ("P7",), "A one-cell difference should be P7")
    _must(cell_delta(one_cell, gold) == {(2, 2)}, f"The cell delta should be one cell, got {cell_delta(one_cell, gold)}")
    _must(is_local(one_cell, gold) is True, "A one-cell difference should be local")
    _must(match_predicate(transposed, gold) == ("P1",), "A transpose must not fall through to P7")
    _must(cell_delta(gold, gold) == set(), "Identical matrices should have an empty cell delta")

    vector = parse_answer("[1, 0, -1]", "vector")
    vector_neg = parse_answer("[-1, 0, 1]", "vector")
    vector_cell = parse_answer("[1, 1, -1]", "vector")
    _must(vector.status == "ok" and len(vector.cells) == 3, "A vector should have 3 components")
    _must(match_predicate(vector_neg, vector) == ("P2",), "P2 should hit a negated vector")
    _must(match_predicate(vector, vector) == (), "P2 should not hit the vector gold")
    _must(match_predicate(vector_cell, vector) == ("P7",), "A local vector difference should be P7")
    _must(cell_delta(vector_cell, vector) == {(1,)}, f"Unexpected vector cell delta: {cell_delta(vector_cell, vector)}")

    expr = parse_answer("sin(α_M)", "expression")
    expr_neg = parse_answer("-sin(α_M)", "expression")
    expr_local = parse_answer("cos(α_M)", "expression")
    _must(match_predicate(expr_neg, expr) == ("P2",), "P2 should hit a negated expression")
    _must(match_predicate(expr, expr) == (), "P2 should not hit the expression gold")
    _must(match_predicate(expr_local, expr) == ("P7",), "A local expression difference should be P7")
    _must(cell_delta(expr_local, expr) == {"alpha_M"}, f"Unexpected expression cell delta: {cell_delta(expr_local, expr)}")
    _must(expr.cells == ("alpha_M",), f"Unexpected expression variable order: {expr.cells}")

    sympy, _locals = _sympy_modules()
    first = sympy.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    second = sympy.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    factors = (("R1", first), ("R2", second))
    single = parse_answer(_render(first), "matrix")
    product = parse_answer(_render(first * second), "matrix")
    swapped = parse_answer(_render(second * first), "matrix")
    _must(match_predicate(single, product, factors) == ("P4",), "P4 should hit a single factor")
    _must("P4" not in match_predicate(product, identity, factors), "P4 should not hit a product of two factors")
    _must(match_predicate(swapped, product, factors) == ("P5",), "P5 should hit a swapped product")
    _must("P5" not in match_predicate(product, identity, factors), "P5 should not hit the original product order")
    _must("P4" not in match_predicate(single, product) and "P5" not in match_predicate(swapped, product), "P4 and P5 do not apply without a factor table")

    left = sympy.Matrix([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1]])
    right = sympy.Matrix([[-1, -1, -1], [-1, -1, 0], [-1, 0, 0]])
    both = (("A", left), ("B", right))
    gold_product = parse_answer(_render(left * right), "matrix")
    pred_product = parse_answer(_render(right * left), "matrix")
    _must(match_predicate(pred_product, gold_product, both) == ("P1", "P5"), "A transpose and a permutation that both hold should both be returned")

    broken = parse_answer("nope", "expression")
    wide = parse_answer(
        "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]",
        "matrix",
    )
    _must(broken.status == "unparseable" and match_predicate(broken, expr) == ("R6",), "cannot parse is R6")
    _must("P6" not in match_predicate(broken, expr), "cannot parse must not enter P6")
    _must(wide.status == "wrong_shape" and match_predicate(wide, gold) == ("P6",), "A 5×5 matrix should be P6")
    _must("P6" not in match_predicate(gold, gold), "A proper 3×3 matrix is not P6")
    blank = parse_answer("  ", "matrix")
    _must(blank.status == "empty" and match_predicate(blank, gold) == ("R6",), "A blank answer is empty and routes to R6")
    _must(cell_delta(broken, expr) is None, "cannot parse makes the cell delta None")
    constant = parse_answer("1", "expression")
    other = parse_answer("2", "expression")
    _must(match_predicate(constant, other) == ("P7",), "A constant difference is still P7")
    _must(cell_delta(constant, other) == {("const",)}, "A constant difference must not be an empty set")
    _must(is_local(constant, other) is False, "A constant difference is not a local cell difference")
    everywhere = parse_answer("[[2, 2, 2], [2, 2, 2], [2, 2, 2]]", "matrix")
    _must(match_predicate(everywhere, identity) == ("P7",), "A wholly different matrix still returns P7")
    _must(is_local(everywhere, identity) is False, "A wholly different matrix is not a local difference")
    outside = parse_answer("Z", "choice", ["body-axis frame", "balance frame", "wind frame"])
    _must(outside.status == "out_of_options" and match_predicate(outside, outside) == ("R6",), "An answer outside the options is R6")
    letter = parse_answer("C", "choice", None)
    _must(letter.status == "ok" and letter.value == normalize("C"), "A choice with no options falls back to the text")
    _must(match_predicate(letter, parse_answer("B", "choice", None)) == (), "A text difference with no options does not enter P6")

    options = ["positive", "negative", "indeterminate"]
    for index, option in enumerate(options):
        by_letter = parse_answer(choice_letter(index), "choice", options)
        by_text = parse_answer(option, "choice", options)
        _must(by_letter.status == "ok" and by_letter.value == option, f"Letter {choice_letter(index)} did not map back to the option")
        _must(by_text.value == option and "A:positive" in by_text.note, "The option map was not kept in the note")
    _must(parse_answer("up", "choice", ["upright", "inverted"]).value == "upright", "A unique prefix should hit upright")
    _must(parse_answer("negative sign", "sign").value == "negative", "negative sign canonicalizes to negative")
    kept = parse_answer("same", "sign", ["same", "opposite"])
    _must(kept.value == "same", "same and opposite stay out of positive, negative, and indeterminate")
    _must(parse_answer("", "choice", options).status == "empty", "An empty choice is empty")


def audit(dump: bool = False) -> Mapping[str, int]:
    """A1–A4. Raises when a row or a predicate case fails."""
    _check_display()
    _check_predicates()
    from domains.wind_tunnel.error_graph import NODE_BY_ID

    rows = _load_observations()
    failures = []
    disagreements = []
    wrong_rows = []
    counts = {code: 0 for code in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "R6")}
    none_count = 0
    for row in rows:
        node_id = str(row["node_id"])
        node = NODE_BY_ID.get(node_id)
        if node is None:
            failures.append(f"{row['_file']} ID{node_id} is not in the graph")
            continue
        kind = node.answer_type
        pred = parse_answer(row["model_answer"], kind, row.get("options"))
        gold = parse_answer(row["gold_answer"], kind, row.get("options"))
        if pred.status != "ok" or gold.status != "ok":
            failures.append(
                f"{row['_file']} ID{node_id} {kind} pred={pred.status}:{pred.note} gold={gold.status}:{gold.note}"
            )
        if kind in _STRUCTURAL and pred.status == "ok" and gold.status == "ok":
            ours = _exprs_equal(pred.value, gold.value)
            theirs = sympy_equal(row["model_answer"], row["gold_answer"])
            if ours != theirs:
                disagreements.append(f"{row['_file']} ID{node_id}")
        hits = match_predicate(pred, gold)
        if row["correct"] and hits:
            failures.append(f"{row['_file']} ID{node_id} gold was judged {hits}")
        if not row["correct"]:
            if kind in _STRUCTURAL and not hits:
                failures.append(f"{row['_file']} ID{node_id} wrong answer has no predicate")
            shown = "+".join(hits) if hits else "None"
            if not hits:
                none_count += 1
            for code in hits:
                counts[code] += 1
            wrong_rows.append((row, pred, gold, shown))
    if len(rows) != 704:
        failures.append(f"Expected 704 observation rows, found {len(rows)}")
    if failures or disagreements:
        detail = "; ".join((failures + disagreements)[:8])
        raise RuntimeError(f"Parse audit failed {len(failures)} rows, with {len(disagreements)} equivalence disagreements. {detail}")
    if dump:
        for row, pred, gold, shown in wrong_rows:
            delta = cell_delta(pred, gold)
            delta = "None" if delta is None else sorted(delta, key=_delta_key)
            print(
                f"{row['_file']}\tID{row['node_id']}\t{pred.kind}\t{shown}\t{delta}\t{pred.status}"
            )
            print(f"  gold {row['gold_answer']}")
            print(f"  pred {row['model_answer']}")
    summary = {
        "rows": len(rows),
        "parsed": len(rows) - len(failures),
        "wrong": len(wrong_rows),
        "none": none_count,
        "disagreements": len(disagreements),
        **counts,
    }
    return summary


def _delta_key(item) -> tuple:
    if isinstance(item, tuple):
        return (0, item)
    return (1, str(item))


def _print_summary(summary: Mapping[str, int]) -> None:
    print(
        "rows={rows} parsed={parsed} wrong={wrong} none={none} disagreements={disagreements}".format(
            **summary
        )
    )
    print(
        "P1={P1} P2={P2} P3={P3} P4={P4} P5={P5} P6={P6} P7={P7} R6={R6}".format(**summary)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audit answer parsing and P1-P7")
    parser.add_argument("--dump", action="store_true", help="Print the parse and predicates of wrong answers")
    args = parser.parse_args()
    _print_summary(audit(dump=args.dump))
