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
    "正": "正",
    "正号": "正",
    "+": "正",
    "＋": "正",
    "负": "负",
    "负号": "负",
    "-": "负",
    "－": "负",
    "−": "负",
    "不确定": "不确定",
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
        return _form(source, kind, "empty", note="答案为空")
    try:
        if kind in {"choice", "sign"}:
            return _parse_word(source, text, kind, options)
        if kind in _STRUCTURAL:
            return _parse_structural(source, text, kind)
    except Exception as exc:
        return _form(source, kind, "unparseable", note=f"解析失败：{exc}")
    return _form(source, "none", "unparseable", note=f"未知 answer_type：{answer_type}")


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
        raise RuntimeError(f"未知解析状态：{status}")
    return AnswerForm(raw=raw, kind=kind, status=status, value=value, cells=tuple(cells), note=note)


def _options(options: Sequence[str] | None) -> list[str]:
    return [str(item) for item in (options or []) if str(item) not in {"", "无"}]


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
            return _form(raw, kind, "empty", note="答案为空")
        return _form(raw, kind, "ok", value=value, note="via=text")
    if selected is None:
        if choices:
            return _form(raw, kind, "out_of_options", note="不在选项内")
        return _form(raw, kind, "unparseable", note="不是正、负或不确定")
    value = _canon_sign(selected) or selected if kind == "sign" else selected
    extra = f"via={via}"
    note = extra if not note else f"{note};{extra}"
    return _form(raw, kind, "ok", value=value, note=note)


def _known_symbols(expr) -> bool:
    """A bare word such as 不懂 must not become a symbol. Axis codes may."""
    for symbol in expr.free_symbols:
        name = symbol.name
        if not name.isascii() or not name.replace("_", "").isalnum():
            return False
    return True


def _parse_structural(raw: str, text: str, kind: str) -> AnswerForm:
    parsed = _parse_expr(clean_display(text), *_sympy_modules())
    sympy, _locals = _sympy_modules()
    if not _known_symbols(parsed):
        return _form(raw, kind, "unparseable", note="含有未声明的符号")
    if isinstance(parsed, sympy.MatrixBase):
        if kind == "matrix":
            if parsed.shape != (3, 3):
                return _form(raw, kind, "wrong_shape", value=parsed, note=f"形状是 {parsed.shape}，矩阵要 3×3")
            cells = tuple(
                ((row, column), parsed[row, column])
                for row in range(3)
                for column in range(3)
            )
            return _form(raw, kind, "ok", value=parsed, cells=cells)
        if kind == "vector":
            entries = _vector_entries(parsed)
            if entries is None:
                return _form(raw, kind, "wrong_shape", value=parsed, note=f"形状是 {parsed.shape}，向量要 3 个分量")
            cells = tuple((((index,), entry) for index, entry in enumerate(entries)))
            return _form(raw, kind, "ok", value=parsed, cells=cells)
        return _form(raw, kind, "wrong_shape", value=parsed, note=f"表达式解析成了矩阵 {parsed.shape}")
    if kind != "expression":
        return _form(raw, kind, "wrong_shape", value=parsed, note=f"{kind} 解析成了标量")
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
        raise RuntimeError(f"观测文件应为 4 份，现在是 {len(files)} 份。")
    rows = []
    for path in files:
        found = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_file"] = path.parent.name
                found.append(row)
        if len(found) != 176:
            raise RuntimeError(f"{path.parent.name} 应有 176 行，现在是 {len(found)} 行。")
        rows.extend(found)
    return rows


def _check_display() -> None:
    if clean_display("2cosθ + 1") != "2cosθ + 1":
        raise RuntimeError("清洗不能重排项。")
    if clean_display("1 + 2cosθ") != "1 + 2cosθ":
        raise RuntimeError("清洗不能重排项。")
    if not sympy_equal("2cosθ + 1", "1 + 2cosθ"):
        raise RuntimeError("项序不同的等价式没有交给 sympy。")
    if clean_display("arctan(x)") != "atan(x)" or clean_display("arcsin(x)") != "asin(x)":
        raise RuntimeError("反三角没有归一成 atan/asin。")
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
    _must(match_predicate(transposed, gold) == ("P1",), "P1 应命中转置")
    _must(match_predicate(gold, gold) == (), "P1 不应命中金标本身")
    _must(match_predicate(negated, gold) == ("P2",), "P2 应命中负矩阵")
    _must(match_predicate(identity, gold) == ("P3",), "P3 应命中单位阵")
    _must("P3" not in match_predicate(gold, identity), "P3 反例应是模型没写成单位阵")
    _must(match_predicate(gold, identity) == ("P7",), "单位阵被写成旋转矩阵仍是结构差")
    _must(match_predicate(one_cell, gold) == ("P7",), "单格差异应是 P7")
    _must(cell_delta(one_cell, gold) == {(2, 2)}, f"单元差应为一个格子，现在是 {cell_delta(one_cell, gold)}")
    _must(is_local(one_cell, gold) is True, "单格差异应是局部差")
    _must(match_predicate(transposed, gold) == ("P1",), "转置不能落到 P7")
    _must(cell_delta(gold, gold) == set(), "相同矩阵的单元差应为空")

    vector = parse_answer("[1, 0, -1]", "vector")
    vector_neg = parse_answer("[-1, 0, 1]", "vector")
    vector_cell = parse_answer("[1, 1, -1]", "vector")
    _must(vector.status == "ok" and len(vector.cells) == 3, "向量应有 3 个分量")
    _must(match_predicate(vector_neg, vector) == ("P2",), "P2 应命中负向量")
    _must(match_predicate(vector, vector) == (), "P2 不应命中向量金标")
    _must(match_predicate(vector_cell, vector) == ("P7",), "向量局部差应是 P7")
    _must(cell_delta(vector_cell, vector) == {(1,)}, f"向量单元差不对：{cell_delta(vector_cell, vector)}")

    expr = parse_answer("sin(α_M)", "expression")
    expr_neg = parse_answer("-sin(α_M)", "expression")
    expr_local = parse_answer("cos(α_M)", "expression")
    _must(match_predicate(expr_neg, expr) == ("P2",), "P2 应命中负表达式")
    _must(match_predicate(expr, expr) == (), "P2 不应命中表达式金标")
    _must(match_predicate(expr_local, expr) == ("P7",), "表达式局部差应是 P7")
    _must(cell_delta(expr_local, expr) == {"alpha_M"}, f"表达式单元差不对：{cell_delta(expr_local, expr)}")
    _must(expr.cells == ("alpha_M",), f"表达式变量序不对：{expr.cells}")

    sympy, _locals = _sympy_modules()
    first = sympy.Matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    second = sympy.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    factors = (("R1", first), ("R2", second))
    single = parse_answer(_render(first), "matrix")
    product = parse_answer(_render(first * second), "matrix")
    swapped = parse_answer(_render(second * first), "matrix")
    _must(match_predicate(single, product, factors) == ("P4",), "P4 应命中单项")
    _must("P4" not in match_predicate(product, identity, factors), "P4 不应命中两个因子的乘积")
    _must(match_predicate(swapped, product, factors) == ("P5",), "P5 应命中交换后的乘积")
    _must("P5" not in match_predicate(product, identity, factors), "P5 不应命中原顺序乘积")
    _must("P4" not in match_predicate(single, product) and "P5" not in match_predicate(swapped, product), "未给因子表时 P4/P5 不适用")

    left = sympy.Matrix([[-1, -1, -1], [-1, -1, 0], [-1, 0, -1]])
    right = sympy.Matrix([[-1, -1, -1], [-1, -1, 0], [-1, 0, 0]])
    both = (("A", left), ("B", right))
    gold_product = parse_answer(_render(left * right), "matrix")
    pred_product = parse_answer(_render(right * left), "matrix")
    _must(match_predicate(pred_product, gold_product, both) == ("P1", "P5"), "转置与置换同时成立时应同时返回")

    broken = parse_answer("不懂", "expression")
    wide = parse_answer(
        "[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]",
        "matrix",
    )
    _must(broken.status == "unparseable" and match_predicate(broken, expr) == ("R6",), "无法解析应是 R6")
    _must("P6" not in match_predicate(broken, expr), "无法解析不能进 P6")
    _must(wide.status == "wrong_shape" and match_predicate(wide, gold) == ("P6",), "5×5 应是 P6")
    _must("P6" not in match_predicate(gold, gold), "正常 3×3 不是 P6")
    blank = parse_answer("  ", "matrix")
    _must(blank.status == "empty" and match_predicate(blank, gold) == ("R6",), "空白应是 empty，并归 R6")
    _must(cell_delta(broken, expr) is None, "无法解析时单元差应是 None")
    constant = parse_answer("1", "expression")
    other = parse_answer("2", "expression")
    _must(match_predicate(constant, other) == ("P7",), "常数差仍是 P7")
    _must(cell_delta(constant, other) == {("const",)}, "常数差不能是空集")
    _must(is_local(constant, other) is False, "常数差不是局部单元差")
    everywhere = parse_answer("[[2, 2, 2], [2, 2, 2], [2, 2, 2]]", "matrix")
    _must(match_predicate(everywhere, identity) == ("P7",), "整矩阵不同仍返回 P7")
    _must(is_local(everywhere, identity) is False, "整矩阵不同不是局部差")
    outside = parse_answer("Z", "choice", ["体轴系", "天平系", "风轴系"])
    _must(outside.status == "out_of_options" and match_predicate(outside, outside) == ("R6",), "选项外应是 R6")
    letter = parse_answer("C", "choice", None)
    _must(letter.status == "ok" and letter.value == normalize("C"), "无选项的选择应退回文本")
    _must(match_predicate(letter, parse_answer("B", "choice", None)) == (), "无选项的文本差不进 P6")

    options = ["正", "负", "不确定"]
    for index, option in enumerate(options):
        by_letter = parse_answer(choice_letter(index), "choice", options)
        by_text = parse_answer(option, "choice", options)
        _must(by_letter.status == "ok" and by_letter.value == option, f"字母 {choice_letter(index)} 没有映回选项")
        _must(by_text.value == option and "A:正" in by_text.note, "选项映射没有留在备注里")
    _must(parse_answer("正", "choice", ["正装", "反装"]).value == "正装", "唯一前缀应命中正装")
    _must(parse_answer("负号", "sign").value == "负", "负号应归一为负")
    kept = parse_answer("相同", "sign", ["相同", "相反"])
    _must(kept.value == "相同", "相同/相反不收进正负不确定")
    _must(parse_answer("", "choice", options).status == "empty", "空选择应是 empty")


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
            failures.append(f"{row['_file']} ID{node_id} 不在图里")
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
            failures.append(f"{row['_file']} ID{node_id} 金标被判成 {hits}")
        if not row["correct"]:
            if kind in _STRUCTURAL and not hits:
                failures.append(f"{row['_file']} ID{node_id} 错答没有谓词")
            shown = "+".join(hits) if hits else "None"
            if not hits:
                none_count += 1
            for code in hits:
                counts[code] += 1
            wrong_rows.append((row, pred, gold, shown))
    if len(rows) != 704:
        failures.append(f"观测应有 704 行，现在是 {len(rows)} 行")
    if failures or disagreements:
        detail = "；".join((failures + disagreements)[:8])
        raise RuntimeError(f"解析验收失败 {len(failures)} 条，等价分歧 {len(disagreements)} 条。{detail}")
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

    parser = argparse.ArgumentParser(description="验收答案解析与 P1–P7")
    parser.add_argument("--dump", action="store_true", help="打印错答的解析与谓词")
    args = parser.parse_args()
    _print_summary(audit(dump=args.dump))
