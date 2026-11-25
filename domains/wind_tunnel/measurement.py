# -*- coding: utf-8 -*-
"""Layered measurements for one wind-tunnel answer.

Four layers, and only where the layer can actually be read:

* node — one column per public blank (194)
* field — nine matrix cells, three vector components, and the three
  expression fields that are not structural
* structure — coefficient, lever arm, and outer sign, read off the
  written expression before any canonical rewrite
* format — three manuscript-level probes (valid, equivalent-but-rewritten,
  option ambiguity); they are not fault rows

Field values come from the written text. Sympy decides whether two whole
answers are the same value; it does not erase an outer-sign move that
leaves the expanded polynomial unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from cloze_score import normalize, sympy_equal
from domains.wind_tunnel.answer_form import parse_answer
from domains.wind_tunnel.error_graph import NODE_BY_ID

_EXPRESSION_FIELDS = (
    "coefficient",
    "lever_arm",
    "outer_sign",
    "inner_signs",
    "term_set",
    "symbol_set",
)
_STRUCTURE_FIELDS = frozenset({"coefficient", "lever_arm", "outer_sign"})
_ORDER_FIELDS = frozenset({"inner_signs", "term_set", "symbol_set"})
_FUNCTIONS = frozenset({"sin", "cos", "tan", "asin", "atan", "arcsin", "arctan"})
_IDENT = re.compile(r"[^\W\d_]\w*", re.UNICODE)
_COEF = re.compile(r"([^\W\d_]\w*)\s*[*·]")
_LEVER = re.compile(
    r"([^\W\d_]\w*)\s*[*·]\s*(\([^)]*\)|[^\W\d_]\w*)\s*/\s*([^\W\d_]\w*)"
)


@dataclass(frozen=True)
class Measurement:
    id: str
    node: str
    kind: str
    path: tuple[str, ...]
    value_type: str


def node_id_of(measurement_id: str) -> str:
    """Public blank id of a node, field, or structure column. Empty for format."""
    if not measurement_id.startswith("ID"):
        return ""
    return measurement_id[2:].split(".", 1)[0]


def catalog() -> tuple[Measurement, ...]:
    """Every column the instrument can read. Stable order."""
    found: list[Measurement] = []
    for node_id in sorted(NODE_BY_ID, key=int):
        kind = NODE_BY_ID[node_id].answer_type
        found.append(Measurement(f"ID{node_id}", node_id, "node", (), kind))
        if kind == "matrix":
            for row in range(3):
                for column in range(3):
                    found.append(Measurement(
                        f"ID{node_id}.cell({row},{column})",
                        node_id, "field", ("cell", str(row), str(column)), "expr",
                    ))
        elif kind == "vector":
            for index in range(3):
                found.append(Measurement(
                    f"ID{node_id}.component({index})",
                    node_id, "field", ("component", str(index)), "expr",
                ))
        elif kind == "expression":
            for name in _EXPRESSION_FIELDS:
                layer = "structure" if name in _STRUCTURE_FIELDS else "field"
                value_type = "sign" if name == "outer_sign" else "expr"
                found.append(Measurement(
                    f"ID{node_id}.{name}", node_id, layer, (name,), value_type,
                ))
    found.extend((
        Measurement("format_valid", "", "format", ("format_valid",), "bool"),
        Measurement("equivalence_class", "", "format", ("equivalence_class",), "bool"),
        Measurement("option_ambiguity", "", "format", ("option_ambiguity",), "bool"),
    ))
    return tuple(found)


def by_node() -> dict[str, tuple[Measurement, ...]]:
    """Node, field, and structure columns of one blank. Format is separate."""
    grouped: dict[str, list[Measurement]] = {}
    for item in catalog():
        if item.node:
            grouped.setdefault(item.node, []).append(item)
    return {node_id: tuple(items) for node_id, items in grouped.items()}


def project(node_id: str, text: str) -> dict[str, str]:
    """Column id to the value read off this written answer."""
    kind = NODE_BY_ID[node_id].answer_type
    found = {f"ID{node_id}": str(text or "")}
    if kind == "matrix":
        found.update(_matrix_cells(node_id, text))
    elif kind == "vector":
        found.update(_vector_components(node_id, text))
    elif kind == "expression":
        found.update(_expression_fields(node_id, text))
    return found


def differences(node_id: str, predicted: str, gold: str) -> tuple[str, ...]:
    """Measurement ids whose text differs from gold.

    An empty prediction against a real gold answer marks the node and every
    nonempty field. A whole-answer match still reports a structure-field move,
    so an outer sign that trades places with the lever arm stays visible.
    """
    kind = NODE_BY_ID[node_id].answer_type
    predicted = str(predicted or "")
    gold = str(gold or "")
    if not predicted.strip():
        if not gold.strip():
            return ()
        found = {f"ID{node_id}"}
        for mid, value in project(node_id, gold).items():
            if mid != f"ID{node_id}" and value:
                found.add(mid)
        return tuple(sorted(found))
    same = _equivalent(kind, predicted, gold)
    found: set[str] = set()
    if not same:
        found.add(f"ID{node_id}")
    pred = project(node_id, predicted)
    base = project(node_id, gold)
    for mid, value in pred.items():
        if mid == f"ID{node_id}":
            continue
        name = mid.split(".", 1)[1]
        if same and _suppressed_when_same(name):
            continue
        if value != base.get(mid, ""):
            found.add(mid)
    return tuple(sorted(found))


def check_catalog() -> dict[str, int]:
    """S1 counts. Raises when a layer is missing or the outer-sign case collapses."""
    rows = catalog()
    counts = {
        "node": sum(item.kind == "node" for item in rows),
        "matrix_cells": sum(item.path[:1] == ("cell",) for item in rows),
        "vector_components": sum(item.path[:1] == ("component",) for item in rows),
        "expression_fields": sum(
            item.kind in {"field", "structure"} and item.node and NODE_BY_ID[item.node].answer_type == "expression"
            for item in rows
        ),
        "structure": sum(item.kind == "structure" for item in rows),
        "format": sum(item.kind == "format" for item in rows),
    }
    types = {}
    for node in NODE_BY_ID.values():
        types[node.answer_type] = types.get(node.answer_type, 0) + 1
    if counts["node"] != 207 or types.get("matrix") != 12 or types.get("vector") != 5 or types.get("expression") != 27:
        raise RuntimeError(f"测点节点数不对：{counts} / {types}")
    if counts["matrix_cells"] != 12 * 9 or counts["vector_components"] != 5 * 3:
        raise RuntimeError(f"格子或分量测点数不对：{counts}")
    if counts["expression_fields"] != 27 * 6 or counts["structure"] != 27 * 3 or counts["format"] != 3:
        raise RuntimeError(f"表达式或格式测点数不对：{counts}")
    seen = [item.id for item in rows]
    if len(seen) != len(set(seen)):
        raise RuntimeError("测点 id 重复。")
    outer = differences("198", "-CN·(Xr - Xg)/L_RT", "CN·(Xg - Xr)/L_RT")
    if "ID198.outer_sign" not in outer:
        raise RuntimeError(f"外层符号移动应留下 outer_sign，现在是 {outer}")
    same = differences("198", "CN·Zg/L_RT", "CN·Zg/L_RT")
    if same:
        raise RuntimeError(f"相同力臂式不应有差异：{same}")
    return counts


def _suppressed_when_same(name: str) -> bool:
    """Term order and cell text are M1 once the whole answer matches."""
    if name in _ORDER_FIELDS:
        return True
    return name.startswith("cell(") or name.startswith("component(")


def _matrix_cells(node_id: str, text: str) -> dict[str, str]:
    found = {f"ID{node_id}.cell({row},{column})": "" for row in range(3) for column in range(3)}
    form = parse_answer(text, "matrix")
    if form.status != "ok":
        return found
    for (row, column), value in form.cells:
        found[f"ID{node_id}.cell({row},{column})"] = str(value)
    return found


def _vector_components(node_id: str, text: str) -> dict[str, str]:
    found = {f"ID{node_id}.component({index})": "" for index in range(3)}
    form = parse_answer(text, "vector")
    if form.status != "ok":
        return found
    for (index,), value in form.cells:
        found[f"ID{node_id}.component({index[0] if isinstance(index, tuple) else index})"] = str(value)
    return found


def _expression_fields(node_id: str, text: str) -> dict[str, str]:
    cleaned = _clean(text)
    body = cleaned[1:].strip() if cleaned.startswith("-") else cleaned
    arm = ""
    match = _LEVER.search(cleaned.lstrip("-").lstrip())
    if match:
        arm = match.group(2)
        if arm.startswith("(") and arm.endswith(")"):
            arm = arm[1:-1].strip()
    names = [item for item in _IDENT.findall(cleaned) if item not in _FUNCTIONS]
    terms = _split_terms(body)
    return {
        f"ID{node_id}.coefficient": _coefficient(cleaned),
        f"ID{node_id}.lever_arm": arm,
        f"ID{node_id}.outer_sign": "负" if cleaned.startswith("-") else "正",
        f"ID{node_id}.inner_signs": ",".join(ch for ch in body if ch in "+-"),
        f"ID{node_id}.term_set": " | ".join(term for _sign, term in terms),
        f"ID{node_id}.symbol_set": ",".join(sorted(set(names))),
    }


def _coefficient(cleaned: str) -> str:
    for match in _COEF.finditer(cleaned):
        name = match.group(1)
        if name not in _FUNCTIONS:
            return name
    return "1"


def _split_terms(text: str) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    depth = 0
    sign = "+"
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and char in "+-" and index > 0 and text[index - 1] != "*":
            chunk = text[start:index].strip()
            if chunk:
                terms.append((sign, chunk))
            sign = char
            start = index + 1
    chunk = text[start:].strip()
    if chunk:
        terms.append((sign, chunk))
    return terms


def _clean(text: str) -> str:
    value = str(text or "").replace("·", "*").replace("×", "*").replace("⋅", "*")
    value = value.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", " ", value).strip()


def _equivalent(kind: str, left: str, right: str) -> bool:
    if normalize(left) == normalize(right):
        return True
    if kind not in {"matrix", "vector", "expression"} or not left.strip() or not right.strip():
        return False
    try:
        return sympy_equal(left, right)
    except Exception:
        return False

