# -*- coding: utf-8 -*-
"""Shared fault catalog and binary D-matrix audit.

Recognition is one ability at two depths: DIR reads a stated mapping, RSN
infers the route the contract implies. EXE carries that contract out. ALG
miscalculates a quantity that is already written. RES has no injector.

A row is named by the judgment the blank asks, not by the wrong text written
into it. EXE is one row per parent; a child that reads that parent is
propagation. A singleton site keeps DIR when the blank is direct recognition,
otherwise EXE. A one-blank RSN is not a reasoned route and is dropped. A
larger RSN joint, an ALG on the same blank, and a blank that is only
reachable stay.

Domain content is supplied by a DomainView. Trajectory diagnosis is
domains.conflict_diagnosis. Repair search is domains.repair_search.
"""
from __future__ import annotations

from domains.fault_class import ALG, DIR, EXE, RES, RSN

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

FAULT_MODES = (DIR, RSN, EXE, ALG, RES)
ROW_CLASSES = frozenset({DIR, RSN, EXE, ALG})
REASONS = frozenset({"not injectable", "manuscript unchanged", "masked"})
MATH_FORMATS = frozenset({"matrix", "vector", "expression", "formula"})


def canon_format(fmt: str) -> str:
    """English format name. Legacy manuscript codes take the same branches."""
    key = str(fmt or "").strip()
    return {
        "matrix": "matrix",
        "vector": "vector",
        "expression": "expression",
        "formula": "formula",
        "none": "none",
        "choice": "choice",
        "\u77e9\u9635": "matrix",
        "\u5411\u91cf": "vector",
        "\u8868\u8fbe\u5f0f": "expression",
        "\u516c\u5f0f": "formula",
        "\u65e0": "none",
    }.get(key, key)


def outside() -> tuple[tuple[str, str], ...]:
    return ((RES, "residual has no injector"),)


def sort_key(node: str):
    text = str(node)
    prefix = text[0] if text[:1].isalpha() else ""
    rest = text[1:] if prefix else text
    try:
        return (prefix, int(rest))
    except ValueError:
        return (prefix, rest)


@dataclass(frozen=True)
class DomainView:
    name: str
    node_ids: tuple[str, ...]
    parents: Callable[[str], tuple[str, ...]]
    gold: Callable[[], dict[str, str]]
    question: Callable[[str], dict]
    answer_under: Callable[[str, dict[str, str]], str | None]
    equal: Callable[[str, str, str], bool]
    declaration_roots: frozenset[str]
    reference: Callable[[str, dict[str, str]], object] | None = None


@dataclass(frozen=True)
class FaultInstance:
    id: str
    klass: str
    scope: str
    injector: tuple[tuple[str, str], ...]
    direct: frozenset[str]
    potential: frozenset[str]


@dataclass(frozen=True)
class Measurement:
    id: str
    node: str
    kind: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class InstanceResult:
    instance: FaultInstance
    det: int
    reason: str
    entries: tuple[str, ...]


def _fmt(view: DomainView, node: str) -> str:
    row = view.question(node)
    if row.get("options"):
        return "none"
    return canon_format(str(row.get("format") or "none"))


def _kind(fmt: str) -> str:
    fmt = canon_format(fmt)
    if fmt == "matrix":
        return "matrix"
    if fmt == "vector":
        return "vector"
    if fmt in {"expression", "formula"}:
        return "expression"
    return "scalar"


def catalog(view: DomainView) -> tuple[Measurement, ...]:
    found: list[Measurement] = []
    for node in sorted(view.node_ids, key=sort_key):
        kind = _kind(_fmt(view, node))
        found.append(Measurement(f"ID{node}", node, "node", ()))
        if kind == "matrix":
            for row in range(3):
                for column in range(3):
                    found.append(Measurement(
                        f"ID{node}.cell({row},{column})", node, "field", ("cell", str(row), str(column)),
                    ))
        elif kind == "vector":
            for index in range(3):
                found.append(Measurement(
                    f"ID{node}.component({index})", node, "field", ("component", str(index)),
                ))
        elif kind == "expression":
            for name, layer in (
                ("coefficient", "structure"),
                ("outer_sign", "structure"),
                ("symbol_set", "field"),
            ):
                found.append(Measurement(f"ID{node}.{name}", node, layer, (name,)))
    found.extend((
        Measurement("format_valid", "", "format", ("format_valid",)),
        Measurement("equivalence_class", "", "format", ("equivalence_class",)),
        Measurement("option_ambiguity", "", "format", ("option_ambiguity",)),
    ))
    return tuple(found)


def _by_node(view: DomainView) -> dict[str, tuple[Measurement, ...]]:
    grouped: dict[str, list[Measurement]] = {}
    for item in catalog(view):
        if item.node:
            grouped.setdefault(item.node, []).append(item)
    return {node: tuple(items) for node, items in grouped.items()}


def _children(view: DomainView) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for node in view.node_ids:
        for parent in view.parents(node):
            grouped.setdefault(parent, []).append(node)
    return {parent: tuple(sorted(nodes, key=sort_key)) for parent, nodes in grouped.items()}


def _descendants(view: DomainView, roots: tuple[str, ...]) -> tuple[str, ...]:
    kids = _children(view)
    seen: set[str] = set()
    pending = list(roots)
    while pending:
        node = pending.pop(0)
        for child in kids.get(node, ()):
            if child not in seen:
                seen.add(child)
                pending.append(child)
    return tuple(sorted(seen, key=sort_key))


def _sibling(view: DomainView, node: str) -> str | None:
    gold = view.gold()[node]
    options = list(view.question(node).get("options") or [])
    for option in options:
        if not view.equal(option, gold, node):
            return option
    if options:
        return None
    slipped = _slip(gold, _fmt(view, node))
    if view.equal(slipped, gold, node):
        return None
    return slipped


def _slip(gold: str, fmt: str) -> str:
    text = str(gold).strip()
    fmt = canon_format(fmt)
    if fmt == "matrix" and text.startswith("[["):
        return text.replace("1", "0", 1) if "1" in text else text[:-2] + ", 0]]"
    if fmt == "vector" and text.startswith("["):
        return "[0, 0, 0]" if text != "[0, 0, 0]" else "[1, 0, 0]"
    if text.startswith("-"):
        return text[1:].strip() or "0"
    return f"-({text})" if text else "0"


_SLOT_BODY = re.compile(r"\{ID:(\d+),.*?\}", re.S)
_AXIS_TOKEN = re.compile(r"^-?[XYZ][A-Za-z0-9_]*$")
_ABS_ROLE = re.compile(
    "\u7edd\u5bf9(?:\u8f74\u5411|\u6cd5\u5411|\u4fa7\u5411)\u6b63\u65b9\u5411\uff08[^\uff09]+\uff09"
)
_CLAUSE_DIRECTION = "\u8f6c\u6362\u65b9\u5411\u662f"
_CLAUSE_RP1 = "RP1\u4e3a"
_CLAUSE_FRAMES = "\u6240\u6709\u5750\u6807\u7cfb\u7684\u53d8\u6362\u91c7\u7528"
_CLAUSE_FIRST = "\u7b2c\u4e00\u4e2a\u662f"
_CLAUSE_SECOND = "\u7b2c\u4e8c\u4e2a\u662f"
_CLAUSE_THIRD = "\u7b2c\u4e09\u4e2a\u662f"
_SHORT_CLAUSES = frozenset({
    _CLAUSE_DIRECTION, _CLAUSE_RP1, _CLAUSE_FRAMES,
    _CLAUSE_FIRST, _CLAUSE_SECOND, _CLAUSE_THIRD,
})
_EULER_CLAUSES = frozenset({_CLAUSE_FIRST, _CLAUSE_SECOND, _CLAUSE_THIRD})
_SIGN_OPTIONS = frozenset({"positive", "negative", "\u6b63", "\u8d1f"})
_ANGLE_OPTIONS = _SIGN_OPTIONS | frozenset({"same", "opposite", "\u76f8\u540c", "\u76f8\u53cd"})
_SLOT_ID = re.compile(r"\{ID:(\d+),")


def question_clauses(text: str) -> dict[str, str]:
    """The words that ask the blank, without the option list or the gold answer."""
    slots = list(_SLOT_BODY.finditer(text))
    found = {}
    order = []
    for index, match in enumerate(slots):
        prev = slots[index - 1].end() if index else 0
        left = text[prev:match.start()]
        for sep in ("。", "\n", "；"):
            cut = left.rfind(sep)
            if cut != -1:
                left = left[cut + 1:]
        right = text[match.end():match.end() + 12]
        right = re.split(r"[。\n；，,{]", right, maxsplit=1)[0]
        clause = re.sub(r"\s+", "", left + right)
        clause = re.sub(r"^[0-9]+[.、]", "", clause)
        clause = clause.strip("，,、；;：:。？?")
        if clause in _SHORT_CLAUSES:
            clause = _with_previous_sentence(text, match.start(), clause)
        found[match.group(1)] = clause
        order.append(match.group(1))
    _name_correspondences(text, found)
    carried = ""
    for node in order:
        if found[node]:
            carried = found[node]
        elif carried:
            found[node] = carried
    return found


def _with_previous_sentence(text: str, start: int, clause: str) -> str:
    dot = text.rfind("。", 0, start)
    if dot <= 0:
        return clause
    earlier = text.rfind("。", 0, dot)
    sentence = text[earlier + 1:dot]
    sentence = _SLOT_BODY.sub("", sentence)
    sentence = sentence.replace("#", "")
    sentence = re.sub(r"\s+", "", sentence).split("。")[-1]
    sentence = re.sub(r"^[0-9]+[.、]", "", sentence)
    sentence = re.sub(r"^\u9996\u5148\u786e\u5b9a", "", sentence).strip("，,、")
    if clause == _CLAUSE_DIRECTION:
        match = re.search(
            r"([\u4e00-\u9fff0-9]{2,16}\u5230[\u4e00-\u9fff0-9]{2,16})\u7684\u8f6c\u6362\u77e9\u9635",
            sentence,
        )
        if match:
            owner = match.group(1).split("\u9996\u5148\u786e\u5b9a")[-1]
            return owner + " conversion direction"
    if clause == _CLAUSE_RP1:
        match = re.search(r"\u5c06([\u4e00-\u9fff0-9]{2,16})\u7684\u5750\u6807\u8f74", sentence)
        if match:
            return match.group(1) + " pre-rotation RP1"
        return "pre-rotation RP1"
    if clause in _EULER_CLAUSES:
        return "Euler angle order " + clause
    if not sentence:
        return clause
    return sentence + clause


def _name_correspondences(text: str, found: dict[str, str]) -> None:
    """A, B, and C each correspond to one blank. Each blank keeps its own absolute axis."""
    for marker in re.finditer("\u5206\u522b\u5bf9\u5e94", text):
        left = text[max(0, marker.start() - 220):marker.start()]
        roles = _ABS_ROLE.findall(left)
        if len(roles) < 2:
            continue
        head = left[max(left.rfind("。"), left.rfind("\n")) + 1:]
        first = _ABS_ROLE.search(head)
        owner = head[:first.start()] if first else ""
        owner = re.sub(r"^[\u7531\u4e8e\u5728\u628a\u5c06\u548c\u4e0e\u53ca\s]+", "", owner).strip("，,、")
        right = text[marker.end():marker.end() + 4000]
        ids = []
        cursor = 0
        while len(ids) < len(roles):
            match = _SLOT_ID.search(right, cursor)
            if match is None:
                break
            between = right[cursor:match.start()]
            if ids and re.search(r"[。\n；]", between):
                break
            if ids and not re.fullmatch(r"[\s、]*", between):
                break
            ids.append(match.group(1))
            end = right.find("}", match.end())
            if end < 0:
                break
            cursor = end + 1
        for role, node in zip(roles, ids):
            found[node] = f"{owner}{role}"


def _cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _subject(row: dict) -> str:
    text = re.sub(r"\s+", "", str(row.get("stem") or ""))
    text = text.replace("#", "")
    text = re.sub(
        r"\uff08(?:\u524d\u65b9|\u540e\u65b9|\u4e0a\u65b9|\u4e0b\u65b9|\u5de6\u4fa7|\u53f3\u4fa7|\u5de6\u65b9|\u53f3\u65b9)\uff09",
        "",
        text,
    )
    text = re.sub(r"^[0-9]+[.、]", "", text)
    text = text.strip("，,、；;：:。？?")
    options = [str(item) for item in (row.get("options") or [])]
    if options and set(options) <= _SIGN_OPTIONS and len(text) < 6:
        return "sense"
    if options and len(text) < 6 and len(options) == 2 and all(len(item) <= 8 for item in options):
        return f"{options[0]} or {options[1]}"
    if len(text) > 42:
        cut = len(text) - 42
        if cut > 0 and text[cut - 1:cut + 1] in {"\u76f8\u673a", "\u76ee\u6807", "\u56fe\u50cf", "\u540c\u4e00"}:
            cut -= 1
        text = text[cut:]
        if not _cjk(text[0]):
            for index, char in enumerate(text):
                if char in "，、\u65f6" and index + 1 < len(text) and _cjk(text[index + 1]):
                    text = text[index + 1:]
                    break
    return text.strip("，,、；;：:")


def _judgment(row: dict) -> str:
    """Which kind of judgment the blank asks. The chosen option is not named."""
    options = [str(item) for item in (row.get("options") or [])]
    subject = _subject(row)
    if not options:
        return "judgment failed"
    if set(options) <= _ANGLE_OPTIONS:
        return "angle judgment failed"
    blob = subject + "".join(options)
    if any(token in blob for token in (
        "\u5750\u6807\u8f74",
        "\u7edd\u5bf9\u6cd5\u5411",
        "\u7edd\u5bf9\u8f74\u5411",
        "\u7edd\u5bf9\u4fa7\u5411",
        "\u6b63\u65b9\u5411",
    )):
        return "axis judgment failed"
    if any(token in subject for token in ("\u6cd5\u5411", "\u4fa7\u5411", "\u8f74\u5411")):
        return "axis judgment failed"
    if any(token in subject for token in (
        "\u504f", "\u8f6c\u5411", "\u53f3\u624b", "\u8fce\u89d2", "\u4fa7\u6ed1",
        "\u6eda\u8f6c", "\u4fef\u4ef0", "\u65b9\u4f4d",
    )):
        return "angle judgment failed"
    if options and all(_AXIS_TOKEN.fullmatch(item.strip()) for item in options):
        return "axis judgment failed"
    return "judgment failed"


def _concept(view: DomainView, node: str, kind: str) -> str:
    row = view.question(node)
    subject = _subject(row)
    if not subject:
        raise RuntimeError(f"{view.name} ID{node} has no stem that can be summarized.")
    if kind == "angle judgment failed" and len(subject) < 4:
        subject = "sense"
    return f"{view.name}·{subject}·{kind}"


def _named(used: dict[str, int], klass: str, concept: str) -> str:
    base = f"{klass}:{concept}"
    used[base] = used.get(base, 0) + 1
    if used[base] == 1:
        return base
    return f"{base} ({used[base]})"


def _reach(view: DomainView, injected: tuple[str, ...]) -> tuple[frozenset[str], frozenset[str]]:
    columns = _by_node(view)
    nodes = tuple(dict.fromkeys(injected + _descendants(view, injected)))
    direct = frozenset(f"ID{node}" for node in injected)
    potential: set[str] = set()
    for node in nodes:
        potential.update(item.id for item in columns.get(node, ()))
    return direct, frozenset(potential)


def instances(view: DomainView) -> tuple[FaultInstance, ...]:
    """One concept-level row per fault. The id names the judgment, not the wrong text.

    EXE is one row per parent blank. A child that reads the parent is propagation.
    The injector still holds one witness text so the row can be recomputed.
    A singleton site keeps DIR when that row exists, otherwise EXE. A one-blank
    RSN on that site is dropped.
    """
    gold = view.gold()
    found: list[FaultInstance] = []
    used: dict[str, int] = {}
    for node in sorted(view.declaration_roots, key=sort_key):
        if not view.question(node).get("options"):
            continue
        alt = _sibling(view, node)
        if alt is None:
            continue
        direct, potential = _reach(view, (node,))
        found.append(FaultInstance(
            _named(used, DIR, _concept(view, node, _judgment(view.question(node)))),
            DIR,
            view.name,
            ((node, alt),),
            direct,
            potential,
        ))
    parents_of: dict[str, str] = {}
    joints: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    for node in sorted(view.node_ids, key=sort_key):
        parents = view.parents(node)
        if not parents:
            continue
        slots = []
        for parent in parents:
            alt = _sibling(view, parent)
            if alt is None:
                continue
            slots.append((parent, alt))
            parents_of.setdefault(parent, alt)
        choice_slots = tuple(item for item in slots if view.question(item[0]).get("options"))
        if choice_slots:
            joints.append((node, choice_slots))
    for parent in sorted(parents_of, key=sort_key):
        direct, potential = _reach(view, (parent,))
        found.append(FaultInstance(
            _named(used, EXE, _concept(view, parent, _judgment(view.question(parent)))),
            EXE,
            view.name,
            ((parent, parents_of[parent]),),
            direct,
            potential,
        ))
    for node, choice_slots in joints:
        direct, potential = _reach(view, tuple(slot for slot, _alt in choice_slots))
        found.append(FaultInstance(
            _named(used, RSN, _concept(view, node, "reasoned recognition")),
            RSN,
            view.name,
            choice_slots,
            direct,
            potential,
        ))
    for node in sorted(view.node_ids, key=sort_key):
        if view.question(node).get("options"):
            continue
        if _fmt(view, node) not in MATH_FORMATS:
            continue
        alt = _slip(gold[node], _fmt(view, node))
        if view.equal(alt, gold[node], node):
            continue
        direct, potential = _reach(view, (node,))
        found.append(FaultInstance(
            _named(used, ALG, _concept(view, node, "algebraic error")),
            ALG,
            view.name,
            ((node, alt),),
            direct,
            potential,
        ))
    found = _resolve_duplicate_site(found)
    ids = [item.id for item in found]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{view.name} fault instance ids are duplicated")
    bad = sorted({item.klass for item in found} - ROW_CLASSES)
    if bad:
        raise RuntimeError(f"{view.name} has a class outside the library in a row: {bad}")
    return tuple(found)


def _resolve_duplicate_site(rows: list[FaultInstance]) -> list[FaultInstance]:
    """One singleton site keeps the class that blank actually is.

    The comparison is the set of blanks, not the written text. A direct
    recognition row (DIR) owns a declaration root, so the EXE copy and any
    one-blank RSN are dropped. A derived parent has no DIR: the EXE stays and
    the one-blank RSN is dropped. A reasoning joint writes at least two
    blanks, so it does not share a set with a parent EXE. An ALG on the same
    blank is not compared.
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
    return kept


_EXTRA_SYMBOLS = (
    ("μ", "mu"),
    ("ρ", "rho"),
    ("Λ", "Lambda"),
    ("Γ", "Gamma"),
)
_SYMPY_FUNCTIONS = frozenset({
    "sin", "cos", "tan", "asin", "atan", "atan2", "sqrt", "exp", "log", "Abs", "pi",
})


def _prepare_sympy(text: str) -> str:
    """Degrees and Greek letters that the shared scorer does not rename."""
    value = str(text or "")
    value = re.sub(r"(\d)\s*°", r"\1*pi/180", value)
    for source, target in _EXTRA_SYMBOLS:
        value = value.replace(source, target)
    return value


def _literal_matrix(text: str):
    try:
        data = ast.literal_eval(str(text).replace("，", ","))
    except (SyntaxError, ValueError):
        return None
    if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
        return None
    return [[str(cell) for cell in row] for row in data]


def _symbol_locals(text: str):
    """Every written name is a symbol. SymPy's own S, Lambda and Gamma are not."""
    from cloze_score import _sympy_modules, _to_sympy_text

    sympy, locals_map = _sympy_modules()
    locals_map = dict(locals_map)
    prepared = _to_sympy_text(_prepare_sympy(text))
    for name in set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", prepared)):
        if name in _SYMPY_FUNCTIONS:
            continue
        locals_map[name] = sympy.Symbol(name)
    return sympy, locals_map


@lru_cache(maxsize=4096)
def _cached_matrix(text: str):
    from cloze_score import _parse_expr

    sympy, locals_map = _symbol_locals(text)
    try:
        parsed = _parse_expr(_prepare_sympy(text), sympy, locals_map)
    except Exception:
        return None
    if not isinstance(parsed, sympy.MatrixBase) or parsed.shape != (3, 3):
        return None
    return tuple(
        tuple(sympy.sstr(parsed[row, column], order="lex") for column in range(3))
        for row in range(3)
    )


@lru_cache(maxsize=4096)
def _cached_vector(text: str):
    from cloze_score import _parse_expr

    sympy, locals_map = _symbol_locals(text)
    try:
        parsed = _parse_expr(_prepare_sympy(text), sympy, locals_map)
    except Exception:
        return None
    if not isinstance(parsed, sympy.MatrixBase):
        return None
    if parsed.shape == (1, 3):
        entries = tuple(parsed[0, index] for index in range(3))
    elif parsed.shape == (3, 1):
        entries = tuple(parsed[index, 0] for index in range(3))
    else:
        return None
    return tuple(sympy.sstr(item, order="lex") for item in entries)


def _parse_matrix(text: str):
    cached = _cached_matrix(str(text or "").strip())
    if cached is not None:
        return [list(row) for row in cached]
    return _literal_matrix(text)


def _parse_vector(text: str):
    cached = _cached_vector(str(text or "").strip())
    if cached is not None:
        return list(cached)
    data = _literal_matrix("[" + str(text).strip().strip("[]") + "]")
    if data and len(data) == 1:
        return data[0]
    literal = _literal_matrix(text)
    if literal and len(literal) == 1:
        return literal[0]
    return None


def _project(view: DomainView, node: str, text: str) -> dict[str, str]:
    found = {f"ID{node}": str(text or "")}
    kind = _kind(_fmt(view, node))
    if kind == "matrix":
        matrix = _parse_matrix(text)
        for row in range(3):
            for column in range(3):
                value = ""
                if matrix and row < len(matrix) and column < len(matrix[row]):
                    value = str(matrix[row][column])
                found[f"ID{node}.cell({row},{column})"] = value
    elif kind == "vector":
        vector = _parse_vector(text)
        for index in range(3):
            value = str(vector[index]) if vector and index < len(vector) else ""
            found[f"ID{node}.component({index})"] = value
    elif kind == "expression":
        raw = str(text or "")
        found[f"ID{node}.outer_sign"] = "-" if raw.strip().startswith("-") else "+"
        names = re.findall(r"[A-Za-z_α-ωΑ-Ω][A-Za-z0-9_α-ω]*", raw)
        found[f"ID{node}.symbol_set"] = ",".join(sorted(set(names)))
        found[f"ID{node}.coefficient"] = raw.strip()[:48]
    return found


def evaluate(view: DomainView) -> tuple[InstanceResult, ...]:
    """Binary support: a row is detected when its graph reach is non-empty.

    The support columns are FaultInstance.potential. This does not inject a
    witness or store the value a calculator would write.
    """
    found = []
    for item in instances(view):
        support = tuple(sorted(item.potential))
        if support:
            found.append(InstanceResult(item, 1, "", support))
        else:
            found.append(InstanceResult(item, 0, "not injectable", ()))
    found.sort(key=lambda item: item.instance.id)
    return tuple(found)


def _blocks(rows: list[InstanceResult]) -> tuple[tuple[str, ...], ...]:
    grouped: dict[tuple, list[str]] = defaultdict(list)
    for item in rows:
        signature = item.entries
        grouped[signature].append(item.instance.id)
    return tuple(tuple(sorted(ids)) for ids in grouped.values())


def summarize(view: DomainView, results: tuple[InstanceResult, ...]) -> dict:
    counts = Counter(item.instance.klass for item in results)
    det1 = [item for item in results if item.det == 1]
    det0 = [item for item in results if item.det == 0]
    e1 = [item for item in det1 if item.instance.klass == DIR]
    blocks = _blocks(e1)
    largest = max(blocks, key=len) if blocks else ()
    columns = catalog(view)
    return {
        "counts": dict(counts),
        "det1": len(det1),
        "det0": dict(Counter(item.reason for item in det0)),
        "e1_blocks": len(blocks),
        "e1_max_block": len(largest),
        "e1_singletons": sum(len(block) == 1 for block in blocks),
        "isolation_blocks": len(_blocks(det1)),
        "measurements": Counter(item.kind for item in columns),
        "outside": outside(),
    }


def _check_results(view: DomainView, results: tuple[InstanceResult, ...], summary: dict) -> None:
    library = {item.id for item in instances(view)}
    seen = {item.instance.id for item in results}
    if seen != library:
        raise RuntimeError(f"{view.name} det does not cover the instance library")
    if RES in summary["counts"]:
        raise RuntimeError(f"{view.name} treated RES as a row")
    if summary["counts"].get(DIR, 0) <= 0:
        raise RuntimeError(f"{view.name} has no DIR direct-recognition row")
    if summary["counts"].get(EXE, 0) <= 0:
        raise RuntimeError(f"{view.name} has no EXE execution row")
    if summary["counts"].get(ALG, 0) <= 0:
        raise RuntimeError(f"{view.name} has no ALG algebra row")
    for item in results:
        if item.det not in (0, 1):
            raise RuntimeError(f"{item.instance.id} det is illegal")
        if item.det == 0 and item.reason not in REASONS:
            raise RuntimeError(f"{item.instance.id} det=0 reason {item.reason!r}")
        if item.det == 1 and item.reason:
            raise RuntimeError(f"{item.instance.id} det=1 still carries a reason")
        for mid in item.entries:
            if not mid:
                raise RuntimeError(f"{item.instance.id} support set contains an empty measurement")
    kinds = summary["measurements"]
    if kinds.get("node", 0) != len(view.node_ids) or kinds.get("format", 0) != 3:
        raise RuntimeError(f"{view.name} measurement layers are uneven: {dict(kinds)}")


def _digest(view: DomainView, results: tuple[InstanceResult, ...]) -> str:
    payload = [
        {
            "id": item.instance.id,
            "det": item.det,
            "reason": item.reason,
            "entries": list(item.entries),
        }
        for item in results
    ]
    raw = json.dumps({"domain": view.name, "rows": payload}, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


def audit_stage4(view: DomainView) -> dict:
    first = evaluate(view)
    second = evaluate(view)
    if _digest(view, first) != _digest(view, second):
        raise RuntimeError(f"{view.name} support-layer rerun hash does not match")
    summary = summarize(view, first)
    _check_results(view, first, summary)
    from domains.conflict_diagnosis import remember

    remember(view, first)
    counts = summary["counts"]
    print(
        f"Stage 4 binary support: rows {len(first)} ("
        + " ".join(f"{key} {counts.get(key, 0)}" for key in (DIR, RSN, EXE, ALG))
        + f"), reachable {summary['det1']}, empty reach {summary['det0']}, "
        + f"DIR ambiguity groups {summary['e1_blocks']} (max {summary['e1_max_block']}), "
        + f"library ambiguity groups {summary['isolation_blocks']}, RES is not in the matrix.",
        flush=True,
    )
    return summary
