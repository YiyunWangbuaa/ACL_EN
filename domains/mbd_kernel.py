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
REASONS = frozenset({"不可注入", "稿件未改动", "被掩蔽"})
MATH_FORMATS = frozenset({"矩阵", "向量", "表达式", "公式"})


def outside() -> tuple[tuple[str, str], ...]:
    return ((RES, "残差没有注入器"),)


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
        return "无"
    return str(row.get("format") or "无")


def _kind(fmt: str) -> str:
    if fmt == "矩阵":
        return "matrix"
    if fmt == "向量":
        return "vector"
    if fmt in {"表达式", "公式"}:
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
    if fmt == "矩阵" and text.startswith("[["):
        return text.replace("1", "0", 1) if "1" in text else text[:-2] + ", 0]]"
    if fmt == "向量" and text.startswith("["):
        return "[0, 0, 0]" if text != "[0, 0, 0]" else "[1, 0, 0]"
    if text.startswith("-"):
        return text[1:].strip() or "0"
    return f"-({text})" if text else "0"


_SLOT_BODY = re.compile(r"\{ID:(\d+),.*?\}", re.S)
_AXIS_TOKEN = re.compile(r"^-?[XYZ][A-Za-z0-9_]*$")
_ABS_ROLE = re.compile(r"绝对(?:轴向|法向|侧向)正方向（[^）]+）")
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
        if clause in {"转换方向是", "RP1为", "所有坐标系的变换采用", "第一个是", "第二个是", "第三个是"}:
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
    sentence = re.sub(r"^首先确定", "", sentence).strip("，,、")
    if clause == "转换方向是":
        match = re.search(r"([\u4e00-\u9fff0-9]{2,16}到[\u4e00-\u9fff0-9]{2,16})的转换矩阵", sentence)
        if match:
            owner = match.group(1).split("首先确定")[-1]
            return owner + "的转换方向"
    if clause == "RP1为":
        match = re.search(r"将([\u4e00-\u9fff0-9]{2,16})的坐标轴", sentence)
        if match:
            return match.group(1) + "前置矩阵RP1"
        return "前置矩阵RP1"
    if clause in {"第一个是", "第二个是", "第三个是"}:
        return "欧拉角顺序" + clause
    if not sentence:
        return clause
    return sentence + clause


def _name_correspondences(text: str, found: dict[str, str]) -> None:
    """A、B、C 分别对应 three blanks. Each blank keeps its own absolute axis."""
    for marker in re.finditer("分别对应", text):
        left = text[max(0, marker.start() - 220):marker.start()]
        roles = _ABS_ROLE.findall(left)
        if len(roles) < 2:
            continue
        head = left[max(left.rfind("。"), left.rfind("\n")) + 1:]
        first = _ABS_ROLE.search(head)
        owner = head[:first.start()] if first else ""
        owner = re.sub(r"^[由于在把将和与及\s]+", "", owner).strip("，,、")
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
    text = re.sub(r"（(?:前方|后方|上方|下方|左侧|右侧|左方|右方)）", "", text)
    text = re.sub(r"^[0-9]+[.、]", "", text)
    text = text.strip("，,、；;：:。？?")
    options = [str(item) for item in (row.get("options") or [])]
    if options and set(options) <= {"正", "负"} and len(text) < 6:
        return "转向"
    if options and len(text) < 6 and len(options) == 2 and all(len(item) <= 8 for item in options):
        return f"{options[0]}或{options[1]}"
    if len(text) > 42:
        cut = len(text) - 42
        if cut > 0 and text[cut - 1:cut + 1] in {"相机", "目标", "图像", "同一"}:
            cut -= 1
        text = text[cut:]
        if not _cjk(text[0]):
            for index, char in enumerate(text):
                if char in "，、时" and index + 1 < len(text) and _cjk(text[index + 1]):
                    text = text[index + 1:]
                    break
    return text.strip("，,、；;：:")


def _judgment(row: dict) -> str:
    """Which kind of judgment the blank asks. The chosen option is not named."""
    options = [str(item) for item in (row.get("options") or [])]
    subject = _subject(row)
    if not options:
        return "判断失败"
    if set(options) <= {"正", "负", "相同", "相反"}:
        return "角度判断失败"
    blob = subject + "".join(options)
    if any(token in blob for token in ("坐标轴", "绝对法向", "绝对轴向", "绝对侧向", "正方向")):
        return "轴向判断失败"
    if "法向" in subject or "侧向" in subject or "轴向" in subject:
        return "轴向判断失败"
    if any(token in subject for token in ("偏", "转向", "右手", "迎角", "侧滑", "滚转", "俯仰", "方位")):
        return "角度判断失败"
    if options and all(_AXIS_TOKEN.fullmatch(item.strip()) for item in options):
        return "轴向判断失败"
    return "判断失败"


def _concept(view: DomainView, node: str, kind: str) -> str:
    row = view.question(node)
    subject = _subject(row)
    if not subject:
        raise RuntimeError(f"{view.name} ID{node} 没有可概括的题面。")
    if kind == "角度判断失败" and len(subject) < 4:
        subject = "转向"
    return f"{view.name}·{subject}·{kind}"


def _named(used: dict[str, int], klass: str, concept: str) -> str:
    base = f"{klass}:{concept}"
    used[base] = used.get(base, 0) + 1
    if used[base] == 1:
        return base
    return f"{base}（{used[base]}）"


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
            _named(used, RSN, _concept(view, node, "推理识别")),
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
            _named(used, ALG, _concept(view, node, "数理错误")),
            ALG,
            view.name,
            ((node, alt),),
            direct,
            potential,
        ))
    found = _resolve_duplicate_site(found)
    ids = [item.id for item in found]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"{view.name} 故障实例 id 重复")
    bad = sorted({item.klass for item in found} - ROW_CLASSES)
    if bad:
        raise RuntimeError(f"{view.name} 库外的类进了行：{bad}")
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
            found.append(InstanceResult(item, 0, "不可注入", ()))
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
        raise RuntimeError(f"{view.name} 的 det 没有盖住实例库")
    if RES in summary["counts"]:
        raise RuntimeError(f"{view.name} 把 RES 当成了行")
    if summary["counts"].get(DIR, 0) <= 0:
        raise RuntimeError(f"{view.name} 没有 DIR 直接识别行")
    if summary["counts"].get(EXE, 0) <= 0:
        raise RuntimeError(f"{view.name} 没有 EXE 执行行")
    if summary["counts"].get(ALG, 0) <= 0:
        raise RuntimeError(f"{view.name} 没有 ALG 数理行")
    for item in results:
        if item.det not in (0, 1):
            raise RuntimeError(f"{item.instance.id} det 非法")
        if item.det == 0 and item.reason not in REASONS:
            raise RuntimeError(f"{item.instance.id} det=0 原因 {item.reason!r}")
        if item.det == 1 and item.reason:
            raise RuntimeError(f"{item.instance.id} det=1 仍带原因")
        for mid in item.entries:
            if not mid:
                raise RuntimeError(f"{item.instance.id} 的支持集里有空测点")
    kinds = summary["measurements"]
    if kinds.get("node", 0) != len(view.node_ids) or kinds.get("format", 0) != 3:
        raise RuntimeError(f"{view.name} 测点层不齐：{dict(kinds)}")


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
        raise RuntimeError(f"{view.name} 支持层重跑哈希不一致")
    summary = summarize(view, first)
    _check_results(view, first, summary)
    from domains.conflict_diagnosis import remember

    remember(view, first)
    counts = summary["counts"]
    print(
        f"阶段4 二值支持：行 {len(first)}（"
        + " ".join(f"{key} {counts.get(key, 0)}" for key in (DIR, RSN, EXE, ALG))
        + f"），可达 {summary['det1']}，空可达 {summary['det0']}，"
        + f"DIR 模糊组 {summary['e1_blocks']}（最大 {summary['e1_max_block']}），"
        + f"全库模糊组 {summary['isolation_blocks']}，RES 不进矩阵。",
        flush=True,
    )
    return summary
