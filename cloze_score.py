# -*- coding: utf-8 -*-
"""Strict and normalized cloze scores.

Strict is stripped equality. Normalized also accepts spacing, unicode
punctuation, a leading 答案 prefix, and LaTeX markup. Expressions, matrices,
vectors, distances and moment arms may also match by sympy equivalence.
Choice blanks stay on the option text and are not passed to sympy.
"""

from __future__ import annotations

import random
import re

SYMBOLIC = {"矩阵", "向量", "表达式", "距离", "arctan", "arcsin", "力臂"}
_AXIS = r"XFD|YFD|ZFD|XTZ|YTZ|ZTZ|XTP|YTP|ZYP"
_AXIS_TOKEN = re.compile(rf"-?(?:{_AXIS})")
_AXIS_NAME = re.compile(rf"-?({_AXIS})")
_OPPOSITES = (
    {"正", "负"},
    {"正", "逆"},
    {"相同", "相反"},
    {"顺时针", "逆时针"},
    {"上方", "下方"},
    {"左侧", "右侧"},
    {"前方", "后方"},
)
_GREEK = (
    ("Δα_e", "Dalpha_e"),
    ("Δβ_e", "Dbeta_e"),
    ("Δγ_e", "Dgamma_e"),
    ("Δα_V", "Dalpha_V"),
    ("Δβ_V", "Dbeta_V"),
    ("α_M", "alpha_M"),
    ("β_M", "beta_M"),
    ("γ_M", "gamma_M"),
    ("α_e", "alpha_e"),
    ("β_e", "beta_e"),
    ("γ_e", "gamma_e"),
    ("γ_tp", "gamma_tp"),
    ("α_t", "alpha_t"),
    ("β_t", "beta_t"),
    ("α_V", "alpha_V"),
    ("β_V", "beta_V"),
    ("α", "alpha"),
    ("β", "beta"),
    ("γ", "gamma"),
    ("θ_M", "theta_M"),
    ("φ_M", "phi_M"),
    ("ψ_M", "psi_M"),
    ("θ", "theta"),
    ("φ", "phi"),
    ("ψ", "psi"),
    ("ω", "omega"),
    ("λ", "lam"),
)
_LATEX_FRAC = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_LATEX_MATRIX = re.compile(
    r"\\begin\{(?:array(?:\{[^{}]*\})?|(?:b|p|B|P|v|V)?matrix)\}(.*?)\\end\{(?:array|(?:b|p|B|P|v|V)?matrix)\}",
    re.S,
)
_LATEX_GREEK = (
    ("\\varphi", "φ"),
    ("\\vartheta", "θ"),
    ("\\phi", "φ"),
    ("\\theta", "θ"),
    ("\\psi", "ψ"),
    ("\\omega", "ω"),
    ("\\lambda", "λ"),
    ("\\alpha", "α"),
    ("\\beta", "β"),
    ("\\gamma", "γ"),
    ("\\Delta", "Δ"),
    ("\\delta", "δ"),
    ("\\pi", "π"),
)
_FUNC_CALL = re.compile(r"(arcsin|arctan|asin|atan|sin|cos|tan)(?!\s*\()(?![A-Za-z])")
_SUB_BASE = r"[A-Za-zα-ωΑ-ΩθφψΔ]"
_NOISE = re.compile(r"[\u00a0\u200b\u200c\u200d\ufeff\u3000\u2060]")
_KEYWORD = re.compile(
    r"(?:最终答案|正确答案|参考答案|输出结果|答案|输出为|输出是|输出|结果为|结果是|结果|因此|所以|答)"
    r"\s*[:：]?\s*(?:为|是)?\s*[:：]?\s*"
)
_LEAD = re.compile(r"^(?:[*•]+\s*|-\s+(?=[\u4e00-\u9fff])|[（(]?\d+[)）.、]\s*)")
_NOTE = re.compile(r"（[^（）]{0,16}）$")


def extract_payload(text: str) -> str:
    """Keep the span after the last answer keyword and drop wrapping characters."""
    value = _NOISE.sub("", str(text or "")).replace("\r\n", "\n")
    value = re.sub(r"```(?:\w+)?", "", value)
    value = value.replace("**", "").replace("__", "")
    value = value.strip()
    found = list(_KEYWORD.finditer(value))
    if found:
        tail = value[found[-1].end():].split("\n", 1)[0].strip()
        if tail:
            value = tail
    value = value.strip()
    value = _LEAD.sub("", value)
    value = value.strip(" \t\"'`“”‘’「」『』【】")
    value = _NOTE.sub("", value)
    value = re.sub(r"[。！？；;，,]+$", "", value)
    return value.strip()


def _latex_matrices(value: str) -> str:
    def render(match: re.Match) -> str:
        rows = []
        for raw in re.split(r"\\\\", match.group(1)):
            row = raw.strip()
            if not row:
                continue
            cells = [cell.strip() for cell in row.split("&")]
            rows.append("[" + ", ".join(cells) + "]")
        return "[" + ", ".join(rows) + "]"

    previous = None
    while previous != value:
        previous = value
        value = _LATEX_MATRIX.sub(render, value)
    return value


def _unwrap_subscripts(value: str) -> str:
    value = re.sub(r"_\{([^{}]+)\}", lambda match: "_" + match.group(1).replace(",", ""), value)
    value = re.sub(rf"({_SUB_BASE})_([A-Za-z]*\d+[A-Za-z0-9]*)", r"\1\2", value)
    return value


_FUNC_NAME = re.compile(r"(?:arcsin|arctan|asin|atan|sin|cos|tan)(?![A-Za-z])")


def _read_call_arg(value: str, index: int) -> tuple[str, int]:
    while index < len(value) and value[index].isspace():
        index += 1
    if index >= len(value):
        return "", index
    if value[index] in "[({":
        end = _closing_bracket(value, index)
        if end <= index:
            return "", index
        return value[index:end], end
    if _FUNC_NAME.match(value, index):
        return "", index
    if not re.match(_SUB_BASE, value[index]):
        return "", index
    end = index + 1
    if end < len(value) and value[end] == "_":
        suffix = end + 1
        while suffix < len(value) and value[suffix].isascii() and value[suffix].isalnum():
            if _FUNC_NAME.match(value, suffix):
                break
            suffix += 1
        if suffix > end + 1:
            end = suffix
    elif end < len(value) and value[end].isdigit():
        while end < len(value) and value[end].isdigit():
            end += 1
    return value[index:end], end


def _closing_bracket(value: str, index: int) -> int:
    depth = 0
    cursor = index
    while cursor < len(value):
        char = value[cursor]
        if char in "[{(":
            depth += 1
        elif char in "]})":
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return index


def _implicit_calls(value: str) -> str:
    parts: list[str] = []
    cursor = 0
    while True:
        matched = _FUNC_CALL.search(value, cursor)
        if matched is None:
            parts.append(value[cursor:])
            break
        parts.append(value[cursor:matched.start()])
        argument, end = _read_call_arg(value, matched.end())
        if not argument:
            parts.append(matched.group(1))
            cursor = matched.end()
            continue
        parts.append(f"{matched.group(1)}({argument})")
        cursor = end
    return "".join(parts)


def normalize(text: str) -> str:
    value = extract_payload(text)
    value = value.replace("$", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = _latex_matrices(value)
    value = value.replace("\\[", "").replace("\\]", "").replace("\\(", "").replace("\\)", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = re.sub(r"\\(?:big|Big|bigg|Bigg)[lrm]?", "", value)
    value = value.replace("\\displaystyle", "").replace("\\textstyle", "")
    value = value.replace("\\cdot", "·").replace("\\times", "*")
    value = re.sub(r"\\[,;:! ]", "", value)
    value = value.replace("\\quad", "").replace("\\qquad", "")
    value = re.sub(r"\\(?:mathrm|mathbf|mathit|text|operatorname)\{([^{}]*)\}", r"\1", value)
    for source, target in _LATEX_GREEK:
        value = value.replace(source, target)
    value = re.sub(r"\\(arcsin|arctan|asin|atan|sin|cos|tan)(?![A-Za-z])", r"\1", value)
    while True:
        updated = _LATEX_FRAC.sub(r"(\1)/(\2)", value)
        if updated == value:
            break
        value = updated
    value = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", value)
    value = re.sub(r"\^(?=\d)", "**", value)
    value = _unwrap_subscripts(value)
    value = _implicit_calls(value)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = value.replace("⋅", "·").replace("∙", "·").replace("・", "·")
    value = value.replace("（", "(").replace("）", ")").replace("，", ",").replace("、", ",")
    value = value.replace("。", "")
    value = value.replace("“", "").replace("”", "").replace("‘", "").replace("’", "")
    value = value.replace('"', "").replace("'", "")
    value = re.sub(r"\s+", "", value)
    return value


def normalized_equal(prediction: str, answer: str) -> bool:
    return normalize(prediction) == normalize(answer)


def choice_letter(index: int) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if index < 26:
        return letters[index]
    return letters[index // 26 - 1] + letters[index % 26]


def format_choices(options: list[str]) -> str:
    if not options:
        return "无"
    return "；".join(f"{choice_letter(index)}：{option}" for index, option in enumerate(options))


def _split_choice(raw: str) -> tuple[str, str]:
    matched = re.match(r"^([A-Za-z])(?:\s*[:：.、]\s*(.*))?$", raw, re.S)
    if not matched:
        return "", raw
    body = matched.group(2)
    if body is None:
        return matched.group(1).upper(), ""
    return matched.group(1).upper(), body.strip()


def _pieces(option: str) -> list[str]:
    return [part for part in normalize(option).split(",") if part]


def _unique_piece_owner(prediction: str, options: list[str]) -> str | None:
    key = normalize(prediction)
    if not key:
        return None
    owners = [option for option in options if key in _pieces(option)]
    if len(owners) == 1:
        return owners[0]
    return None


def _index_of_letter(letter: str) -> int | None:
    if not letter or not letter.isalpha() or not letter.isupper():
        return None
    if len(letter) == 1:
        return ord(letter) - ord("A")
    if len(letter) == 2:
        return (ord(letter[0]) - ord("A") + 1) * 26 + (ord(letter[1]) - ord("A"))
    return None


def identified_option(prediction: str, options: list[str]) -> str | None:
    """The option a reply selects, or None when the letter and the text name different options."""
    raw = extract_payload(prediction).strip()
    letter, body = _split_choice(raw)
    by_letter = None
    index = _index_of_letter(letter)
    if index is not None and index < len(options):
        by_letter = options[index]
    by_text = None
    if body:
        for option in options:
            if normalized_equal(body, option):
                by_text = option
                break
        if by_text is None:
            by_text = _unique_piece_owner(body, options)
    if by_letter and by_text and not normalized_equal(by_letter, by_text):
        return None
    if by_letter or by_text:
        return by_text or by_letter
    for option in options:
        if normalized_equal(raw, option):
            return option
    return _unique_piece_owner(raw, options)


def resolve_choice(answer: str, options: list[str]) -> tuple[str, str]:
    """Return the gold letter and its option text. A non-choice answer keeps an empty letter."""
    raw = answer.strip()
    letter, body = _split_choice(raw)
    if letter and not body:
        index = _index_of_letter(letter)
        if index is not None and index < len(options):
            return letter, options[index]
    for index, option in enumerate(options):
        if normalized_equal(option, raw):
            return choice_letter(index), option
    return "", answer


def answer_letter(answer: str, options: list[str] | None) -> str:
    choice = [item for item in (options or []) if item and item != "无"]
    if not choice:
        return answer
    raw = answer.strip()
    # Option text may itself be a single letter, such as the reference point A.
    # That text is the gold wording; it is not the option letter.
    for index, option in enumerate(choice):
        if normalized_equal(option, raw):
            return choice_letter(index)
    letter, _text = resolve_choice(answer, choice)
    return letter or answer


def selects_option(prediction: str, answer: str, options: list[str]) -> bool:
    """True when the reply picks the gold option by letter, full text, or one unique piece."""
    letter, gold_text = resolve_choice(answer, options)
    if letter and prediction.strip() == letter:
        return True
    selected = identified_option(prediction, options)
    return selected is not None and normalized_equal(selected, gold_text)


def _sympy_modules():
    import sympy

    names = [
        "Dalpha_e", "Dbeta_e", "Dgamma_e", "Dalpha_V", "Dbeta_V",
        "alpha_M", "beta_M", "gamma_M", "alpha_e", "beta_e", "gamma_e",
        "gamma_tp", "alpha_t", "beta_t", "alpha_V", "beta_V",
        "alpha", "beta", "gamma",
        "L_RT", "L_t", "L_r", "Xg", "Yg", "Zg", "Xr",
        "CA", "CZ", "CN", "L",
    ]
    locals_map = {name: sympy.Symbol(name) for name in names}
    locals_map.update({
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "arcsin": sympy.asin,
        "arctan": sympy.atan,
        "asin": sympy.asin,
        "atan": sympy.atan,
    })
    return sympy, locals_map


def _to_sympy_text(text: str) -> str:
    value = normalize(text)
    value = value.replace("·", "*").replace("×", "*")
    for source, target in _GREEK:
        value = value.replace(source, target)
    value = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", value)
    value = re.sub(r"\)(?=[A-Za-z(])", ")*", value)
    return value


def _split_row(text: str) -> list[str]:
    cells = []
    current = []
    depth = 0
    for char in text:
        if char == "," and depth == 0:
            cells.append("".join(current))
            current = []
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        current.append(char)
    cells.append("".join(current))
    return cells


def _split_matrix(text: str) -> list[list[str]] | None:
    if not (text.startswith("[[") and text.endswith("]]")):
        return None
    body = text[1:-1]
    rows = []
    current = []
    depth = 0
    for char in body:
        if char == "[" :
            depth += 1
            if depth == 1:
                continue
        elif char == "]":
            depth -= 1
            if depth == 0:
                rows.append("".join(current))
                current = []
                continue
        elif char == "," and depth == 0:
            continue
        if depth:
            current.append(char)
    if not rows:
        return None
    return [_split_row(row) for row in rows]


def _parse_expr(text: str, sympy, locals_map):
    prepared = _to_sympy_text(text)
    matrix = _split_matrix(prepared)
    if matrix is not None:
        return sympy.Matrix([[sympy.sympify(cell, locals=locals_map) for cell in row] for row in matrix])
    if prepared.startswith("[") and prepared.endswith("]") and prepared.count("[") == 1:
        cells = _split_row(prepared[1:-1])
        return sympy.Matrix([[sympy.sympify(cell, locals=locals_map) for cell in cells]])
    return sympy.sympify(prepared, locals=locals_map)


def _is_zero(expr, sympy) -> bool:
    if isinstance(expr, sympy.MatrixBase):
        return all(cell == 0 for cell in sympy.simplify(expr))
    return sympy.simplify(expr) == 0


def sympy_equal(prediction: str, answer: str) -> bool:
    try:
        sympy, locals_map = _sympy_modules()
        left = _parse_expr(prediction, sympy, locals_map)
        right = _parse_expr(answer, sympy, locals_map)
        return _is_zero(left - right, sympy)
    except Exception:
        return False


def _sign_only(prediction: str, answer: str) -> bool:
    left = normalize(prediction)
    right = normalize(answer)
    if not left or not right or left == right:
        return False
    names_left = _AXIS_NAME.findall(left)
    names_right = _AXIS_NAME.findall(right)
    if names_left and names_left == names_right and _AXIS_TOKEN.sub("@", left) == _AXIS_TOKEN.sub("@", right):
        return True
    for pair in _OPPOSITES:
        if left in pair and right in pair and left != right:
            return True
    try:
        sympy, locals_map = _sympy_modules()
        parsed_left = _parse_expr(prediction, sympy, locals_map)
        parsed_right = _parse_expr(answer, sympy, locals_map)
        if _is_zero(parsed_left + parsed_right, sympy):
            return True
        if isinstance(parsed_left, sympy.MatrixBase) and isinstance(parsed_right, sympy.MatrixBase):
            if parsed_left.shape != parsed_right.shape:
                return False
            flipped = False
            for cell_left, cell_right in zip(parsed_left, parsed_right):
                if _is_zero(cell_left - cell_right, sympy):
                    continue
                if _is_zero(cell_left + cell_right, sympy):
                    flipped = True
                    continue
                return False
            return flipped
    except Exception:
        return False
    return False


def _axis_swap(prediction: str, answer: str) -> bool:
    left = normalize(prediction)
    right = normalize(answer)
    names_left = _AXIS_NAME.findall(left)
    names_right = _AXIS_NAME.findall(right)
    if not names_left or not names_right or names_left == names_right:
        return False
    return _AXIS_TOKEN.sub("@", left) == _AXIS_TOKEN.sub("@", right)


def classify(prediction: str, answer: str, accepted: bool, standard: str | None) -> str:
    if accepted:
        return "格式" if normalize(prediction) != normalize(answer) or prediction != answer else ""
    if standard is not None and standard != answer and (
        normalized_equal(prediction, standard) or (
            _looks_symbolic(prediction) and _looks_symbolic(standard) and sympy_equal(prediction, standard)
        )
    ):
        return "定义未用"
    if _sign_only(prediction, answer):
        return "符号"
    if _axis_swap(prediction, answer):
        return "轴"
    return "其他"


def _looks_symbolic(text: str) -> bool:
    value = normalize(text)
    return value.startswith("[") or any(name in value for name in ("sin", "cos", "arctan", "arcsin", "·", "/", "+", "-"))


def judge(
    prediction: str,
    answer: str,
    format_code: str,
    options: list[str] | None = None,
    standard: str | None = None,
) -> dict:
    choice = [item for item in (options or []) if item and item != "无"]
    letter, gold_text = resolve_choice(answer, choice) if choice else ("", answer)
    scored_answer = letter if letter else answer
    exact = prediction == scored_answer
    same_text = normalized_equal(prediction, scored_answer)
    selected = identified_option(prediction, choice) if choice else None
    option_match = None
    if choice:
        option_match = selected is not None and normalized_equal(selected, gold_text)
    sympy_hit = False
    if not exact and not same_text and not choice and format_code in SYMBOLIC:
        sympy_hit = sympy_equal(prediction, answer)
    accepted = exact or same_text or sympy_hit
    if choice:
        accepted = bool(option_match) or exact
    compared = prediction
    class_answer = answer
    class_standard = standard
    if choice and not accepted:
        if selected:
            compared = selected
        class_answer = gold_text
        if standard is not None:
            standard_letter, standard_text = resolve_choice(standard, choice)
            class_standard = standard_text if standard_letter else standard
    kind = ""
    if not exact:
        kind = "格式" if accepted else classify(compared, class_answer, False, class_standard)
    return {
        "exact": exact,
        "normalized": accepted,
        "sympy_equal": sympy_hit,
        "option_match": option_match,
        "error_kind": kind,
    }


def bootstrap_ci(groups: list[list[bool]], draws: int = 2000, seed: int = 20261102) -> tuple[float, float]:
    if not groups or not any(groups):
        return 0.0, 0.0
    rng = random.Random(seed)
    rates = []
    width = len(groups)
    for _ in range(draws):
        chosen = [groups[rng.randrange(width)] for _ in range(width)]
        total = sum(len(group) for group in chosen)
        hit = sum(sum(group) for group in chosen)
        rates.append(hit / total if total else 0.0)
    rates.sort()
    low = rates[int(0.025 * (draws - 1))]
    high = rates[int(0.975 * (draws - 1))]
    return low, high


def _check_scoring() -> None:
    spaced = judge("γ_M + γ_tp", "γ_M+γ_tp", "表达式")
    if not (spaced["normalized"] and not spaced["exact"] and spaced["error_kind"] == "格式"):
        raise RuntimeError(spaced)
    signed_space = judge("- YTP", "-YTP", "无", ["-YTP", "YTP"])
    if not signed_space["normalized"] or signed_space["error_kind"] != "格式":
        raise RuntimeError(signed_space)
    dropped_sign = judge("-YTP", "YTP", "无", ["-YTP", "YTP"])
    if dropped_sign["normalized"] or dropped_sign["error_kind"] != "符号":
        raise RuntimeError(dropped_sign)
    axis = judge("YTZ", "YFD", "无", ["YTZ", "YFD"])
    if axis["error_kind"] != "轴":
        raise RuntimeError(axis)
    matrix = judge("[[1,0,0],[0,1,0],[0,0,1]]", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "矩阵")
    if not matrix["normalized"] or matrix["error_kind"] != "格式":
        raise RuntimeError(matrix)
    trig = judge("sin(α_M+β_M)", "sin(α_M)*cos(β_M)+cos(α_M)*sin(β_M)", "表达式")
    if not trig["sympy_equal"] or not trig["normalized"]:
        raise RuntimeError(trig)
    juxtaposed = judge(
        "arctan((sin(α_M)cos(γ_M)+cos(α_M)sin(β_M)sin(γ_M))/(cos(α_M)cos(β_M)))",
        "arctan((sin(α_M)*cos(γ_M) + sin(β_M)*sin(γ_M)*cos(α_M))/(cos(α_M)*cos(β_M)))",
        "arctan",
    )
    if not juxtaposed["normalized"] or juxtaposed["exact"] or juxtaposed["error_kind"] != "格式":
        raise RuntimeError(juxtaposed)
    outer_minus = judge(
        "-arcsin(cos(α_M)sin(β_M)cos(γ_M)-sin(α_M)sin(γ_M))",
        "arcsin(sin(β_M)*cos(α_M)*cos(γ_M) - sin(α_M)*sin(γ_M))",
        "arcsin",
    )
    if outer_minus["normalized"] or outer_minus["error_kind"] != "符号":
        raise RuntimeError(outer_minus)
    prefix = judge("因此答案为：正装。", "正装", "无", ["正装", "反装"])
    if not prefix["normalized"] or prefix["exact"] or prefix["error_kind"] != "格式":
        raise RuntimeError(prefix)
    wrapped = judge("所以 γ_M + γ_tp 。", "γ_M+γ_tp", "表达式")
    if not wrapped["normalized"] or wrapped["error_kind"] != "格式":
        raise RuntimeError(wrapped)
    marked = judge("最终答案：[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]", "矩阵")
    if not marked["normalized"] or marked["exact"]:
        raise RuntimeError(marked)
    piece = judge("CZ_TZ", "CZ_TZ,MZ_TZ", "无", ["CN_TZ,MN_TZ", "CA_TZ,MA_TZ", "CZ_TZ,MZ_TZ"])
    if not piece["normalized"] or piece["exact"] or piece["error_kind"] != "格式":
        raise RuntimeError(piece)
    letter = judge("A", "正", "无", ["正", "负", "不确定"])
    if not letter["normalized"] or not letter["exact"] or letter["error_kind"]:
        raise RuntimeError(letter)
    letter_stored = judge("A", "A", "无", ["正", "负", "不确定"])
    if not letter_stored["exact"]:
        raise RuntimeError(letter_stored)
    option_text = judge("正", "A", "无", ["正", "负", "不确定"])
    if not option_text["normalized"] or option_text["exact"] or option_text["error_kind"] != "格式":
        raise RuntimeError(option_text)
    pair_letter = judge("C", "CZ_TZ,MZ_TZ", "无", ["CN_TZ,MN_TZ", "CA_TZ,MA_TZ", "CZ_TZ,MZ_TZ"])
    if not pair_letter["exact"]:
        raise RuntimeError(pair_letter)
    wrong_letter = judge("B", "正", "无", ["正", "负", "不确定"])
    if wrong_letter["normalized"] or wrong_letter["error_kind"] != "符号":
        raise RuntimeError(wrong_letter)
    shared = judge("CZ_TZ", "CN_TZ,MN_TZ", "无", ["CN_TZ,MN_TZ", "CA_TZ,MA_TZ", "CZ_TZ,MZ_TZ"])
    if shared["normalized"]:
        raise RuntimeError(shared)
    conflict = judge("A：负", "正", "无", ["正", "负", "不确定"])
    if conflict["normalized"]:
        raise RuntimeError(conflict)
    unused_phrase = judge("因此答案为：正装。", "反装", "无", ["正装", "反装"], standard="正装")
    if unused_phrase["normalized"] or unused_phrase["error_kind"] != "定义未用":
        raise RuntimeError(unused_phrase)
    latex = judge(r"$\sin(\alpha_M)$", "sin(α_M)", "表达式")
    if not latex["normalized"]:
        raise RuntimeError(latex)
    latex_call = judge(r"\sin\theta_M\cos\varphi_M", "sin(θ_M)*cos(φ_M)", "表达式")
    if not latex_call["normalized"]:
        raise RuntimeError(latex_call)
    latex_sub = judge(
        r"[(c_{y1}-v_1)/f_{y1}, -1, (u_1-c_{x1})/f_{x1}]",
        "[(cy1 - v1)/fy1, -1, (-cx1 + u1)/fx1]",
        "向量",
    )
    if not latex_sub["normalized"]:
        raise RuntimeError(latex_sub)
    latex_frac = judge(r"\frac{u1-cx1}{fx1}", "(u1-cx1)/fx1", "表达式")
    if not latex_frac["normalized"]:
        raise RuntimeError(latex_frac)
    latex_matrix = judge(
        r"\begin{bmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix}",
        "[[0, -1, 0], [1, 0, 0], [0, 0, 0]]",
        "矩阵",
    )
    if not latex_matrix["normalized"]:
        raise RuntimeError(latex_matrix)
    latex_display = judge(
        r"\[\left[\frac{cy1-v1}{fy1},\,-1,\,\frac{u1-cx1}{fx1}\right]\]",
        "[(cy1 - v1)/fy1, -1, (u1 - cx1)/fx1]",
        "向量",
    )
    if not latex_display["normalized"]:
        raise RuntimeError(latex_display)
    unused = judge("正装", "反装", "无", ["正装", "反装"], standard="正装")
    if unused["error_kind"] != "定义未用":
        raise RuntimeError(unused)
    sign_matrix = judge("[[0, 0, 1], [0, 1, 0], [1, 0, 0]]", "[[0, 0, -1], [0, -1, 0], [-1, 0, 0]]", "矩阵")
    if sign_matrix["normalized"] or sign_matrix["error_kind"] != "符号":
        raise RuntimeError(sign_matrix)
    low, high = bootstrap_ci([[True, False], [True, True]])
    if not (0 <= low <= high <= 1):
        raise RuntimeError((low, high))
