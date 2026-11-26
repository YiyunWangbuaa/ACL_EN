# -*- coding: utf-8 -*-
"""Four kinematics manuscripts that flip joint positive directions.

Sample 01 stays the standard. Each new file changes only which joints use the
opposite of the right-hand rule, then recomputes the blanks that follow:
rotation matrices from the contract verifiers, the example pose, task angles,
the position expressions, and the joint-extraction formula. A variant is not
written if those pieces disagree.

Run::

    python -m domains.robotics.kin_variant_generate
"""
from __future__ import annotations

import re
from pathlib import Path

import sympy as sp

from domain_layout import DOMAINS
from domains.robotics.kin_contract_verifiers import (
    DERIVED_NODES,
    derive,
    gold_texts,
    input_ids,
)

OUT = Path(DOMAINS["robotics"]["kinematics"]["review"])
SOURCE = OUT / "sample_01.md"
SLOT = re.compile(
    r"\{ID:(\d+), options:(.*?), format:(.*?), label:(.*?), answer:(.*?)\}",
    re.S,
)
Q1, Q2, Q3 = sp.symbols("q1 q2 q3")
LENGTH = sp.symbols("L", positive=True)
S = sp.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
EXAMPLE = {Q1: 0, Q2: sp.rad(30), Q3: sp.rad(-60)}
JOINT = {"q1": "F12", "q2": "F17", "q3": "F22"}
BULLET = {
    "q1": "- About the normal axis, the positive sense agrees with the right-hand rule about up.",
    "q2": "- About the lateral axis, the positive sense agrees with the right-hand rule about right.",
    "q3": "- About the axial axis, the positive sense agrees with the right-hand rule about forward.",
}
BULLET_OPPOSITE = {
    "q1": "- About the normal axis, the positive sense is opposite the right-hand rule about up.",
    "q2": "- About the lateral axis, the positive sense is opposite the right-hand rule about right.",
    "q3": "- About the axial axis, the positive sense is opposite the right-hand rule about forward.",
}
VARIANTS = (
    ("02", "q1 sense reversed", ("q1",)),
    ("03", "q2 sense reversed", ("q2",)),
    ("04", "q1 and q2 sense reversed", ("q1", "q2")),
    ("05", "all three joint senses reversed", ("q1", "q2", "q3")),
)
_ROLL_DEFAULT = (
    "In this example the tool-roll task angle may be written "
    "arcsin(1/2) or atan2(1, sqrt(3)) or as an equivalent inverse trigonometric form."
)


def _roll_clause(angle: str) -> str | None:
    if angle.startswith("-arcsin(1/2)"):
        form = "-arcsin(1/2)"
    elif angle.startswith("atan2(1/2"):
        form = "atan2(1/2"
    else:
        return None
    return (
        "In this example the tool-roll task angle may be written "
        + form
        + " or as an equivalent inverse trigonometric form."
    )


def _cascade(flips: tuple[str, ...]) -> dict[str, str]:
    texts = dict(gold_texts())
    texts.update({JOINT[name]: "opposite" for name in flips})
    pending = set(DERIVED_NODES)
    guard = 0
    while pending and guard < 400:
        guard += 1
        ready = [node for node in list(pending) if all(parent not in pending for parent in input_ids(node))]
        if not ready:
            raise RuntimeError("derived nodes have no computable order")
        for node in ready:
            got = derive(None, node, texts)
            if not got.ok:
                raise RuntimeError(f"{node} {got.reason}")
            texts[node] = str(got.matrix)
            pending.remove(node)
    return texts


def _matrix(text: str) -> sp.Matrix:
    return sp.Matrix(sp.sympify(text))


def _rotations(texts: dict[str, str]) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    return _matrix(texts["F14"]), _matrix(texts["F19"]), _matrix(texts["F24"])


def _pose(r1: sp.Matrix, r2: sp.Matrix, r3: sp.Matrix) -> sp.Matrix:
    return sp.simplify((r1 * r2 * r3 * S).subs(EXAMPLE))


def _signs(flips: tuple[str, ...]) -> tuple[int, int, int]:
    return (
        1 if "q1" in flips else -1,
        -1 if "q2" in flips else 1,
        1 if "q3" in flips else -1,
    )


def _formula(s31: int, s21: int, s23: int) -> str:
    def term(sign: int, name: str) -> str:
        return name if sign == 1 else f"-{name}"

    return (
        f"atan2({term(s31, 'h31')}, h11); "
        f"atan2({term(s21, 'h21')}, sqrt(h11^2 + h31^2)); "
        f"atan2({term(s23, 'h23')}, h22)"
    )


def _split_top(text: str) -> tuple[str, str]:
    depth = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            return text[:index], text[index + 1:]
    raise RuntimeError(f"formula is missing a comma: {text}")


def _same(left: sp.Matrix, right: sp.Matrix) -> bool:
    return sp.simplify(left - right) == sp.zeros(*left.shape)


def _eval_part(part: str, env: dict) -> sp.Expr:
    body = part.strip()
    if not body.startswith("atan2(") or not body.endswith(")"):
        raise RuntimeError(part)
    left, right = _split_top(body[len("atan2("):-1])

    def parse(text: str) -> sp.Expr:
        return sp.sympify(text.replace("^", "**"), locals=env)

    return sp.atan2(parse(left), parse(right))


def _extract_degrees(formula: str, pose: sp.Matrix) -> list[float]:
    env = {f"h{row}{col}": pose[row - 1, col - 1] for row in range(1, 4) for col in range(1, 4)}
    env["sqrt"] = sp.sqrt
    return [float(sp.deg(_eval_part(part, env))) for part in formula.split(";")]


def _check_formula(r1: sp.Matrix, r2: sp.Matrix, r3: sp.Matrix, formula: str) -> None:
    grid = (-50, -20, 0, 20, 50)
    lateral = (-70, -30, 0, 30, 70)
    for a in grid:
        for b in lateral:
            for c in lateral:
                pose = sp.simplify((r1 * r2 * r3).subs({Q1: sp.rad(a), Q2: sp.rad(b), Q3: sp.rad(c)}))
                got = _extract_degrees(formula, pose)
                if max(abs(got[0] - a), abs(got[1] - b), abs(got[2] - c)) > 1e-6:
                    raise RuntimeError(f"extraction at ({a}, {b}, {c}) gave {got}")
    example = _extract_degrees(formula, sp.simplify((r1 * r2 * r3).subs(EXAMPLE)))
    if max(abs(example[0]), abs(example[1] - 30), abs(example[2] + 60)) > 1e-6:
        raise RuntimeError(f"example joint angles extracted as {example}")


def _task_angles(pose: sp.Matrix) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    dx, dy, dz = (sp.simplify(pose[0, index]) for index in range(3))
    alpha = sp.simplify(sp.atan2(-dy, dx))
    beta = sp.simplify(sp.atan2(dz, sp.sqrt(dx**2 + dy**2)))
    forward = pose[:, 0]
    upward = pose[:, 1]
    reference = sp.Matrix([0, 1, 0])
    raw = reference - reference.dot(forward) * forward
    unit = sp.simplify(raw / raw.norm())
    side = sp.simplify(forward.cross(unit))
    gamma = sp.simplify(sp.atan2(upward.dot(side), upward.dot(unit)))
    return alpha, beta, gamma


def _rebuild(alpha: sp.Expr, beta: sp.Expr, gamma: sp.Expr) -> sp.Matrix:
    a = sp.cos(alpha) * sp.cos(beta)
    bv = -sp.sin(alpha) * sp.cos(beta)
    c = sp.sin(beta)
    d = bv * sp.cos(gamma) - c * sp.sin(gamma)
    norm = sp.sqrt(a**2 + d**2)
    row1 = sp.Matrix([a, bv, c])
    row2 = sp.Matrix([-d / norm, a * sp.cos(gamma) / norm, -a * sp.sin(gamma) / norm])
    row3 = sp.simplify(row1.cross(row2))
    return sp.simplify(sp.Matrix.vstack(row1.T, row2.T, row3.T))


def _rational(expr: sp.Expr) -> tuple[int, int] | None:
    expr = sp.simplify(expr)
    if not expr.is_rational:
        return None
    num, den = sp.fraction(sp.together(expr))
    return int(num), int(den)


def _component(expr: sp.Expr) -> str:
    return str(sp.simplify(expr))


def _angle_text(value: sp.Expr) -> str:
    value = sp.simplify(value)
    if value == 0:
        return "0"
    if _positive(sp.cos(value)):
        ratio = _rational(sp.sin(value))
        if ratio is not None:
            kind = "arcsin"
        else:
            ratio = _rational(sp.tan(value))
            kind = "arctan"
        if ratio is not None:
            num, den = ratio
            sign = "-" if num < 0 else ""
            arg = str(abs(num)) if den == 1 else f"{abs(num)}/{den}"
            return f"{sign}{kind}({arg})"
    return f"atan2({_component(sp.sin(value))}, {_component(sp.cos(value))})"


def _show_scalar(expr: sp.Expr) -> str:
    text = str(sp.simplify(expr))
    if text == "0":
        return "0"
    if "L" not in text:
        raise RuntimeError(f"position expression has no length: {text}")
    return text.replace("L", "(L + L_mount)")


def _positions(r1: sp.Matrix, r2: sp.Matrix) -> dict[str, str]:
    middle = sp.simplify(r2 * sp.Matrix([LENGTH, 0, 0]))
    base = sp.simplify(r1 * middle)
    vector = "[" + ", ".join(_show_scalar(middle[index]) for index in range(3)) + "]"
    return {
        "F118": vector,
        "F119": _show_scalar(base[1]),
        "F124": _show_scalar(base[0]),
        "F125": _show_scalar(base[2]),
    }


def _standard_rotation(axis: str, degree: int) -> sp.Matrix:
    angle = sp.rad(degree)
    cosine, sine = sp.cos(angle), sp.sin(angle)
    if axis == "Z":
        return sp.Matrix([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    if axis == "X":
        return sp.Matrix([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]])
    raise RuntimeError(axis)


def _pair(flips: tuple[str, ...]) -> str:
    first = [0, 90, 0]
    second = [30, 90, -30]
    for index, name in enumerate(("q1", "q2", "q3")):
        if name in flips:
            first[index] *= -1
            second[index] *= -1
    for triple in (first, second):
        if not (-60 <= triple[0] <= 60 and -90 <= triple[1] <= 90 and -90 <= triple[2] <= 90):
            raise RuntimeError(f"singular pair is outside the declared range: {triple}")

    def fmt(number: int) -> str:
        return f"{number}°"

    return (
        f"q=({fmt(first[0])}, {fmt(first[1])}, {fmt(first[2])}) and "
        f"q=({fmt(second[0])}, {fmt(second[1])}, {fmt(second[2])})"
    )


def _inverse_sentence(flips: tuple[str, ...]) -> str:
    if flips == ("q1",):
        return (
            "The positive sense of q1 is opposite the right-hand rule about its absolute direction, "
            "and q2 and q3 agree with it."
        )
    if flips == ("q2",) or flips == ("q1", "q2"):
        return (
            "The positive sense of q2 is opposite the right-hand rule about its absolute direction, "
            "and q1 and q3 agree with it."
        )
    if flips == ("q3",):
        return (
            "The positive sense of q3 is opposite the right-hand rule about its absolute direction, "
            "and q1 and q2 agree with it."
        )
    return "Every positive sense agrees with the right-hand rule about the corresponding absolute direction."


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"phrase appears {count} times: {old}")
    return text.replace(old, new, 1)


def _options(raw: str) -> list[str]:
    if raw.strip() == "none":
        return []
    found = []
    for part in raw.split(";"):
        part = part.strip()
        if ":" in part:
            found.append(part.split(":", 1)[1].strip())
    return found


def _apply_slots(section: str, prefix: str, updates: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        node = f"{prefix}{match.group(1)}"
        if node not in updates:
            return match.group(0)
        raw_options = match.group(2)
        options = _options(raw_options)
        answer = updates[node]
        if options:
            if answer not in options:
                letter = chr(ord("A") + len(options))
                raw_options = f"{raw_options}; {letter}: {answer}"
            else:
                letter = chr(ord("A") + options.index(answer))
            stored = letter
        else:
            stored = answer
        return (
            "{ID:%s, options:%s, format:%s, label:%s, answer:%s}"
            % (match.group(1), raw_options, match.group(3), match.group(4), stored)
        )

    return SLOT.sub(repl, section)


def _positive(value: sp.Expr) -> bool:
    flag = value.is_positive
    if flag is None:
        flag = float(value) > 0
    return bool(flag)


def _direction(alpha: sp.Expr, beta: sp.Expr) -> tuple[str, str, str]:
    if alpha == 0 or beta == 0:
        raise RuntimeError("an example task angle is 0, so the direction sentence needs another wording")
    vertical = "downward" if _positive(alpha) else "upward"
    lateral = "right" if _positive(beta) else "left"
    pointed = {
        ("downward", "right"): "down and right",
        ("downward", "left"): "down and left",
        ("upward", "right"): "up and right",
        ("upward", "left"): "up and left",
    }[(vertical, lateral)]
    return vertical, lateral, pointed


def build_text(sample_id: str, title: str, flips: tuple[str, ...]) -> str:
    texts = _cascade(flips)
    gold = gold_texts()
    r1, r2, r3 = _rotations(texts)
    pose = _pose(r1, r2, r3)
    if not _same(pose, sp.simplify(_matrix(texts["F45"]))):
        raise RuntimeError(f"{sample_id} attitude matrix does not match the joint-matrix product")
    s31, s21, s23 = _signs(flips)
    formula = _formula(s31, s21, s23)
    _check_formula(r1, r2, r3, formula)
    alpha, beta, gamma = _task_angles(pose)
    rebuilt = _rebuild(alpha, beta, gamma)
    if not _same(rebuilt, pose):
        raise RuntimeError(f"{sample_id} task angles do not recover the attitude matrix")
    angles = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    q2_degree = -30 if "q2" in flips else 30
    roll_degree = 150 if "q3" in flips else 30
    phrase = f"{q2_degree}° about Z times {roll_degree}° about X"
    if not _same(_standard_rotation("Z", q2_degree) * _standard_rotation("X", roll_degree), pose):
        raise RuntimeError(f"{sample_id} attitude phrase does not match the matrix")
    combined = -150 if "q3" in flips else 30
    target = sp.simplify(r3.subs({Q3: sp.rad(-60)}) * S)
    if not _same(sp.simplify(r3.subs({Q3: sp.rad(combined)})), target):
        raise RuntimeError(f"{sample_id} combined angle is not {combined}°")
    pair = _pair(flips)
    numbers = [int(item[:-1]) for item in re.findall(r"-?\d+°", pair)]
    first_pose = (r1 * r2 * r3).subs({
        Q1: sp.rad(numbers[0]), Q2: sp.rad(numbers[1]), Q3: sp.rad(numbers[2]),
    })
    other = (r1 * r2 * r3).subs({
        Q1: sp.rad(numbers[3]), Q2: sp.rad(numbers[4]), Q3: sp.rad(numbers[5]),
    })
    if not _same(sp.simplify(first_pose), sp.simplify(other)):
        raise RuntimeError(f"{sample_id} boundary joint triples are not the same attitude")
    vertical, lateral, pointed = _direction(alpha, beta)
    positions = _positions(r1, r2)
    updates = {node: texts[node] for node in texts if texts[node] != gold[node]}
    updates["F39"] = f"{combined}°"
    updates["F40"] = phrase
    updates["F65"] = angles
    updates["F64"] = "positive" if _positive(gamma) else "negative"
    updates["F53"] = pointed
    updates["F59"] = "positive" if vertical == "downward" else "negative"
    updates["F60"] = "positive" if lateral == "right" else "negative"
    updates.update(positions)
    updates["I7"] = texts["F48"]
    updates["I9"] = texts["F45"]
    updates["I15"] = formula
    updates["I16"] = "[0, 30, -60]"
    updates = {node: answer for node, answer in updates.items() if answer != gold[node]}

    text = SOURCE.read_text(encoding="utf-8")
    text = _replace_once(text, "# Data 01 upright wrist", f"# Data {sample_id} {title}")
    text = _replace_once(
        text,
        "The positive sense of a joint angle follows the right-hand rule about that absolute direction, and it does not flip when a coordinate axis is reversed.",
        "The positive sense of a joint angle agrees or disagrees with the right-hand rule about that absolute direction according to the variable definition, and it does not flip when a coordinate axis is reversed.",
    )
    text = _replace_once(
        text,
        "This is a variable definition. The positive sense follows the right-hand rule about the absolute axis being turned, not a separate rule based on axes X, Y, and Z.",
        "This is a variable definition. The positive sense is the three items below, not a separate rule based on axes X, Y, and Z.",
    )
    for name in flips:
        text = _replace_once(text, BULLET[name], BULLET_OPPOSITE[name])
    text = _replace_once(
        text,
        "The positive sense of a joint angle follows the right-hand rule about the absolute axis it turns about, and it does not flip when a coordinate axis is reversed.",
        "The positive sense of a joint angle follows the declared joint-angle sign, and it does not flip when a coordinate axis is reversed.",
    )
    text = _replace_once(
        text,
        "Every positive sense agrees with the right-hand rule about the corresponding absolute direction.",
        _inverse_sentence(flips),
    )
    text = _replace_once(
        text,
        "Both joint triples are inside the declared range: q=(0°, 90°, 0°) and q=(30°, 90°, -30°).",
        "Both joint triples are inside the declared range: " + pair + ".",
    )
    if vertical != "downward":
        text = _replace_once(
            text,
            "59. In this example the task direction points tool-down,",
            "59. In this example the task direction points tool-up,",
        )
    if lateral != "right":
        text = _replace_once(
            text,
            "60. In this example the task direction points tool-right,",
            "60. In this example the task direction points tool-left,",
        )
    clause = _roll_clause(angles.split(", ")[2])
    if clause is not None:
        text = _replace_once(text, _ROLL_DEFAULT, clause)
    text = _replace_once(
        text,
        "Under the default order the extraction is atan2(-h31, h11); atan2(h21, sqrt(h11^2 + h31^2)); atan2(-h23, h22).",
        f"The extraction for this sample is {formula}.",
    )
    forward, inverse = text.split("# Inverse kinematics", 1)
    inverse, tail = inverse.split("# Cross-layer correspondence", 1)
    forward = _apply_slots(forward, "F", updates)
    inverse = _apply_slots(inverse, "I", updates)
    return forward + "# Inverse kinematics" + inverse + "# Cross-layer correspondence" + tail


def _check_written(path: Path, flips: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    forward, inverse = text.split("# Inverse kinematics", 1)
    inverse = inverse.split("# Cross-layer correspondence", 1)[0]
    for prefix, section, last in (("F", forward, 140), ("I", inverse, 16)):
        found = [int(match.group(1)) for match in SLOT.finditer(section)]
        if found != list(range(1, last + 1)):
            raise RuntimeError(f"{path.name} {prefix} ids are not 1-{last}")
    for name, node in JOINT.items():
        number = node[1:]
        match = re.search(rf"\{{ID:{number}, options:(.*?), format:choice, label:C1, answer:(.)\}}", forward)
        if match is None:
            raise RuntimeError(f"{path.name} missing {node}")
        options = _options(match.group(1))
        letter = match.group(2)
        expected = "opposite" if name in flips else "same"
        if options[ord(letter) - ord("A")] != expected:
            raise RuntimeError(f"{path.name} {node} is not {expected}")


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError(f"missing {SOURCE}")
    # The unflipped matrices must reproduce the published expressions and angles.
    base = _cascade(())
    r1, r2, r3 = _rotations(base)
    pose = _pose(r1, r2, r3)
    alpha, beta, gamma = _task_angles(pose)
    published = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    if published != gold_texts()["F65"]:
        raise RuntimeError(f"default task angles were written as {published}")
    if _positions(r1, r2) != {node: gold_texts()[node] for node in ("F118", "F119", "F124", "F125")}:
        raise RuntimeError("default position expressions were not recovered")
    _check_formula(r1, r2, r3, _formula(-1, 1, -1))
    for sample_id, title, flips in VARIANTS:
        text = build_text(sample_id, title, flips)
        path = OUT / f"sample_{sample_id}.md"
        path.write_text(text, encoding="utf-8")
        _check_written(path, flips)
        print(f"wrote {path.name}  {title}", flush=True)


if __name__ == "__main__":
    main()
