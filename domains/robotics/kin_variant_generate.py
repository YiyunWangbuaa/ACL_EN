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
    r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}",
    re.S,
)
Q1, Q2, Q3 = sp.symbols("q1 q2 q3")
LENGTH = sp.symbols("L", positive=True)
S = sp.Matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
EXAMPLE = {Q1: 0, Q2: sp.rad(30), Q3: sp.rad(-60)}
JOINT = {"q1": "F12", "q2": "F17", "q3": "F22"}
BULLET = {
    "q1": "绕法向轴时，正方向与绕上方的右手定则相同。",
    "q2": "绕侧向轴时，正方向与绕右侧的右手定则相同。",
    "q3": "绕轴向轴时，正方向与绕前方的右手定则相同。",
}
VARIANTS = (
    ("02", "q1 正方向相反", ("q1",)),
    ("03", "q2 正方向相反", ("q2",)),
    ("04", "q1 与 q2 正方向相反", ("q1", "q2")),
    ("05", "三个关节正方向相反", ("q1", "q2", "q3")),
)


def _cascade(flips: tuple[str, ...]) -> dict[str, str]:
    texts = dict(gold_texts())
    texts.update({JOINT[name]: "相反" for name in flips})
    pending = set(DERIVED_NODES)
    guard = 0
    while pending and guard < 400:
        guard += 1
        ready = [node for node in list(pending) if all(parent not in pending for parent in input_ids(node))]
        if not ready:
            raise RuntimeError("派生节点没有可计算的顺序")
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
    raise RuntimeError(f"公式缺少逗号：{text}")


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
                    raise RuntimeError(f"提取公式在 ({a}, {b}, {c}) 得到 {got}")
    example = _extract_degrees(formula, sp.simplify((r1 * r2 * r3).subs(EXAMPLE)))
    if max(abs(example[0]), abs(example[1] - 30), abs(example[2] + 60)) > 1e-6:
        raise RuntimeError(f"例题关节角提取为 {example}")


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
        raise RuntimeError(f"位置表达式里没有长度：{text}")
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
            raise RuntimeError(f"奇异配对超出声明范围：{triple}")

    def fmt(number: int) -> str:
        return f"{number}°"

    return (
        f"q=({fmt(first[0])}, {fmt(first[1])}, {fmt(first[2])}) 和 "
        f"q=({fmt(second[0])}, {fmt(second[1])}, {fmt(second[2])})"
    )


def _inverse_sentence(flips: tuple[str, ...]) -> str:
    names = {"q1": "q1", "q2": "q2", "q3": "q3"}
    opposite = [names[name] for name in ("q1", "q2", "q3") if name in flips]
    same = [names[name] for name in ("q1", "q2", "q3") if name not in flips]
    if not same:
        return "正方向都与绕所对应绝对正方向的右手定则相反。"
    return (
        f"{'、'.join(opposite)} 的正方向与绕所对应绝对正方向的右手定则相反，"
        f"{'、'.join(same)} 相同。"
    )


def _replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"文句出现 {count} 次：{old}")
    return text.replace(old, new, 1)


def _options(raw: str) -> list[str]:
    if raw.strip() == "无":
        return []
    found = []
    for part in raw.split("；"):
        part = part.strip()
        if "：" in part:
            found.append(part.split("：", 1)[1].strip())
        elif ":" in part:
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
                raw_options = f"{raw_options}；{letter}：{answer}"
            else:
                letter = chr(ord("A") + options.index(answer))
            stored = letter
        else:
            stored = answer
        return (
            "{ID:%s, 选项:%s, 输出格式:%s, 标签:%s, 答案:%s}"
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
        raise RuntimeError("例题任务角有一个为 0，指向句需要另写")
    vertical = "下方" if _positive(alpha) else "上方"
    lateral = "右侧" if _positive(beta) else "左侧"
    return vertical, lateral, f"{vertical}且{lateral}"


def build_text(sample_id: str, title: str, flips: tuple[str, ...]) -> str:
    texts = _cascade(flips)
    gold = gold_texts()
    r1, r2, r3 = _rotations(texts)
    pose = _pose(r1, r2, r3)
    if not _same(pose, sp.simplify(_matrix(texts["F45"]))):
        raise RuntimeError(f"{sample_id} 的姿态矩阵和关节矩阵乘积不一致")
    s31, s21, s23 = _signs(flips)
    formula = _formula(s31, s21, s23)
    _check_formula(r1, r2, r3, formula)
    alpha, beta, gamma = _task_angles(pose)
    rebuilt = _rebuild(alpha, beta, gamma)
    if not _same(rebuilt, pose):
        raise RuntimeError(f"{sample_id} 的任务角不能还原姿态矩阵")
    angles = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    q2_degree = -30 if "q2" in flips else 30
    roll_degree = 150 if "q3" in flips else 30
    phrase = f"绕 Z 的 {q2_degree}° 乘绕 X 的 {roll_degree}°"
    if not _same(_standard_rotation("Z", q2_degree) * _standard_rotation("X", roll_degree), pose):
        raise RuntimeError(f"{sample_id} 的姿态口诀和矩阵不一致")
    combined = -150 if "q3" in flips else 30
    target = sp.simplify(r3.subs({Q3: sp.rad(-60)}) * S)
    if not _same(sp.simplify(r3.subs({Q3: sp.rad(combined)})), target):
        raise RuntimeError(f"{sample_id} 的合成转角不是 {combined}°")
    pair = _pair(flips)
    numbers = [int(item[:-1]) for item in re.findall(r"-?\d+°", pair)]
    first_pose = (r1 * r2 * r3).subs({
        Q1: sp.rad(numbers[0]), Q2: sp.rad(numbers[1]), Q3: sp.rad(numbers[2]),
    })
    other = (r1 * r2 * r3).subs({
        Q1: sp.rad(numbers[3]), Q2: sp.rad(numbers[4]), Q3: sp.rad(numbers[5]),
    })
    if not _same(sp.simplify(first_pose), sp.simplify(other)):
        raise RuntimeError(f"{sample_id} 的两组边界关节角不是同一姿态")
    vertical, lateral, pointed = _direction(alpha, beta)
    positions = _positions(r1, r2)
    updates = {node: texts[node] for node in texts if texts[node] != gold[node]}
    updates["F39"] = f"{combined}°"
    updates["F40"] = phrase
    updates["F65"] = angles
    updates["F64"] = "正" if _positive(gamma) else "负"
    updates["F53"] = pointed
    updates["F59"] = "正" if vertical == "下方" else "负"
    updates["F60"] = "正" if lateral == "右侧" else "负"
    updates.update(positions)
    updates["I7"] = texts["F48"]
    updates["I9"] = texts["F45"]
    updates["I15"] = formula
    updates["I16"] = "[0, 30, -60]"
    updates = {node: answer for node, answer in updates.items() if answer != gold[node]}

    text = SOURCE.read_text(encoding="utf-8")
    text = _replace_once(text, "# 数据 01 球腕正装", f"# 数据 {sample_id} {title}")
    text = _replace_once(
        text,
        "关节角的正方向绑定在所绕的绝对方向轴上，按该绝对正方向的右手定则，不随坐标轴反向而翻转。",
        "关节角的正方向绑定在所绕的绝对方向轴上，与该绝对正方向的右手定则相同或相反以可变定义为准，不随坐标轴反向而翻转。",
    )
    text = _replace_once(
        text,
        "这是可变定义。正方向一律按所绕绝对方向轴的右手定则，不按坐标轴 X、Y、Z 的指向另行规定。",
        "这是可变定义。正方向以下列三项为准，不按坐标轴 X、Y、Z 的指向另行规定。",
    )
    for name in flips:
        text = _replace_once(text, BULLET[name], BULLET[name].replace("相同", "相反"))
    text = _replace_once(
        text,
        "关节角的正方向按所绕绝对方向轴的右手定则，不随坐标轴反向而翻转。",
        "关节角的正方向以关节转角正方向的定义为准，不随坐标轴反向而翻转。",
    )
    text = _replace_once(
        text,
        "正方向都与绕所对应绝对正方向的右手定则相同。",
        _inverse_sentence(flips),
    )
    text = _replace_once(
        text,
        "q=(0°, 90°, 0°) 和 q=(30°, 90°, -30°)",
        pair,
    )
    if vertical != "下方":
        text = _replace_once(text, "59.本例任务方向指向工具下方，", f"59.本例任务方向指向工具{vertical}，")
    if lateral != "右侧":
        text = _replace_once(text, "60.本例任务方向指向工具右侧，", f"60.本例任务方向指向工具{lateral}，")
    if angles.split(", ")[2] != "arcsin(1/2)":
        text = _replace_once(
            text,
            "本例工具滚转任务角写成 arcsin(1/2) 或 atan2(1, sqrt(3)) 都对。",
            f"本例工具滚转任务角写成 {angles.split(', ')[2]} 或与之等价的反三角写法都对。",
        )
    text = _replace_once(
        text,
        "默认顺序下的提取式是 atan2(-h31, h11); atan2(h21, sqrt(h11^2 + h31^2)); atan2(-h23, h22)。",
        f"本样本的提取式是 {formula}。",
    )
    forward, inverse = text.split("# 逆运动学样本", 1)
    inverse, tail = inverse.split("# 层间对应", 1)
    forward = _apply_slots(forward, "F", updates)
    inverse = _apply_slots(inverse, "I", updates)
    return forward + "# 逆运动学样本" + inverse + "# 层间对应" + tail


def _check_written(path: Path, flips: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    forward, inverse = text.split("# 逆运动学样本", 1)
    inverse = inverse.split("# 层间对应", 1)[0]
    for prefix, section, last in (("F", forward, 140), ("I", inverse, 16)):
        found = [int(match.group(1)) for match in SLOT.finditer(section)]
        if found != list(range(1, last + 1)):
            raise RuntimeError(f"{path.name} {prefix} 编号不是 1–{last}")
    for name, node in JOINT.items():
        number = node[1:]
        match = re.search(rf"\{{ID:{number}, 选项:(.*?), 输出格式:无, 标签:C1, 答案:(.)\}}", forward)
        if match is None:
            raise RuntimeError(f"{path.name} 缺少 {node}")
        options = _options(match.group(1))
        letter = match.group(2)
        expected = "相反" if name in flips else "相同"
        if options[ord(letter) - ord("A")] != expected:
            raise RuntimeError(f"{path.name} {node} 不是{expected}")


def main() -> None:
    if not SOURCE.is_file():
        raise RuntimeError(f"缺少 {SOURCE}")
    # The unflipped matrices must reproduce the published expressions and angles.
    base = _cascade(())
    r1, r2, r3 = _rotations(base)
    pose = _pose(r1, r2, r3)
    alpha, beta, gamma = _task_angles(pose)
    published = ", ".join(_angle_text(item) for item in (alpha, beta, gamma))
    if published != gold_texts()["F65"]:
        raise RuntimeError(f"统一默认的任务角写成了 {published}")
    if _positions(r1, r2) != {node: gold_texts()[node] for node in ("F118", "F119", "F124", "F125")}:
        raise RuntimeError("统一默认的位置表达式没有还原")
    _check_formula(r1, r2, r3, _formula(-1, 1, -1))
    for sample_id, title, flips in VARIANTS:
        text = build_text(sample_id, title, flips)
        path = OUT / f"sample_{sample_id}.md"
        path.write_text(text, encoding="utf-8")
        _check_written(path, flips)
        print(f"wrote {path.name}  {title}", flush=True)


if __name__ == "__main__":
    main()
