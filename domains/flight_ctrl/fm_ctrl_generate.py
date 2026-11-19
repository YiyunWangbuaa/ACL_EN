# -*- coding: utf-8 -*-
"""FM-CTRL：飞行力学对照。地面系 →（机构角 俯仰/偏航/滚转）→ 非正常机体系 → 重算实际迎角/侧滑角。

风洞风轴系对应这里的地面系，体轴系对应机体系。机构角只表示机体系相对地面系的姿态。
逐空对齐风洞「风轴系到体轴系」机制段（63 空），去掉天平/支杆/安装/弹性/气流偏角。
build() 是按档位复制的对照组，samples.jsonl 用它。
build_synced() 按风洞样本同步地面系、机体系以及理论角和机构角的正方向。
"""
from __future__ import annotations

import sympy as sp

AM, BM, GM = sp.symbols("α_M β_M γ_M")

FORWARD = (1, 0, 0)
RIGHT = (0, 1, 0)
DOWN = (0, 0, 1)
UP = (0, 0, -1)
LEFT = (0, -1, 0)
DIRECTIONS = {"forward": FORWARD, "rearward": (-1, 0, 0), "right": RIGHT,
              "left": LEFT, "down": DOWN, "up": UP}
DIR_CN = {"forward": "前方", "rearward": "后方", "right": "右侧",
          "left": "左侧", "down": "下方", "up": "上方"}
SIX = ["上方", "下方", "左侧", "右侧", "前方", "后方"]
FRAME_OPTIONS = ["机体系", "气流系", "地面系"]
AXIS_OPTIONS = ["Xg", "Yg", "Zg", "xb", "yb", "zb", "Xa", "Ya", "Za"]
_DIRECTED_NAMES = ["Xg", "Yg", "Zg", "xb", "yb", "zb", "Xa", "Ya", "Za"]
DIRECTED_OPTIONS = _DIRECTED_NAMES + [f"-{name}" for name in _DIRECTED_NAMES]
CONVERSION_FRAMES = ["机体系", "气流系", "地面系"]
CONVERSION_PAIRS = [f"{source}向{target}" for source in CONVERSION_FRAMES for target in CONVERSION_FRAMES]
_WIND_AXIS_SHOW = {
    "XFD": "Xg", "YFD": "Yg", "ZFD": "Zg",
    "XTZ": "xb", "YTZ": "yb", "ZTZ": "zb",
    "XTP": "Xa", "YTP": "Ya", "ZYP": "Za",
}
FACTOR_OPTIONS = ["RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M"]
PROCESS = "机体系相对地面系"
_ORDER_LABEL = {"pitch": "俯仰角 α_M", "yaw": "偏航角 β_M", "roll": "滚转角 γ_M"}


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def rot(role, t, normal):
    """Passive coordinate transform about the role axis: p_new = R * p_old.

    Positive about the second axis is [[c, 0, -s], [0, 1, 0], [s, 0, c]].
    ``normal`` is unused; the angle sign already follows the declared axis.
    """
    del normal
    c, s = sp.cos(t), sp.sin(t)
    if role == "axial":
        return sp.Matrix([[1, 0, 0], [0, c, s], [0, -s, c]])
    if role == "lateral":
        return sp.Matrix([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    return sp.Matrix([[c, s, 0], [-s, c, 0], [0, 0, 1]])


def sense(axis, moving, toward):
    return 1 if dot(cross(axis, moving), toward) > 0 else -1


def tier_spec(tier):
    s = {"normal": "down", "order": ["pitch", "yaw", "roll"],
         "attack": "up", "sideslip": "right", "bank": "down"}
    # attack=up 表示「实际迎角>0 = 机头向上（空速在机体下方）」
    if tier == 50:
        s["order"] = ["yaw", "pitch", "roll"]
    if tier == 75:
        s["attack"] = "down"; s["sideslip"] = "left"
    if tier == 100:
        s["normal"] = "up"; s["order"] = ["roll", "pitch", "yaw"]; s["bank"] = "up"
    return s


def build(tier):
    s = tier_spec(tier)
    normal = DIRECTIONS[s["normal"]]
    lateral = RIGHT if s["normal"] == "down" else LEFT
    axial = FORWARD
    angles = {
        "pitch": {"name": "俯仰角 α_M", "sym": AM, "role": "lateral",
                  "moving": axial, "toward": DIRECTIONS[s["attack"]]},
        "yaw": {"name": "偏航角 β_M", "sym": BM, "role": "normal",
                "moving": axial, "toward": DIRECTIONS[s["sideslip"]]},
        "roll": {"name": "滚转角 γ_M", "sym": GM, "role": "axial",
                 "moving": lateral, "toward": DIRECTIONS[s["bank"]]},
    }
    role_axis = {"lateral": lateral, "normal": normal, "axial": axial}
    role_cn = {"lateral": "侧向", "normal": "法向", "axial": "轴向"}
    role_coord = {"lateral": "yb", "normal": "zb", "axial": "xb"}
    for key in angles:
        a = angles[key]
        a["sense"] = sense(role_axis[a["role"]], a["moving"], a["toward"])
        a["matrix"] = rot(a["role"], a["sense"] * a["sym"], s["normal"])
    # 总矩阵：先转者矩阵最右
    R = sp.eye(3)
    for key in reversed(s["order"]):
        R = R * angles[key]["matrix"]
    R = sp.simplify(R)
    XTZ, YTZ, ZTZ = R[0, 0], R[1, 0], R[2, 0]  # 空速单位向量在机体系 = R_t 第一列
    alpha = sp.atan2(-ZTZ, XTZ)
    beta = sp.asin(YTZ)

    # 各角>0 时，空速/右翼偏机体的方向（用于方向判断题）
    def airspeed_bias(key):
        # 仅该机构角>0 时，空速在机体系中的方向 = R_key 的第一列
        m = angles[key]["matrix"].subs({AM: 1 if key == "pitch" else 0,
                                        BM: 1 if key == "yaw" else 0,
                                        GM: 1 if key == "roll" else 0})
        col = [m[0, 0], m[1, 0], m[2, 0]]
        # 判断非零主方向（相对 axial）
        best = "前方"
        bestv = -1
        for dname, vec in DIRECTIONS.items():
            if dname in ("forward", "rearward"):
                continue
            v = dot(col, vec)
            if v > bestv:
                bestv, best = v, DIR_CN[dname]
        return best

    ground_rows = _clone_rows("tunnel", [
        ("Xg", "axial", "forward", "fixed"),
        ("Yg", "lateral", "right" if s["normal"] == "down" else "left", "inferred"),
        ("Zg", "normal", s["normal"], "declared"),
    ])
    body_rows = _clone_rows("body", [
        ("Xb", "axial", "forward", "declared"),
        ("Yb", "lateral", "right" if s["normal"] == "down" else "left", "inferred"),
        ("Zb", "normal", s["normal"], "declared"),
    ])
    qs = _prefix_questions(ground_rows, body_rows)

    def c(prompt, options, correct, label):
        letters = "ABCDEFGHIJKLMNOP"
        return {"prompt": prompt, "label": label, "format_code": "无",
                "options": list(options), "answer": letters[options.index(correct)]}

    def m(prompt, mat, label="A2"):
        return _matrix_item(prompt, mat, label)

    def e(prompt, expr, label="A2"):
        return _expr_item(prompt, expr, label)

    cn = {"pitch": "迎角", "yaw": "侧滑角", "roll": "滚转角"}
    for i in range(3):
        qs.append(c(f"第{i + 1}个旋转角为？", ["迎角", "侧滑角", "滚转角"], cn[s["order"][i]], "A1"))

    # 句子 4/5：α_M、β_M（各 8 空）
    for key in ("pitch", "yaw"):
        a = angles[key]
        bias = airspeed_bias(key)
        theory_bias = DIR_CN[{v: k for k, v in DIRECTIONS.items()}[a["toward"]]]  # 实际角定义方向
        qs.append(c("模型采用？", ["正装", "倒装"], "正装", "C1"))
        qs.append(c(f"机构{a['name']}>0 时，空速偏机体的哪一方？", SIX, bias, "A1"))
        qs.append(c(f"实际{'迎角' if key=='pitch' else '侧滑角'}>0 时，空速偏机体的哪一方？", SIX, theory_bias, "C1"))
        qs.append(c(f"机构{a['name']}>0 时，实际{'迎角' if key=='pitch' else '侧滑角'}的符号为？",
                    ["正", "负", "不确定"], "正" if a["sense"] > 0 else "负", "A1"))
        qs.append(c(f"这相当于哪个坐标系平正时绕轴旋转？", FRAME_OPTIONS, "机体系", "C2"))
        qs.append(c(f"绕机体系的哪个坐标轴旋转？", AXIS_OPTIONS, role_coord[a["role"]], "C1"))
        qs.append(c("绕该轴旋转方向为？", ["正", "负", "不确定"], "正" if a["sense"] > 0 else "负", "A1"))
        qs.append(m(f"对应的旋转矩阵 R_{a['sym']} 为？", a["matrix"]))

    # 句子 5：γ_M（9 空）
    a = angles["roll"]
    wing_bias = airspeed_bias("roll")  # 近似：右翼方向
    theory_bias = DIR_CN[{v: k for k, v in DIRECTIONS.items()}[a["toward"]]]
    qs.append(c("以模型绝对轴向正方向为旋转轴，与右手定则？", ["相同", "相反"],
                "相同" if a["sense"] > 0 else "相反", "C1"))
    qs.append(c("此时右翼偏机体的哪一方？", SIX, wing_bias, "A1"))
    qs.append(c("实际滚转角>0 时，以轴向为旋转轴与右手定则？", ["相同", "相反"],
                "相同" if a["sense"] > 0 else "相反", "C1"))
    qs.append(c("此时右翼偏机体的哪一方？", SIX, theory_bias, "A1"))
    qs.append(c("机构滚转角>0 时，实际滚转角的符号为？", ["正", "负", "不确定"],
                "正" if a["sense"] > 0 else "负", "A1"))
    qs.append(c("这相当于哪个坐标系平正时绕轴旋转？", FRAME_OPTIONS, "机体系", "C2"))
    qs.append(c("绕机体系的哪个坐标轴旋转？", AXIS_OPTIONS, role_coord[a["role"]], "C1"))
    qs.append(c("绕该轴旋转方向为？", ["正", "负", "不确定"], "正" if a["sense"] > 0 else "负", "A1"))
    qs.append(m("对应的旋转矩阵 R_γ_M 为？", a["matrix"]))

    # 句子 7：变换 + 因子（3 + 5 空）
    qs.append(c("坐标变换采用？", ["主动", "被动"], "被动", "C2"))
    qs.append(c("机体系是？", ["动轴", "定轴"], "动轴", "C2"))
    qs.append(c("旋转采用？", ["内旋", "外旋"], "内旋", "C2"))
    for index, answer in enumerate(_factor_answers(s["order"]), start=1):
        qs.append(c(f"公式中对应的R{index}矩阵为？", FACTOR_OPTIONS, answer, "A2"))
    # 句子 8：先写 R1、R2、R3 的展开式，再写总矩阵
    for index, key in enumerate(reversed(s["order"]), start=1):
        qs.append(m(f"总矩阵前 R{index} 的展开式为？", angles[key]["matrix"]))
    qs.append(m("写出总矩阵 R_t（符号化简）。", R))
    # 句子 9：空速单位向量（2 空）。档位样本的两套坐标轴在初始状态重合，前置矩阵为单位阵。
    qs.append(_vector_item("RP2*RP1乘地面系中的空速单位向量，结果一定是？", [1, 0, 0]))
    qs.append(c(
        "因此只需要研究哪个坐标系中的空速坐标到姿态角变化后的机体系的坐标映射？",
        ["平置机体系", "地面系"],
        "平置机体系",
        "C2",
    ))
    # 句子 10：分量（4 空）
    qs.append(_vector_item("此时平置坐标系中的空速单位向量[X_V,Y_V,Z_V]是？", [1, 0, 0]))
    qs.append(e("代入计算得到的 xb 分量为？", XTZ))
    qs.append(e("代入计算得到的 yb 分量为？", YTZ))
    qs.append(e("代入计算得到的 zb 分量为？", ZTZ))
    # 句子 11/12：符号判断（6 空，仅 α_M>0 与仅 β_M>0）
    for key, nm in (("pitch", "α_M"), ("yaw", "β_M")):
        subs = {AM: 1 if key == "pitch" else 0, BM: 1 if key == "yaw" else 0, GM: 0}
        Xs, Ys, Zs = R.subs(subs)[0, 0], R.subs(subs)[1, 0], R.subs(subs)[2, 0]
        for val in (Xs, Ys, Zs):
            qs.append(c(f"仅 {nm}>0 时，该分量符号为？", ["正", "负", "零"], _sign_word(val), "A1"))
    # 句子 13/14：α/β 公式符号（各 6 空，简化）
    for nm, expr, comp in (("α", alpha, None), ("β", beta, None)):
        qs.append(c(f"{nm} 的角度符号为？", ["正", "负"], "正", "A1"))
        qs.append(c(f"{nm} 的反三角函数为？", ["arctan", "arcsin"], "arctan" if nm == "α" else "arcsin", "A1"))
        qs.append(c(f"{nm} 分子符号为？", ["正", "负"], "正", "A1"))
        qs.append(c(f"{nm} 分子分量为？", ["zb", "yb", "xb", "1"], _num_component(nm), "A1"))
        qs.append(c(f"{nm} 分母符号为？", ["正", "负"], "正", "A1"))
        qs.append(c(f"{nm} 分母分量为？", ["zb", "yb", "xb", "1"], _den_component(nm), "A1"))
    # 句子 15：最终表达式（2 空）
    qs.append(e("求得实际迎角 α 为？", alpha))
    qs.append(e("求得实际侧滑角 β 为？", beta))

    for i, item in enumerate(qs, start=1):
        item["local_id"] = i
        item["id"] = str(i)
        item["process"] = PROCESS
        item["corresponds_to"] = str(i)
    return {
        "sample_id": f"fmctrl-{tier:03d}", "tier": tier, "spec": s,
        "group": "control",  # 对照组标记
        "declaration": _decl(s), "questions": qs,
    }


def _sign_word(expr):
    v = sp.simplify(expr)
    if v == 0:
        return "零"
    return "正" if v.subs({AM: 0.1, BM: 0.1, GM: 0.1}) > 0 else "负"


def _body_component(name):
    """风洞体轴分量名换成机体系分量名。1 表示分母取 1。"""
    return {"XTZ": "xb", "YTZ": "yb", "ZTZ": "zb", "1": "1"}[name]


def _num_component(nm):
    return _body_component("ZTZ" if nm == "α" else "YTZ")


def _den_component(nm):
    return _body_component("XTZ" if nm == "α" else "1")


def _factor_answers(order) -> list[str]:
    """连乘从左到右是 R1 到 R5。最后旋转的姿态矩阵在最左，RP1 在最右。"""
    code = {"pitch": "R_α_M", "yaw": "R_β_M", "roll": "R_γ_M"}
    named = [code[key] for key in order]
    if len(named) != 3:
        raise RuntimeError(f"旋转顺序必须是三项：{order}")
    return [named[2], named[1], named[0], "RP2", "RP1"]


def _prematrix_lines() -> list[str]:
    return [
        "- 坐标系转换采用被动变换。姿态旋转采用动轴内旋。",
        "- 必须先变换坐标轴对应关系，之后再变换姿态。写成 [xb,yb,zb]^T=R1*R2*R3*R4*R5*[Xg,Yg,Zg]^T。连乘从右向左先写前置矩阵，再写机构角矩阵。最先旋转的机构角，其矩阵紧挨着前置矩阵。",
        "- 地面系向机体系、平正位置处的前置变换矩阵是 RP1。该矩阵使机体系的坐标轴与自身绝对方向的关系，与地面系的坐标轴与自身绝对方向的关系一致。",
        "- 安装只考虑正装。正装时，平正状态下的机体上方和地面上方一致。此时机体绝对法向正方向（上方）和绝对侧向正方向（右侧）与地面的绝对法向正方向和绝对侧向正方向相同。如果反装时机体上方与地面上方相反，所以还需要另一个前置变换矩阵 RP2。该矩阵使机体系的坐标轴方向与地面系的坐标方向相同。正装时 RP2 为单位阵。",
    ]


def _show_directed(token: str) -> str:
    sign = "-" if token.startswith("-") else ""
    raw = token[1:] if sign else token
    return sign + _WIND_AXIS_SHOW.get(raw, _BODY_SHOW.get(raw, raw))


def _row_vector(row: dict) -> tuple[int, int, int]:
    return {
        "forward": (1, 0, 0), "rearward": (-1, 0, 0),
        "right": (0, 1, 0), "left": (0, -1, 0),
        "down": (0, 0, 1), "up": (0, 0, -1),
    }[row["points"].split("_", 1)[1]]


def _directed_name(rows: list[dict], side: str) -> str:
    target = {
        "forward": (1, 0, 0), "right": (0, 1, 0), "up": (0, 0, -1),
    }[side]
    for row in rows:
        vector = _row_vector(row)
        if vector == target:
            return _show_directed(row["name"])
        if tuple(-item for item in vector) == target:
            return "-" + _show_directed(row["name"])
    raise RuntimeError(f"没有坐标轴指向{side}：{rows}")


def _alignment_matrix(ground: list[dict], body: list[dict]):
    return sp.Matrix([
        [int(sum(a * b for a, b in zip(_row_vector(body_row), _row_vector(ground_row))))
         for ground_row in ground]
        for body_row in body
    ])


def _prefix_questions(ground: list[dict], body: list[dict]) -> list[dict]:
    """风洞开头的坐标对齐段。选项改成地面系、机体系、气流系及其坐标轴。安装固定正装，RP2 为单位阵。"""
    alignment = _alignment_matrix(ground, body)
    mount = sp.eye(3)
    items = [
        _choice("应当进行哪种转换？", CONVERSION_PAIRS, "地面系向机体系", "C2"),
    ]
    for side in ("up", "forward", "right"):
        items.append(_choice("地面绝对方向对应的有向坐标轴？", DIRECTED_OPTIONS, _directed_name(ground, side), "C1"))
    for side in ("up", "forward", "right"):
        items.append(_choice("机体绝对方向对应的有向坐标轴？", DIRECTED_OPTIONS, _directed_name(body, side), "C1"))
    items.extend([
        _matrix_item("地面系向体轴系平正位置处的前置变换矩阵 RP1 为？", alignment),
        _choice("该矩阵对齐的坐标系？", CONVERSION_FRAMES, "机体系", "C2"),
        _choice("该矩阵对照的坐标系？", CONVERSION_FRAMES, "地面系", "C2"),
        _choice("模型安装方式为？", ["正装", "反装"], "正装", "C1"),
        _choice("模型与地面的绝对法向、侧向是否相同？", ["相同", "相反"], "相同", "A1"),
        _matrix_item("前置变换矩阵 RP2 为？", mount),
        _choice("RP2 对齐的坐标系？", CONVERSION_FRAMES, "机体系", "C2"),
        _choice("RP2 对照的坐标系？", CONVERSION_FRAMES, "地面系", "C2"),
    ])
    if len(items) != 15:
        raise RuntimeError(f"前置段应为 15 空，实际 {len(items)}。")
    return items


def _choice(prompt, options, correct, label):
    letters = "ABCDEFGHIJKLMNOP"
    return {"prompt": prompt, "label": label, "format_code": "无",
            "options": list(options), "answer": letters[list(options).index(correct)]}


def _matrix_item(prompt, mat, label="A2"):
    from domains.wind_tunnel.tz_fd_generate import _matrix_text

    return {"prompt": prompt, "label": label, "format_code": "矩阵", "options": [],
            "answer": _matrix_text(sp.Matrix(mat))}


def _vector_item(prompt, values, label="A1"):
    from domains.wind_tunnel.tz_fd_generate import _vector_text

    return {"prompt": prompt, "label": label, "format_code": "向量", "options": [],
            "answer": _vector_text(values)}


def _expr_item(prompt, expr, label="A2", format_code="表达式"):
    from domains.wind_tunnel.tz_fd_generate import _text

    answer = expr if isinstance(expr, str) else _text(expr)
    return {"prompt": prompt, "label": label, "format_code": format_code, "options": [],
            "answer": answer}


def build_synced(wind_spec: dict, *, source_id: str, tier: int) -> dict:
    """对照组一条样本。地面系对齐风轴系，机体系对齐体轴系，五个角度再加理论滚转。

    安装方式固定正装，不把风洞反装带进来。欧拉顺序保持迎角、侧滑角、滚转角。
    """
    import copy

    from body_frame_model import (
        ALPHA,
        ALPHA_M,
        BETA,
        BETA_M,
        BODY_AXES,
        FRONT,
        GAMMA,
        GAMMA_M,
        WIND_VOCAB,
        BodyFrameModel,
        DefinitionError,
    )
    from domains.wind_tunnel.tz_fd_generate import (
        ALPHA_SYM,
        BETA_SYM,
        GAMMA_SYM,
        ROTATION_ZH,
        STANDARD_SPEC,
        _apply_sign,
        _level_airspeed,
        _magnitude_sign,
        _mechanism_matrix,
        _require_positive_part,
        _same_sense,
        _signed_component,
    )

    spec = copy.deepcopy(STANDARD_SPEC)
    for key in ("alpha", "beta", "gamma", "alpha_m", "beta_m", "gamma_m", "body_frame", "wind_frame"):
        spec[key] = copy.deepcopy(wind_spec[key])
    spec["placement"] = "upright"
    if spec["body_frame"].get("euler_order") != ["alpha", "beta", "gamma"]:
        raise DefinitionError(f"{source_id} 的欧拉顺序不是迎角、侧滑角、滚转角。")
    model = BodyFrameModel().declare(spec)
    matrices = {
        ALPHA_M: _display(_mechanism_matrix(model, ALPHA_M)),
        BETA_M: _display(_mechanism_matrix(model, BETA_M)),
        GAMMA_M: _display(_mechanism_matrix(model, GAMMA_M)),
    }
    total = sp.simplify(
        _mechanism_matrix(model, GAMMA_M)
        * _mechanism_matrix(model, BETA_M)
        * _mechanism_matrix(model, ALPHA_M)
    )
    _wind_vector, level = _level_airspeed(model)
    posed = sp.simplify(total * level)
    alpha_formula = _inverse_formula(
        model, posed, ALPHA, ALPHA_M, ALPHA_SYM, "arctan", _signed_component(model, FRONT)[0],
    )
    beta_formula = _inverse_formula(
        model, posed, BETA, BETA_M, BETA_SYM, "arcsin", "1",
    )
    shown = _display(posed)
    shown_total = _display(total)

    ground = _frame_rows(model.wind_frame, WIND_VOCAB.coordinate_axes, {"XFD": "Xg", "YFD": "Yg", "ZFD": "Zg"})
    body = _frame_rows(model.body_frame, BODY_AXES, {"XTZ": "Xb", "YTZ": "Yb", "ZTZ": "Zb"})
    def direction_answer(record, field: str) -> str:
        return _DIR[getattr(record, field).split("_", 1)[1]]

    def rotation_answer(record) -> str:
        if record.rotation_when_positive not in ROTATION_ZH:
            raise DefinitionError(f"{source_id} 的 {record.name} 没有确定的旋转方向。")
        return ROTATION_ZH[record.rotation_when_positive]

    def axis_answer(record) -> str:
        return {"XTZ": "xb", "YTZ": "yb", "ZTZ": "zb"}[record.associated_axis]

    pitch = model.mechanism_angles[ALPHA_M]
    yaw = model.mechanism_angles[BETA_M]
    roll = model.mechanism_angles[GAMMA_M]
    theory = model.angles
    blocks = (
        ("俯仰角 α_M", "迎角", pitch, theory[ALPHA], matrices[ALPHA_M], "positive_airspeed_toward"),
        ("偏航角 β_M", "侧滑角", yaw, theory[BETA], matrices[BETA_M], "positive_airspeed_toward"),
    )
    qs = _prefix_questions(ground, body)
    for index, name in enumerate(("迎角", "侧滑角", "滚转角"), start=1):
        qs.append(_choice(f"第{index}个旋转角为？", ["迎角", "侧滑角", "滚转角"], name, "A1"))
    for name, theory_name, mechanism, theory_record, matrix, field in blocks:
        outer = _same_sense(getattr(mechanism, field), getattr(theory_record, field))
        qs.append(_choice("模型采用？", ["正装", "倒装"], "正装", "C1"))
        qs.append(_choice(f"机构{name}>0 时，空速偏机体的哪一方？", SIX, direction_answer(mechanism, field), "A1"))
        qs.append(_choice(f"实际{theory_name}>0 时，空速偏机体的哪一方？", SIX, direction_answer(theory_record, field), "C1"))
        qs.append(_choice(f"机构{name}>0 时，实际{theory_name}的符号为？", ["正", "负", "不确定"], outer, "A1"))
        qs.append(_choice("这相当于哪个坐标系平正时绕轴旋转？", FRAME_OPTIONS, "机体系", "C2"))
        qs.append(_choice("绕机体系的哪个坐标轴旋转？", AXIS_OPTIONS, axis_answer(mechanism), "C1"))
        qs.append(_choice("绕该轴旋转方向为？", ["正", "负", "不确定"], rotation_answer(mechanism), "A1"))
        qs.append(_matrix_item(f"对应的旋转矩阵 R_{mechanism.symbol} 为？", matrix))

    roll_outer = _same_sense(roll.right_wing_toward, theory[GAMMA].right_wing_toward)
    qs.append(_choice("以模型绝对轴向正方向为旋转轴，与右手定则？", ["相同", "相反"], _roll_hand(roll.right_wing_toward), "C1"))
    qs.append(_choice("此时右翼偏机体的哪一方？", SIX, direction_answer(roll, "right_wing_toward"), "A1"))
    qs.append(_choice("实际滚转角>0 时，以轴向为旋转轴与右手定则？", ["相同", "相反"], _roll_hand(theory[GAMMA].right_wing_toward), "C1"))
    qs.append(_choice("此时右翼偏机体的哪一方？", SIX, direction_answer(theory[GAMMA], "right_wing_toward"), "A1"))
    qs.append(_choice("机构滚转角>0 时，实际滚转角的符号为？", ["正", "负", "不确定"], roll_outer, "A1"))
    qs.append(_choice("这相当于哪个坐标系平正时绕轴旋转？", FRAME_OPTIONS, "机体系", "C2"))
    qs.append(_choice("绕机体系的哪个坐标轴旋转？", AXIS_OPTIONS, axis_answer(roll), "C1"))
    qs.append(_choice("绕该轴旋转方向为？", ["正", "负", "不确定"], rotation_answer(roll), "A1"))
    qs.append(_matrix_item("对应的旋转矩阵 R_γ_M 为？", matrices[GAMMA_M]))

    qs.append(_choice("坐标变换采用？", ["主动", "被动"], "被动", "C2"))
    qs.append(_choice("机体系是？", ["动轴", "定轴"], "动轴", "C2"))
    qs.append(_choice("旋转采用？", ["内旋", "外旋"], "内旋", "C2"))
    synced_order = ("pitch", "yaw", "roll")
    synced_matrix = {"pitch": ALPHA_M, "yaw": BETA_M, "roll": GAMMA_M}
    for index, answer in enumerate(_factor_answers(synced_order), start=1):
        qs.append(_choice(f"公式中对应的R{index}矩阵为？", FACTOR_OPTIONS, answer, "A2"))
    for index, key in enumerate(reversed(synced_order), start=1):
        qs.append(_matrix_item(f"总矩阵前 R{index} 的展开式为？", matrices[synced_matrix[key]]))
    qs.append(_matrix_item("写出总矩阵 R_t（符号化简）。", shown_total))
    qs.append(_vector_item("RP2*RP1乘地面系中的空速单位向量，结果一定是？", level))
    qs.append(_choice(
        "因此只需要研究哪个坐标系中的空速坐标到姿态角变化后的机体系的坐标映射？",
        ["平置机体系", "地面系"],
        "平置机体系",
        "C2",
    ))
    qs.append(_vector_item("此时平置坐标系中的空速单位向量[X_V,Y_V,Z_V]是？", level))
    for index, axis_name in enumerate(("xb", "yb", "zb")):
        qs.append(_expr_item(f"代入计算得到的 {axis_name} 分量为？", shown[index]))
    for symbol, active in (("α_M", AM), ("β_M", BM)):
        quiet = {AM: 0, BM: 0, GM: 0}
        quiet.pop(active)
        for index, axis_name in enumerate(("xb", "yb", "zb")):
            qs.append(_choice(
                f"仅 {symbol}>0 时，空速在机体系的 {axis_name} 分量符号为？",
                ["正", "负", "零"],
                _component_sign(shown[index].subs(quiet), active),
                "A1",
            ))
    for name, formula in (("α", alpha_formula), ("β", beta_formula)):
        qs.append(_choice(f"{name} 的角度符号为？", ["正", "负"], formula["outer"], "A1"))
        qs.append(_choice(f"{name} 的反三角函数为？", ["arctan", "arcsin"], formula["inverse"], "A1"))
        qs.append(_choice(f"{name} 分子符号为？", ["正", "负"], formula["numerator_sign"], "A1"))
        qs.append(_choice(f"{name} 分子分量为？", ["zb", "yb", "xb", "1"], _body_component(formula["numerator"]), "A1"))
        qs.append(_choice(f"{name} 分母符号为？", ["正", "负"], formula["denominator_sign"], "A1"))
        qs.append(_choice(f"{name} 分母分量为？", ["zb", "yb", "xb", "1"], _body_component(formula["denominator"]), "A1"))
    qs.append(_expr_item("求得实际迎角 α 为？", alpha_formula["answer"], format_code="arctan"))
    qs.append(_expr_item("求得实际侧滑角 β 为？", beta_formula["answer"], format_code="arcsin"))
    if len(qs) != 81:
        raise DefinitionError(f"{source_id} 生成了 {len(qs)} 空，应为 81。")
    for index, item in enumerate(qs, start=1):
        item["local_id"] = index
        item["id"] = str(index)
        item["process"] = PROCESS
        item["corresponds_to"] = str(index)
    return {
        "sample_id": f"fmctrl-sync-{source_id}",
        "id": f"fmctrl-sync-{source_id}",
        "tier": tier,
        "title": f"飞行力学对照（同步风洞 {source_id}）",
        "group": "control",
        "spec": {
            "source_id": source_id,
            "placement": "upright",
            "alpha": spec["alpha"],
            "beta": spec["beta"],
            "gamma": spec["gamma"],
            "alpha_m": spec["alpha_m"],
            "beta_m": spec["beta_m"],
            "gamma_m": spec["gamma_m"],
            "body_frame": spec["body_frame"],
            "wind_frame": spec["wind_frame"],
            "ground_axes": ground,
            "body_axes": body,
        },
        "declaration": _decl_synced(model),
        "questions": qs,
    }


def build_standard_synced() -> dict:
    """同步对照组使用的标准定义：风洞标准风轴、体轴和五个角度。"""
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

    return build_synced(STANDARD_SPEC, source_id="standard", tier=0)


def emit_synced(out_path=None, pilot_path=None) -> list[dict]:
    """按风洞 pilot_10 写 10 条同步对照组。不覆盖 samples.jsonl。"""
    import json
    from pathlib import Path

    from domain_layout import SETS

    wind_batch = SETS["cross.wind_flight"]["batch"]
    source_ids = json.loads((wind_batch / "pilot_10.json").read_text(encoding="utf-8"))["ids"]
    wanted = set(source_ids)
    found = {}
    for line in (wind_batch / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in wanted:
            found[row["id"]] = row
    missing = wanted - set(found)
    if missing:
        raise RuntimeError("风洞样本里没有这些编号：" + "、".join(sorted(missing)))
    rows = [build_synced(found[source_id]["spec"], source_id=source_id, tier=found[source_id]["tier"])
            for source_id in source_ids]
    # 坐标轴以空气动力学元定义为准，不再按样本另写一套轴。角度定义相同的样本可以共用同一份声明。
    destination = Path(out_path) if out_path else SETS["reconstruction.flight"]["synced"]
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    pilot = {
        "ids": [row["id"] for row in rows],
        "source_ids": source_ids,
        "source": "data/wind_tunnel/batch/pilot_10.json",
        "placement": "upright",
        "synced": [
            "wind_frame -> ground",
            "body_frame -> body",
            "alpha",
            "beta",
            "gamma",
            "alpha_m",
            "beta_m",
            "gamma_m",
        ],
    }
    pilot_destination = Path(pilot_path) if pilot_path else SETS["reconstruction.flight"]["root"] / "pilot_synced.json"
    pilot_destination.write_text(json.dumps(pilot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def _magnitude_text(expr) -> str:
    """把已经取正的分子或分母写成不以内置负号开头的式子。"""
    from domains.wind_tunnel.tz_fd_generate import _text

    simplified = sp.expand(sp.simplify(expr))
    if not isinstance(simplified, sp.Add):
        return _text(simplified)
    positive, negative = [], []
    for term in simplified.args:
        rendered = _text(term)
        if rendered.startswith("-"):
            negative.append(rendered[1:])
        else:
            positive.append(rendered)
    if not positive:
        return _text(simplified)
    parts = [positive[0]]
    for item in positive[1:]:
        parts.append("+ " + item)
    for item in negative:
        parts.append("- " + item)
    return " ".join(parts)


def _display(matrix):
    from domains.wind_tunnel.tz_fd_generate import ALPHA_SYM, BETA_SYM, GAMMA_SYM

    return sp.simplify(matrix).subs({ALPHA_SYM: AM, BETA_SYM: BM, GAMMA_SYM: GM})


def _inverse_formula(model, posed, theory_name, mechanism_name, active, inverse, denominator):
    from body_frame_model import DefinitionError
    from domains.wind_tunnel.tz_fd_generate import (
        _apply_sign,
        _magnitude_sign,
        _require_positive_part,
        _same_sense,
        _signed_component,
    )

    theory = model.angles[theory_name]
    mechanism = model.mechanism_angles[mechanism_name]
    field = "right_wing_toward" if theory_name == "gamma" else "positive_airspeed_toward"
    toward = getattr(theory, field)
    numerator, _sense = _signed_component(model, toward)
    signs = mechanism.airspeed_signs if field == "positive_airspeed_toward" else mechanism.component_signs
    numerator_sign = _magnitude_sign(signs, numerator, theory.symbol)
    if denominator == "1":
        denominator_sign = "正"
        denominator_expr = sp.Integer(1)
    else:
        denominator_sign = _magnitude_sign(signs, denominator, theory.symbol)
        denominator_expr = posed[{"XTZ": 0, "YTZ": 1, "ZTZ": 2}[denominator]]
    numerator_expr = _apply_sign(numerator_sign, posed[{"XTZ": 0, "YTZ": 1, "ZTZ": 2}[numerator]])
    denominator_expr = _apply_sign(denominator_sign, denominator_expr)
    _require_positive_part(numerator_expr, active, f"{theory.symbol} numerator")
    _require_positive_part(denominator_expr, active, f"{theory.symbol} denominator")
    outer = _same_sense(getattr(mechanism, field), toward)
    if outer not in ("正", "负"):
        raise DefinitionError(f"{theory.symbol} 的外层符号无法确定。")
    from domains.wind_tunnel.tz_fd_generate import _signed_formula

    if inverse == "arctan":
        interior = f"({_magnitude_text(numerator_expr)})/({_magnitude_text(denominator_expr)})"
    elif inverse == "arcsin":
        interior = _magnitude_text(numerator_expr)
    else:
        raise DefinitionError(f"未知的反三角函数：{inverse}")
    return {
        "outer": outer,
        "inverse": inverse,
        "numerator_sign": numerator_sign,
        "numerator": numerator,
        "denominator_sign": denominator_sign,
        "denominator": denominator,
        "answer": _signed_formula(outer, inverse, interior),
    }


def _frame_rows(frame, names, shorts) -> list[dict]:
    rows = []
    for name in names:
        axis = frame[name]
        rows.append({
            "name": shorts[name],
            "code": name,
            "along": axis.absolute_axis,
            "points": axis.points_to,
            "source": axis.source,
        })
    return rows


def _component_sign(expr, active) -> str:
    value = sp.simplify(expr)
    if value == 0:
        return "零"
    number = sp.simplify(value.subs({AM: 0, BM: 0, GM: 0, active: sp.Rational(1, 10)}))
    if number == 0:
        return "零"
    if number > 0:
        return "正"
    if number < 0:
        return "负"
    raise RuntimeError(f"分量符号无法判断：{value}")


def _roll_hand(right_wing_toward: str) -> str:
    from domains.wind_tunnel.tz_fd_generate import _declared_roll_hand

    return _declared_roll_hand(right_wing_toward)


_DIR = {
    "forward": "前方", "rearward": "后方", "up": "上方", "down": "下方",
    "left": "左侧", "right": "右侧",
}
_ROLE = {"axial": "轴向", "normal": "法向", "lateral": "侧向"}
_SOURCE = {"declared": "声明", "inferred": "由手性推断", "fixed": "固定"}
_BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
def _toward_phrase(direction: str) -> str:
    owner, side = direction.split("_", 1)
    who = "机体" if owner == "body" else "地面"
    return f"{who}{_DIR[side]}"


def _nose_word(airspeed_toward: str) -> str:
    """机头偏向与空速偏向相反，从而和风洞的空速正方向等价。"""
    return {
        "body_down": "上偏",
        "body_up": "下偏",
        "body_left": "右",
        "body_right": "左",
    }[airspeed_toward]


def _wing_word(right_wing_toward: str) -> str:
    return {"body_down": "下方", "body_up": "上方"}[right_wing_toward]


def _mechanism_lines(alpha_toward: str, beta_toward: str, gamma_toward: str) -> list[str]:
    from body_frame_model import SENSE_SAME, roll_about_nose_sense
    from domains.wind_tunnel.dataset_generate import _balance_deflection

    roll = "相同" if roll_about_nose_sense(gamma_toward) == SENSE_SAME else "相反"
    return [
        "机构迎角 α_M：功角支撑杆在铅垂面上运动，可视为绕地面系的侧向轴运动，为支撑杆（机体轴向）与水平面的夹角。"
        f"机体偏向地面{_balance_deflection(alpha_toward)}时为正。",
        "机构侧滑角 β_M：侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，即可视为侧滑角机构基于功角机构在作动。"
        "β_M是侧滑角支撑杆与铅垂面的夹角。"
        f"使机体偏向地面{_balance_deflection(beta_toward)}时为正。",
        "机构滚转角 γ_M：机体绕侧滑角支撑杆的转角。"
        f"角度为正时，以模型绝对轴向正方向（前方）旋转，与右手定则方向{roll}。",
    ]


def _aero_base() -> str:
    """空气动力学标准声明。对照组只在这份全文上补本任务多出来的内容。"""
    from copy import deepcopy

    from flight_frame_model import STANDARD_SPEC, FlightFrameModel
    from domains.aerodynamics.dataset_generate import _declaration

    text = _declaration(FlightFrameModel().declare(deepcopy(STANDARD_SPEC)))
    text = text.replace(
        "- 坐标系转换采用被动变换。姿态旋转采用动轴内旋。最先旋转的角度，其矩阵写在连乘式最右侧。",
        "- 坐标系转换采用被动变换。姿态旋转、航迹旋转和气流角旋转采用动轴内旋。最先旋转的角度，其矩阵写在连乘式最右侧。",
        1,
    )
    stub = (
        "机构角\n"
        "机构迎角 α_M：功角支撑杆在铅垂面上运动，可视为绕地面系的侧向轴运动，为支撑杆（机体轴向）与水平面的夹角。机体偏向地面上方时为正。\n"
        "机构侧滑角 β_M：侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，即可视为侧滑角机构基于功角机构在作动。β_M是侧滑角支撑杆与铅垂面的夹角。使机体偏向地面右侧时为正。\n"
        "机构滚转角 γ_M：机体绕侧滑角支撑杆的转角。角度为正时，以模型绝对轴向正方向（前方）旋转，与右手定则方向相同。\n\n"
    )
    if stub not in text:
        raise RuntimeError("空气动力学声明里没有机构角占位，无法补入对照组定义。")
    return text.replace(stub, "")


def _special_scene(order) -> str:
    del order
    return (
        "其中有一类特殊任务，初始状态下，机体系与地面系的绝对方向完全重合，并且气流方向与地面系的轴向重合，吹向机头。"
        "机体尾部固定在了一个机械结构装置上，该装置会采取一种全新的欧拉角变换方式将机体变换到一种特殊姿态。"
        "变换的顺序与正常变换顺序不同，必须先理解机构角度的变换特性，然后推断出正确的角度变换顺序。"
        "这些角度的定义方式已在下方的角度定义给出。"
        "这些角度把初始与地面系重合的机体系转到一个特殊的区别于传统姿态角位置的机体系。"
        "在这个位置下，需要再按理论迎角、侧滑角的定义重新计算该位置下的理论气流角。"
    )


def _show_axis(name: str) -> str:
    return _BODY_SHOW.get(name, name)


def _axis_line(row: dict, who: str) -> str:
    role = row["along"].split("_", 1)[1]
    side = _DIR[row["points"].split("_", 1)[1]]
    return (
        f"- {_show_axis(row['name'])}：沿{who}{_ROLE[role]}，"
        f"指向{who}{side}（{_SOURCE[row['source']]}）。"
    )


def _frame_block(title: str, hand: str, rows: list[dict], who: str) -> str:
    return "\n".join([f"{title}（{hand}）", *[_axis_line(row, who) for row in rows]])


def _splice_ground_body(coords: str, ground_block: str, body_block: str) -> str:
    marker = "\n\n气流系"
    if marker not in coords:
        raise RuntimeError("坐标系里没有气流系，无法改写地面系和机体系。")
    return f"{ground_block}\n\n{body_block}{marker}{coords.split(marker, 1)[1]}"


def _clone_rows(owner: str, binding: list[tuple[str, str, str, str]]) -> list[dict]:
    return [
        {"name": name, "along": f"{owner}_{role}", "points": f"{owner}_{side}", "source": source}
        for name, role, side, source in binding
    ]


def _compose_declaration(mechanism_block: str, order, ground: list[dict], body: list[dict], ground_hand: str, body_hand: str) -> str:
    fixed, rest = _aero_base().rstrip("\n").split("\n\n绝对方向\n", 1)
    direction, rest = rest.split("\n\n场景\n", 1)
    scene_body, rest = rest.split("\n\n角度\n", 1)
    angles, coords = rest.split("\n坐标系\n", 1)
    task_rules = [line for line in _prematrix_lines() if "坐标系转换采用被动变换" not in line]
    xg = next(row for row in ground if row["name"] == "Xg")
    role = xg["along"].split("_", 1)[1]
    side = _DIR[xg["points"].split("_", 1)[1]]
    fixed = fixed.replace(
        "- 地面坐标轴 Xg 沿地面轴向，指向地面前方。",
        f"- 地面坐标轴 Xg 沿地面{_ROLE[role]}，指向地面{side}。",
        1,
    )
    axial = next(row for row in ground if row["along"].endswith("axial"))
    axial_name = _show_axis(axial["name"])
    if axial["points"].split("_", 1)[1] == "forward":
        forward = f"正方向朝前，是 {axial_name} 的正方向。"
    else:
        forward = f"正方向朝前，是 {axial_name} 正方向的反方向。"
    direction = direction.replace("正方向朝前，是 Xg 的正方向。", forward, 1)
    coords = _splice_ground_body(
        coords,
        _frame_block("地面系", ground_hand, ground, "地面"),
        _frame_block("机体系", body_hand, body, "机体"),
    )
    return "\n".join([
        fixed,
        "",
        "绝对方向",
        direction.rstrip(),
        "",
        "场景",
        scene_body.rstrip(),
        _special_scene(order),
        "这个过程中有如下限制：",
        *task_rules,
        "",
        "角度",
        angles.rstrip(),
        "",
        "机构角",
        mechanism_block.strip(),
        "",
        "坐标系",
        coords.rstrip(),
    ])


def _decl_synced(model) -> str:
    mechanism = model.mechanism_angles
    mechanism_block = "\n".join([
        *_mechanism_lines(
            mechanism["alpha_m"].positive_airspeed_toward,
            mechanism["beta_m"].positive_airspeed_toward,
            mechanism["gamma_m"].right_wing_toward,
        ),
    ])
    from body_frame_model import BODY_AXES, WIND_VOCAB

    ground = _frame_rows(model.wind_frame, WIND_VOCAB.coordinate_axes, {"XFD": "Xg", "YFD": "Yg", "ZFD": "Zg"})
    body = _frame_rows(model.body_frame, BODY_AXES, {"XTZ": "Xb", "YTZ": "Yb", "ZTZ": "Zb"})
    return _compose_declaration(
        mechanism_block,
        ("pitch", "yaw", "roll"),
        ground,
        body,
        "右手系" if model.wind_chirality == "right" else "左手系",
        "右手系" if model.chirality == "right" else "左手系",
    )


def _decl(s):
    mechanism_block = "\n".join([
        *_mechanism_lines(
            {"up": "body_down", "down": "body_up"}[s["attack"]],
            {"right": "body_left", "left": "body_right"}[s["sideslip"]],
            {"down": "body_down", "up": "body_up"}[s["bank"]],
        ),
    ])
    lateral = "right" if s["normal"] == "down" else "left"
    normal = s["normal"]
    hand = "右手系" if s["normal"] == "down" else "左手系"
    ground = _clone_rows("tunnel", [
        ("Xg", "axial", "forward", "fixed"),
        ("Yg", "lateral", lateral, "inferred"),
        ("Zg", "normal", normal, "declared"),
    ])
    body = _clone_rows("body", [
        ("Xb", "axial", "forward", "declared"),
        ("Yb", "lateral", lateral, "inferred"),
        ("Zb", "normal", normal, "declared"),
    ])
    return _compose_declaration(mechanism_block, s["order"], ground, body, hand, hand)


if __name__ == "__main__":
    import json
    for tier in (25, 50, 75, 100):
        sample = build(tier)
        from domain_layout import SETS
        out = SETS["reconstruction.flight"]["root"] / f"sample_fmctrl_{tier}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"tier {tier}: {len(sample['questions'])} blanks -> {out}")
