# -*- coding: utf-8 -*-
"""Flight-mechanics control. Ground frame, then mechanism pitch, yaw, and roll, then a nonstandard body frame, then the actual angle of attack and sideslip.

The wind-tunnel wind frame is the ground frame here, and the body-axis frame is the body frame. Mechanism angles state only the attitude of body relative to ground.
Blanks line up with the wind-tunnel "wind frame to body-axis frame" mechanism section (63 blanks), without balance, sting, installation, elastic, or flow-offset angles.
build() is the tier-copied control group used by samples.jsonl.
build_synced() copies the ground frame, the body frame, and the positive senses of the theoretical angles and mechanism angles from a wind-tunnel sample.
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
_DIR = {"forward": "forward", "rearward": "rearward", "right": "right",
        "left": "left", "down": "downward", "up": "upward"}
SIX = ["upward", "downward", "left", "right", "forward", "rearward"]
FRAME_OPTIONS = ["body frame", "airflow frame", "ground frame"]
AXIS_OPTIONS = ["Xg", "Yg", "Zg", "xb", "yb", "zb", "Xa", "Ya", "Za"]
_DIRECTED_NAMES = ["Xg", "Yg", "Zg", "xb", "yb", "zb", "Xa", "Ya", "Za"]
DIRECTED_OPTIONS = _DIRECTED_NAMES + [f"-{name}" for name in _DIRECTED_NAMES]
CONVERSION_FRAMES = ["body frame", "airflow frame", "ground frame"]
CONVERSION_PAIRS = [f"{source} to {target}" for source in CONVERSION_FRAMES for target in CONVERSION_FRAMES]
_WIND_AXIS_SHOW = {
    "XFD": "Xg", "YFD": "Yg", "ZFD": "Zg",
    "XTZ": "xb", "YTZ": "yb", "ZTZ": "zb",
    "XTP": "Xa", "YTP": "Ya", "ZYP": "Za",
}
FACTOR_OPTIONS = ["RP1", "RP2", "R_α_M", "R_β_M", "R_γ_M"]
PROCESS = "body relative to ground"
_ORDER_LABEL = {"pitch": "pitch α_M", "yaw": "yaw β_M", "roll": "roll γ_M"}
_ROLE = {"axial": "axial", "normal": "normal", "lateral": "lateral"}
_SOURCE = {"declared": "declared", "inferred": "inferred from chirality", "fixed": "fixed"}
_BODY_SHOW = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
_MANUSCRIPT_PERIOD = chr(0x3002)


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
    # attack=up means a positive actual angle of attack is nose up (airspeed below the body).
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
        "pitch": {"name": "pitch α_M", "sym": AM, "role": "lateral",
                  "moving": axial, "toward": DIRECTIONS[s["attack"]]},
        "yaw": {"name": "yaw β_M", "sym": BM, "role": "normal",
                "moving": axial, "toward": DIRECTIONS[s["sideslip"]]},
        "roll": {"name": "roll γ_M", "sym": GM, "role": "axial",
                 "moving": lateral, "toward": DIRECTIONS[s["bank"]]},
    }
    role_axis = {"lateral": lateral, "normal": normal, "axial": axial}
    role_coord = {"lateral": "yb", "normal": "zb", "axial": "xb"}
    for key in angles:
        a = angles[key]
        a["sense"] = sense(role_axis[a["role"]], a["moving"], a["toward"])
        a["matrix"] = rot(a["role"], a["sense"] * a["sym"], s["normal"])
    # Total matrix: the first rotation is the rightmost factor.
    R = sp.eye(3)
    for key in reversed(s["order"]):
        R = R * angles[key]["matrix"]
    R = sp.simplify(R)
    XTZ, YTZ, ZTZ = R[0, 0], R[1, 0], R[2, 0]  # airspeed unit vector in the body frame = first column of R_t
    alpha = sp.atan2(-ZTZ, XTZ)
    beta = sp.asin(YTZ)

    # Direction of the airspeed or the right wing when each angle is positive.
    def airspeed_bias(key):
        # With only this mechanism angle positive, the airspeed direction in the body frame is the first column of R_key.
        m = angles[key]["matrix"].subs({AM: 1 if key == "pitch" else 0,
                                        BM: 1 if key == "yaw" else 0,
                                        GM: 1 if key == "roll" else 0})
        col = [m[0, 0], m[1, 0], m[2, 0]]
        # Dominant nonzero direction, other than axial.
        best = "forward"
        bestv = -1
        for dname, vec in DIRECTIONS.items():
            if dname in ("forward", "rearward"):
                continue
            v = dot(col, vec)
            if v > bestv:
                bestv, best = v, _DIR[dname]
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
        return {"prompt": prompt, "label": label, "format_code": "choice",
                "options": list(options), "answer": letters[options.index(correct)]}

    def m(prompt, mat, label="A2"):
        return _matrix_item(prompt, mat, label)

    def e(prompt, expr, label="A2"):
        return _expr_item(prompt, expr, label)

    cn = {"pitch": "angle of attack", "yaw": "sideslip", "roll": "roll"}
    for i in range(3):
        qs.append(c(
            f"What is rotation angle {i + 1}?",
            ["angle of attack", "sideslip", "roll"],
            cn[s["order"][i]],
            "A1",
        ))

    # Sentences 4 and 5: α_M and β_M (8 blanks each).
    for key in ("pitch", "yaw"):
        a = angles[key]
        bias = airspeed_bias(key)
        theory_bias = _DIR[{v: k for k, v in DIRECTIONS.items()}[a["toward"]]]  # direction named by the actual angle
        theory_name = "angle of attack" if key == "pitch" else "sideslip"
        qs.append(c("What installation does the model use?", ["upright", "inverted"], "upright", "C1"))
        qs.append(c(
            f"When mechanism {a['name']} > 0, which way does the airspeed point on the body?",
            SIX, bias, "A1",
        ))
        qs.append(c(
            f"When the actual {theory_name} > 0, which way does the airspeed point on the body?",
            SIX, theory_bias, "C1",
        ))
        qs.append(c(
            f"When mechanism {a['name']} > 0, what is the sign of the actual {theory_name}?",
            ["positive", "negative", "indeterminate"],
            "positive" if a["sense"] > 0 else "negative",
            "A1",
        ))
        qs.append(c(
            "This is a rotation about an axis of which frame at the level pose?",
            FRAME_OPTIONS, "body frame", "C2",
        ))
        qs.append(c(
            "Which body-frame axis is the rotation about?",
            AXIS_OPTIONS, role_coord[a["role"]], "C1",
        ))
        qs.append(c(
            "What is the sense of the rotation about that axis?",
            ["positive", "negative", "indeterminate"],
            "positive" if a["sense"] > 0 else "negative",
            "A1",
        ))
        qs.append(m(f"What is the rotation matrix R_{a['sym']}?", a["matrix"]))

    # Sentence 6: γ_M (9 blanks).
    a = angles["roll"]
    wing_bias = airspeed_bias("roll")  # stand-in for the right-wing direction
    theory_bias = _DIR[{v: k for k, v in DIRECTIONS.items()}[a["toward"]]]
    qs.append(c(
        "About the positive absolute model axis, the sense is how related to the right-hand rule?",
        ["same", "opposite"],
        "same" if a["sense"] > 0 else "opposite",
        "C1",
    ))
    qs.append(c("Which way does the right wing then point on the body?", SIX, wing_bias, "A1"))
    qs.append(c(
        "When the actual roll > 0, about the axial axis, the sense is how related to the right-hand rule?",
        ["same", "opposite"],
        "same" if a["sense"] > 0 else "opposite",
        "C1",
    ))
    qs.append(c("Which way does the right wing then point on the body?", SIX, theory_bias, "A1"))
    qs.append(c(
        "When the mechanism roll > 0, what is the sign of the actual roll?",
        ["positive", "negative", "indeterminate"],
        "positive" if a["sense"] > 0 else "negative",
        "A1",
    ))
    qs.append(c(
        "This is a rotation about an axis of which frame at the level pose?",
        FRAME_OPTIONS, "body frame", "C2",
    ))
    qs.append(c(
        "Which body-frame axis is the rotation about?",
        AXIS_OPTIONS, role_coord[a["role"]], "C1",
    ))
    qs.append(c(
        "What is the sense of the rotation about that axis?",
        ["positive", "negative", "indeterminate"],
        "positive" if a["sense"] > 0 else "negative",
        "A1",
    ))
    qs.append(m("What is the rotation matrix R_γ_M?", a["matrix"]))

    # Sentence 7: transform and factors (3 + 5 blanks).
    qs.append(c("The coordinate transform uses?", ["active", "passive"], "passive", "C2"))
    qs.append(c("The body frame is?", ["moving axes", "fixed axes"], "moving axes", "C2"))
    qs.append(c("The rotation uses?", ["intrinsic", "extrinsic"], "intrinsic", "C2"))
    for index, answer in enumerate(_factor_answers(s["order"]), start=1):
        qs.append(c(f"Which matrix is R{index} in the formula?", FACTOR_OPTIONS, answer, "A2"))
    # Sentence 8: expansions of R1, R2, and R3, then the total matrix.
    for index, key in enumerate(reversed(s["order"]), start=1):
        qs.append(m(f"What is the expansion of R{index} before the total matrix?", angles[key]["matrix"]))
    qs.append(m("Write the symbolically simplified total matrix R_t.", R))
    # Sentence 9: airspeed unit vector (2 blanks). At the initial pose the two axis sets coincide, so the pre-matrix is the identity.
    qs.append(_vector_item("RP2*RP1 times the ground-frame airspeed unit vector must be?", [1, 0, 0]))
    qs.append(c(
        "Which frame's airspeed coordinates are mapped into the body frame after the attitude change?",
        ["level body frame", "ground frame"],
        "level body frame",
        "C2",
    ))
    # Sentence 10: components (4 blanks).
    qs.append(_vector_item("What is the airspeed unit vector [X_V,Y_V,Z_V] in the level frame?", [1, 0, 0]))
    qs.append(e("What xb component does substitution give?", XTZ))
    qs.append(e("What yb component does substitution give?", YTZ))
    qs.append(e("What zb component does substitution give?", ZTZ))
    # Sentences 11 and 12: signs (6 blanks, only α_M > 0 and only β_M > 0).
    for key, nm in (("pitch", "α_M"), ("yaw", "β_M")):
        subs = {AM: 1 if key == "pitch" else 0, BM: 1 if key == "yaw" else 0, GM: 0}
        Xs, Ys, Zs = R.subs(subs)[0, 0], R.subs(subs)[1, 0], R.subs(subs)[2, 0]
        for val in (Xs, Ys, Zs):
            qs.append(c(
                f"When only {nm} > 0, what is the sign of this component?",
                ["positive", "negative", "zero"],
                _sign_word(val),
                "A1",
            ))
    # Sentences 13 and 14: signs in the α and β formulas (6 blanks each, simplified).
    for nm, expr, comp in (("α", alpha, None), ("β", beta, None)):
        del expr, comp
        qs.append(c(f"What is the angle sign of {nm}?", ["positive", "negative"], "positive", "A1"))
        qs.append(c(
            f"What is the inverse trigonometric function of {nm}?",
            ["arctan", "arcsin"],
            "arctan" if nm == "α" else "arcsin",
            "A1",
        ))
        qs.append(c(f"What is the numerator sign of {nm}?", ["positive", "negative"], "positive", "A1"))
        qs.append(c(
            f"What is the numerator component of {nm}?",
            ["zb", "yb", "xb", "1"],
            _num_component(nm),
            "A1",
        ))
        qs.append(c(f"What is the denominator sign of {nm}?", ["positive", "negative"], "positive", "A1"))
        qs.append(c(
            f"What is the denominator component of {nm}?",
            ["zb", "yb", "xb", "1"],
            _den_component(nm),
            "A1",
        ))
    # Sentence 15: final expressions (2 blanks).
    qs.append(e("What is the actual angle of attack α?", alpha))
    qs.append(e("What is the actual sideslip β?", beta))

    for i, item in enumerate(qs, start=1):
        item["local_id"] = i
        item["id"] = str(i)
        item["process"] = PROCESS
        item["corresponds_to"] = str(i)
    return {
        "sample_id": f"fmctrl-{tier:03d}", "tier": tier, "spec": s,
        "group": "control",  # control-group marker
        "declaration": _decl(s), "questions": qs,
    }


def _sign_word(expr):
    v = sp.simplify(expr)
    if v == 0:
        return "zero"
    return "positive" if v.subs({AM: 0.1, BM: 0.1, GM: 0.1}) > 0 else "negative"


def _body_component(name):
    """Map a wind-tunnel body-axis component name to a body-frame component name. 1 means the denominator is 1."""
    return {"XTZ": "xb", "YTZ": "yb", "ZTZ": "zb", "1": "1"}[name]


def _num_component(nm):
    return _body_component("ZTZ" if nm == "α" else "YTZ")


def _den_component(nm):
    return _body_component("XTZ" if nm == "α" else "1")


def _factor_answers(order) -> list[str]:
    """The product from left to right is R1 through R5. The last attitude matrix is on the left, and RP1 is on the right."""
    code = {"pitch": "R_α_M", "yaw": "R_β_M", "roll": "R_γ_M"}
    named = [code[key] for key in order]
    if len(named) != 3:
        raise RuntimeError(f"The rotation order must contain three items: {order}")
    return [named[2], named[1], named[0], "RP2", "RP1"]


def _prematrix_lines() -> list[str]:
    return [
        "- Coordinate transforms are passive. Attitude rotations use intrinsic moving axes.",
        "- Transform the axis correspondence before the attitude. Write [xb,yb,zb]^T=R1*R2*R3*R4*R5*[Xg,Yg,Zg]^T. In the product, write the pre-matrices from right to left first, then the mechanism-angle matrices. The matrix of the mechanism angle that rotates first sits next to the pre-matrices.",
        "- RP1 is the pre-transform from the ground frame to the body frame at the level pose. It makes the relation between the body-frame axes and their own absolute directions the same as the relation between the ground-frame axes and their own absolute directions.",
        "- Installation is upright only. When the model is upright and level, upward on the body agrees with upward on the ground. The positive absolute normal (upward) and positive absolute lateral (right) of the body then agree with the positive absolute normal and positive absolute lateral of the ground. Inverted installation would make upward on the body opposite upward on the ground, and would need another pre-transform RP2. RP2 makes the body-frame axis directions the same as the ground-frame coordinate directions. For upright installation RP2 is the identity.",
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
    raise RuntimeError(f"No coordinate axis points {side}: {rows}")


def _alignment_matrix(ground: list[dict], body: list[dict]):
    return sp.Matrix([
        [int(sum(a * b for a, b in zip(_row_vector(body_row), _row_vector(ground_row))))
         for ground_row in ground]
        for body_row in body
    ])


def _prefix_questions(ground: list[dict], body: list[dict]) -> list[dict]:
    """Opening alignment block. Options are the ground frame, the body frame, the airflow frame, and their axes.

    Installation is upright, so RP2 is the identity.
    """
    alignment = _alignment_matrix(ground, body)
    mount = sp.eye(3)
    items = [
        _choice("Which transform is required?", CONVERSION_PAIRS, "ground frame to body frame", "C2"),
    ]
    for side in ("up", "forward", "right"):
        items.append(_choice(
            "Which directed axis is the ground absolute direction?",
            DIRECTED_OPTIONS, _directed_name(ground, side), "C1",
        ))
    for side in ("up", "forward", "right"):
        items.append(_choice(
            "Which directed axis is the body absolute direction?",
            DIRECTED_OPTIONS, _directed_name(body, side), "C1",
        ))
    items.extend([
        _matrix_item("What is the pre-transform RP1 from the ground frame to the body frame at the level pose?", alignment),
        _choice("Which frame does this matrix align?", CONVERSION_FRAMES, "body frame", "C2"),
        _choice("Which frame does this matrix take as reference?", CONVERSION_FRAMES, "ground frame", "C2"),
        _choice("What is the model installation?", ["upright", "inverted"], "upright", "C1"),
        _choice(
            "Are the absolute normal and lateral directions of the model the same as those of the ground?",
            ["same", "opposite"], "same", "A1",
        ),
        _matrix_item("What is the pre-transform RP2?", mount),
        _choice("Which frame does RP2 align?", CONVERSION_FRAMES, "body frame", "C2"),
        _choice("Which frame does RP2 take as reference?", CONVERSION_FRAMES, "ground frame", "C2"),
    ])
    if len(items) != 15:
        raise RuntimeError(f"The prefix section must have 15 blanks, got {len(items)}.")
    return items


def _choice(prompt, options, correct, label):
    letters = "ABCDEFGHIJKLMNOP"
    return {"prompt": prompt, "label": label, "format_code": "choice",
            "options": list(options), "answer": letters[list(options).index(correct)]}


def _matrix_item(prompt, mat, label="A2"):
    from domains.wind_tunnel.tz_fd_generate import _matrix_text

    return {"prompt": prompt, "label": label, "format_code": "matrix", "options": [],
            "answer": _matrix_text(sp.Matrix(mat))}


def _vector_item(prompt, values, label="A1"):
    from domains.wind_tunnel.tz_fd_generate import _vector_text

    return {"prompt": prompt, "label": label, "format_code": "vector", "options": [],
            "answer": _vector_text(values)}


def _expr_item(prompt, expr, label="A2", format_code="expression"):
    from domains.wind_tunnel.tz_fd_generate import _text

    answer = expr if isinstance(expr, str) else _text(expr)
    return {"prompt": prompt, "label": label, "format_code": format_code, "options": [],
            "answer": answer}


def build_synced(wind_spec: dict, *, source_id: str, tier: int) -> dict:
    """One control sample. The ground frame matches the wind frame, and the body frame matches the body-axis frame.

    Installation is fixed upright, so an inverted wind-tunnel mount is not copied. The Euler order stays angle of attack, sideslip, roll.
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
        raise DefinitionError(
            f"The Euler order of {source_id} is not angle of attack, sideslip, roll."
        )
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
            raise DefinitionError(f"{source_id} has no definite rotation sense for {record.name}.")
        return ROTATION_ZH[record.rotation_when_positive]

    def axis_answer(record) -> str:
        return {"XTZ": "xb", "YTZ": "yb", "ZTZ": "zb"}[record.associated_axis]

    pitch = model.mechanism_angles[ALPHA_M]
    yaw = model.mechanism_angles[BETA_M]
    roll = model.mechanism_angles[GAMMA_M]
    theory = model.angles
    blocks = (
        ("pitch α_M", "angle of attack", pitch, theory[ALPHA], matrices[ALPHA_M], "positive_airspeed_toward"),
        ("yaw β_M", "sideslip", yaw, theory[BETA], matrices[BETA_M], "positive_airspeed_toward"),
    )
    qs = _prefix_questions(ground, body)
    for index, name in enumerate(("angle of attack", "sideslip", "roll"), start=1):
        qs.append(_choice(
            f"What is rotation angle {index}?",
            ["angle of attack", "sideslip", "roll"],
            name,
            "A1",
        ))
    for name, theory_name, mechanism, theory_record, matrix, field in blocks:
        outer = _same_sense(getattr(mechanism, field), getattr(theory_record, field))
        qs.append(_choice("What installation does the model use?", ["upright", "inverted"], "upright", "C1"))
        qs.append(_choice(
            f"When mechanism {name} > 0, which way does the airspeed point on the body?",
            SIX, direction_answer(mechanism, field), "A1",
        ))
        qs.append(_choice(
            f"When the actual {theory_name} > 0, which way does the airspeed point on the body?",
            SIX, direction_answer(theory_record, field), "C1",
        ))
        qs.append(_choice(
            f"When mechanism {name} > 0, what is the sign of the actual {theory_name}?",
            ["positive", "negative", "indeterminate"], outer, "A1",
        ))
        qs.append(_choice(
            "This is a rotation about an axis of which frame at the level pose?",
            FRAME_OPTIONS, "body frame", "C2",
        ))
        qs.append(_choice(
            "Which body-frame axis is the rotation about?",
            AXIS_OPTIONS, axis_answer(mechanism), "C1",
        ))
        qs.append(_choice(
            "What is the sense of the rotation about that axis?",
            ["positive", "negative", "indeterminate"], rotation_answer(mechanism), "A1",
        ))
        qs.append(_matrix_item(f"What is the rotation matrix R_{mechanism.symbol}?", matrix))

    roll_outer = _same_sense(roll.right_wing_toward, theory[GAMMA].right_wing_toward)
    qs.append(_choice(
        "About the positive absolute model axis, the sense is how related to the right-hand rule?",
        ["same", "opposite"], _roll_hand(roll.right_wing_toward), "C1",
    ))
    qs.append(_choice(
        "Which way does the right wing then point on the body?",
        SIX, direction_answer(roll, "right_wing_toward"), "A1",
    ))
    qs.append(_choice(
        "When the actual roll > 0, about the axial axis, the sense is how related to the right-hand rule?",
        ["same", "opposite"], _roll_hand(theory[GAMMA].right_wing_toward), "C1",
    ))
    qs.append(_choice(
        "Which way does the right wing then point on the body?",
        SIX, direction_answer(theory[GAMMA], "right_wing_toward"), "A1",
    ))
    qs.append(_choice(
        "When the mechanism roll > 0, what is the sign of the actual roll?",
        ["positive", "negative", "indeterminate"], roll_outer, "A1",
    ))
    qs.append(_choice(
        "This is a rotation about an axis of which frame at the level pose?",
        FRAME_OPTIONS, "body frame", "C2",
    ))
    qs.append(_choice(
        "Which body-frame axis is the rotation about?",
        AXIS_OPTIONS, axis_answer(roll), "C1",
    ))
    qs.append(_choice(
        "What is the sense of the rotation about that axis?",
        ["positive", "negative", "indeterminate"], rotation_answer(roll), "A1",
    ))
    qs.append(_matrix_item("What is the rotation matrix R_γ_M?", matrices[GAMMA_M]))

    qs.append(_choice("The coordinate transform uses?", ["active", "passive"], "passive", "C2"))
    qs.append(_choice("The body frame is?", ["moving axes", "fixed axes"], "moving axes", "C2"))
    qs.append(_choice("The rotation uses?", ["intrinsic", "extrinsic"], "intrinsic", "C2"))
    synced_order = ("pitch", "yaw", "roll")
    synced_matrix = {"pitch": ALPHA_M, "yaw": BETA_M, "roll": GAMMA_M}
    for index, answer in enumerate(_factor_answers(synced_order), start=1):
        qs.append(_choice(f"Which matrix is R{index} in the formula?", FACTOR_OPTIONS, answer, "A2"))
    for index, key in enumerate(reversed(synced_order), start=1):
        qs.append(_matrix_item(
            f"What is the expansion of R{index} before the total matrix?",
            matrices[synced_matrix[key]],
        ))
    qs.append(_matrix_item("Write the symbolically simplified total matrix R_t.", shown_total))
    qs.append(_vector_item("RP2*RP1 times the ground-frame airspeed unit vector must be?", level))
    qs.append(_choice(
        "Which frame's airspeed coordinates are mapped into the body frame after the attitude change?",
        ["level body frame", "ground frame"],
        "level body frame",
        "C2",
    ))
    qs.append(_vector_item("What is the airspeed unit vector [X_V,Y_V,Z_V] in the level frame?", level))
    for index, axis_name in enumerate(("xb", "yb", "zb")):
        qs.append(_expr_item(f"What {axis_name} component does substitution give?", shown[index]))
    for symbol, active in (("α_M", AM), ("β_M", BM)):
        quiet = {AM: 0, BM: 0, GM: 0}
        quiet.pop(active)
        for index, axis_name in enumerate(("xb", "yb", "zb")):
            qs.append(_choice(
                f"When only {symbol} > 0, what is the sign of the airspeed {axis_name} component in the body frame?",
                ["positive", "negative", "zero"],
                _component_sign(shown[index].subs(quiet), active),
                "A1",
            ))
    for name, formula in (("α", alpha_formula), ("β", beta_formula)):
        qs.append(_choice(f"What is the angle sign of {name}?", ["positive", "negative"], formula["outer"], "A1"))
        qs.append(_choice(
            f"What is the inverse trigonometric function of {name}?",
            ["arctan", "arcsin"], formula["inverse"], "A1",
        ))
        qs.append(_choice(
            f"What is the numerator sign of {name}?",
            ["positive", "negative"], formula["numerator_sign"], "A1",
        ))
        qs.append(_choice(
            f"What is the numerator component of {name}?",
            ["zb", "yb", "xb", "1"], _body_component(formula["numerator"]), "A1",
        ))
        qs.append(_choice(
            f"What is the denominator sign of {name}?",
            ["positive", "negative"], formula["denominator_sign"], "A1",
        ))
        qs.append(_choice(
            f"What is the denominator component of {name}?",
            ["zb", "yb", "xb", "1"], _body_component(formula["denominator"]), "A1",
        ))
    qs.append(_expr_item("What is the actual angle of attack α?", alpha_formula["answer"], format_code="arctan"))
    qs.append(_expr_item("What is the actual sideslip β?", beta_formula["answer"], format_code="arcsin"))
    if len(qs) != 81:
        raise DefinitionError(f"{source_id} produced {len(qs)} blanks; expected 81.")
    for index, item in enumerate(qs, start=1):
        item["local_id"] = index
        item["id"] = str(index)
        item["process"] = PROCESS
        item["corresponds_to"] = str(index)
    return {
        "sample_id": f"fmctrl-sync-{source_id}",
        "id": f"fmctrl-sync-{source_id}",
        "tier": tier,
        "title": f"Flight-mechanics control (synced with wind tunnel {source_id})",
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
    """Standard definition for the synced control: the wind-tunnel standard wind frame, body frame, and five angles."""
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

    return build_synced(STANDARD_SPEC, source_id="standard", tier=0)


def emit_synced(out_path=None, pilot_path=None) -> list[dict]:
    """Write 10 synced control items from the wind-tunnel pilot_10. Does not overwrite samples.jsonl."""
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
        raise RuntimeError("The wind-tunnel samples do not contain these ids: " + ", ".join(sorted(missing)))
    rows = [build_synced(found[source_id]["spec"], source_id=source_id, tier=found[source_id]["tier"])
            for source_id in source_ids]
    # Axes follow the aerodynamics meta-definition. Samples that share angle definitions can share one declaration.
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
    """Write an already-positive numerator or denominator so the text does not start with a built-in minus."""
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
        denominator_sign = "positive"
        denominator_expr = sp.Integer(1)
    else:
        denominator_sign = _magnitude_sign(signs, denominator, theory.symbol)
        denominator_expr = posed[{"XTZ": 0, "YTZ": 1, "ZTZ": 2}[denominator]]
    numerator_expr = _apply_sign(numerator_sign, posed[{"XTZ": 0, "YTZ": 1, "ZTZ": 2}[numerator]])
    denominator_expr = _apply_sign(denominator_sign, denominator_expr)
    _require_positive_part(numerator_expr, active, f"{theory.symbol} numerator")
    _require_positive_part(denominator_expr, active, f"{theory.symbol} denominator")
    outer = _same_sense(getattr(mechanism, field), toward)
    if outer not in ("positive", "negative"):
        raise DefinitionError(f"The outer sign of {theory.symbol} is not determined.")
    from domains.wind_tunnel.tz_fd_generate import _signed_formula

    if inverse == "arctan":
        interior = f"({_magnitude_text(numerator_expr)})/({_magnitude_text(denominator_expr)})"
    elif inverse == "arcsin":
        interior = _magnitude_text(numerator_expr)
    else:
        raise DefinitionError(f"Unknown inverse trigonometric function: {inverse}")
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
        return "zero"
    number = sp.simplify(value.subs({AM: 0, BM: 0, GM: 0, active: sp.Rational(1, 10)}))
    if number == 0:
        return "zero"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    raise RuntimeError(f"Cannot judge the component sign: {value}")


def _roll_hand(right_wing_toward: str) -> str:
    from domains.wind_tunnel.tz_fd_generate import _declared_roll_hand

    return _declared_roll_hand(right_wing_toward)


def _toward_phrase(direction: str) -> str:
    owner, side = direction.split("_", 1)
    who = "body" if owner == "body" else "ground"
    return f"{_DIR[side]} on the {who}"


def _nose_word(airspeed_toward: str) -> str:
    """Nose deflection is opposite the airspeed deflection, matching the wind-tunnel positive airspeed direction."""
    return {
        "body_down": "nose up",
        "body_up": "nose down",
        "body_left": "right",
        "body_right": "left",
    }[airspeed_toward]


def _wing_word(right_wing_toward: str) -> str:
    return {"body_down": "downward", "body_up": "upward"}[right_wing_toward]


def _mechanism_lines(alpha_toward: str, beta_toward: str, gamma_toward: str) -> list[str]:
    from body_frame_model import SENSE_SAME, roll_about_nose_sense
    from domains.wind_tunnel.dataset_generate import _balance_deflection

    roll = "same" if roll_about_nose_sense(gamma_toward) == SENSE_SAME else "opposite"
    return [
        (
            "Mechanism angle of attack α_M: the pitch support rod moves in the vertical plane and may be treated as a rotation about the lateral axis of the ground frame. "
            "It is the angle between the rod (the body axis) and the horizontal plane. "
            f"Positive when the body deflects {_balance_deflection(alpha_toward)} relative to the ground."
        ),
        (
            "Mechanism sideslip β_M: the sideslip support rod moves in the plane perpendicular to the vertical plane and containing the pitch support rod, "
            "so the sideslip mechanism is driven on top of the pitch mechanism. "
            "β_M is the angle between the sideslip support rod and the vertical plane. "
            f"Positive when the body deflects {_balance_deflection(beta_toward)} relative to the ground."
        ),
        (
            "Mechanism roll γ_M: the rotation of the body about the sideslip support rod. "
            "When the angle is positive, the rotation about the positive absolute model axis (forward) is "
            f"{roll} the right-hand rule."
        ),
    ]


def _aero_base() -> str:
    """Standard aerodynamics declaration, with the stock mechanism-angle block removed.

    The control task writes its own mechanism angles back in.
    """
    from copy import deepcopy

    from flight_frame_model import STANDARD_SPEC, FlightFrameModel
    from domains.aerodynamics.dataset_generate import _declaration

    text = _declaration(FlightFrameModel().declare(deepcopy(STANDARD_SPEC)))
    return _drop_mechanism_stub(text)


def _drop_mechanism_stub(text: str) -> str:
    """Remove the stock mechanism-angle block that the control declaration replaces.

    The block is the heading immediately before the α_M sentence, the mechanism
    sentences, and the blank line after them.
    """
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if "α_M" in line:
            if index == 0 or not lines[index - 1].strip():
                raise RuntimeError("The aerodynamics declaration has no mechanism-angle heading.")
            start = index - 1
            break
    if start is None:
        raise RuntimeError("The aerodynamics declaration has no mechanism-angle block to replace.")
    end = start + 1
    while end < len(lines) and lines[end].strip():
        end += 1
    if end < len(lines) and not lines[end].strip():
        end += 1
    kept = lines[:start] + lines[end:]
    return "\n".join(kept).rstrip() + "\n"


def _special_scene(order) -> str:
    del order
    return (
        "One special task starts with the body frame and the ground frame sharing the same absolute directions, "
        "and with the airflow aligned with the ground-frame axis, blowing toward the nose. "
        "The tail of the body is fixed to a mechanism that carries the body into a special attitude by a new Euler sequence. "
        "The sequence differs from the ordinary one, so the mechanism angles must be understood before the correct order can be inferred. "
        "Those angles are defined below. "
        "They carry the body frame, initially coincident with the ground frame, to a body frame that is not a traditional attitude. "
        "At that pose the theoretical flow angles are computed again from the definitions of the theoretical angle of attack and sideslip."
    )


def _show_axis(name: str) -> str:
    return _BODY_SHOW.get(name, name)


def _axis_line(row: dict, who: str) -> str:
    role = row["along"].split("_", 1)[1]
    side = _DIR[row["points"].split("_", 1)[1]]
    return (
        f"- {_show_axis(row['name'])}: along the {who} {_ROLE[role]} direction, "
        f"pointing {side} on the {who} ({_SOURCE[row['source']]})."
    )


def _frame_block(title: str, hand: str, rows: list[dict], who: str) -> str:
    return "\n".join([f"{title} ({hand})", *[_axis_line(row, who) for row in rows]])


def _splice_ground_body(coords: str, ground_block: str, body_block: str) -> str:
    parts = coords.split("\n\n")
    airflow_at = None
    for index, part in enumerate(parts):
        if any(line.startswith("- Xa") for line in part.splitlines()):
            airflow_at = index
            break
    if airflow_at is None:
        raise RuntimeError(
            "The coordinate frames have no airflow frame, so the ground frame and the body frame cannot be rewritten."
        )
    tail = "\n\n".join(parts[airflow_at:])
    return f"{ground_block}\n\n{body_block}\n\n{tail}"


def _clone_rows(owner: str, binding: list[tuple[str, str, str, str]]) -> list[dict]:
    return [
        {"name": name, "along": f"{owner}_{role}", "points": f"{owner}_{side}", "source": source}
        for name, role, side, source in binding
    ]


def _body_after_heading(block: str) -> str:
    lines = block.splitlines()
    if len(lines) < 2:
        raise RuntimeError("A declaration section is missing its body.")
    return "\n".join(lines[1:])


def _split_angles_and_first_frame(block: str) -> tuple[str, str]:
    """Angle lines, then the coordinate heading, then the first frame.

    The first frame begins at the line before the Xg axis bullet.
    """
    lines = block.splitlines()
    if len(lines) < 3:
        raise RuntimeError("The angle section does not reach the coordinate frames.")
    body = lines[1:]
    ground_at = None
    for index, line in enumerate(body):
        if line.startswith("- Xg"):
            ground_at = index - 1
            break
    if ground_at is None or ground_at < 1:
        raise RuntimeError("The aerodynamics declaration has no ground-frame axes.")
    heading_at = ground_at - 1
    angles = "\n".join(body[:heading_at]).rstrip("\n")
    first_frame = "\n".join(body[ground_at:])
    return angles, first_frame


def _declaration_sections(text: str) -> tuple[str, str, str, str, str]:
    """Split conventions, directions, scene, angles, and frames on blank lines.

    After the mechanism block is removed, the angle lines and the first frame
    share one block. Later frames stay in the following blocks.
    """
    parts = text.rstrip("\n").split("\n\n")
    if len(parts) < 6:
        raise RuntimeError(
            f"The aerodynamics declaration has {len(parts)} blocks; expected the conventions, directions, scene, angles, and frames."
        )
    fixed = parts[0]
    direction = _body_after_heading(parts[1])
    scene_body = _body_after_heading(parts[2])
    angles, first_frame = _split_angles_and_first_frame(parts[3])
    coords = "\n\n".join([first_frame, *parts[4:]])
    return fixed, direction, scene_body, angles, coords


def _replace_ground_axis_line(fixed: str, role: str, side: str) -> str:
    lines = fixed.splitlines()
    for index, line in enumerate(lines):
        if "Xg" in line:
            lines[index] = f"- Ground axis Xg lies along the ground {role} direction and points {side}."
            return "\n".join(lines)
    raise RuntimeError("The conventions do not name ground axis Xg.")


def _replace_forward_clause(direction: str, axial_name: str, points_forward: bool) -> str:
    if points_forward:
        clause = f"The forward sense is the positive direction of {axial_name}."
    else:
        clause = f"The forward sense is opposite the positive direction of {axial_name}."
    lines = direction.splitlines()
    for index, line in enumerate(lines):
        if "Xg" not in line or "Yg" in line:
            continue
        mark = line.find(" Xg ")
        if mark < 0:
            mark = line.find("Xg")
        for separator in (_MANUSCRIPT_PERIOD, "."):
            before = line.rfind(separator, 0, mark)
            after = line.find(separator, mark)
            if before >= 0 and after >= 0:
                lines[index] = line[: before + 1] + clause + line[after + 1 :]
                return "\n".join(lines)
        raise RuntimeError("Could not locate the forward-direction clause of Xg.")
    raise RuntimeError("The absolute-direction section has no forward-direction clause for Xg.")


def _compose_declaration(mechanism_block: str, order, ground: list[dict], body: list[dict], ground_hand: str, body_hand: str) -> str:
    fixed, direction, scene_body, angles, coords = _declaration_sections(_aero_base())
    task_rules = _prematrix_lines()[1:]
    xg = next(row for row in ground if row["name"] == "Xg")
    role = xg["along"].split("_", 1)[1]
    side = _DIR[xg["points"].split("_", 1)[1]]
    fixed = _replace_ground_axis_line(fixed, _ROLE[role], side)
    axial = next(row for row in ground if row["along"].endswith("axial"))
    axial_name = _show_axis(axial["name"])
    direction = _replace_forward_clause(
        direction,
        axial_name,
        axial["points"].split("_", 1)[1] == "forward",
    )
    coords = _splice_ground_body(
        coords,
        _frame_block("Ground frame", ground_hand, ground, "ground"),
        _frame_block("Body frame", body_hand, body, "body"),
    )
    return "\n".join([
        fixed,
        "",
        "Absolute directions",
        direction.rstrip(),
        "",
        "Scene",
        scene_body.rstrip(),
        _special_scene(order),
        "The following restrictions apply to this process:",
        *task_rules,
        "",
        "Angles",
        angles.rstrip(),
        "",
        "Mechanism angles",
        mechanism_block.strip(),
        "",
        "Coordinate frames",
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
        "right-handed" if model.wind_chirality == "right" else "left-handed",
        "right-handed" if model.chirality == "right" else "left-handed",
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
    hand = "right-handed" if s["normal"] == "down" else "left-handed"
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
