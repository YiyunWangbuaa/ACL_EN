# -*- coding: utf-8 -*-
"""CV-CTRL：机器视觉对照，逐空对应 CV 主数据的纯 CV 片段。

CV 主数据的纯 CV 片段 = camera_attitude(42) + projection(12) + extrinsic(2) + epipolar(5)
共 61 空。本对照把这 61 空原样保留，只把「风轴系」换成「世界系」、「机头/机体轴线」
换成「世界原点/世界轴」，去掉 body_attitude 与 axis_wind（飞行力学片段）。
"""
from __future__ import annotations

import sympy as sp

from domains.computer_vision import cv_model as cm

DIR_CN = {"forward": "前方", "rearward": "后方", "right": "右侧",
          "left": "左侧", "down": "下方", "up": "上方"}
DIR_NAME = {value: key for key, value in cm.DIRECTIONS.items()}
SIX = ["前方", "后方", "右侧", "左侧", "下方", "上方"]
AXIS_ALL = ["Xw", "Yw", "Zw", "X1", "Y1", "Z1", "X2", "Y2", "Z2"]
AXIS_OPTIONS = AXIS_ALL + ["-" + name for name in AXIS_ALL]
CAM_EULER = ["方位角", "俯仰角", "滚转角"]

fx1, fy1, cx1, cy1 = sp.symbols("fx1 fy1 cx1 cy1")
fx2, fy2, cx2, cy2 = sp.symbols("fx2 fy2 cx2 cy2")
x1, y1, z1, x2, y2, z2 = sp.symbols("x1 y1 z1 x2 y2 z2")
K1 = sp.Matrix([[fx1, 0, cx1], [0, fy1, cy1], [0, 0, 1]])
K2 = sp.Matrix([[fx2, 0, cx2], [0, fy2, cy2], [0, 0, 1]])
C1 = sp.Matrix([x1, y1, z1])
C2 = sp.Matrix([x2, y2, z2])

SPECS = {
    25: {
        "world": {"chirality": "left", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "up"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "right"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    50: {
        "world": {"chirality": "left", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "up"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "rearward"},
                    "X1": {"along": "lateral", "points": "left"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    75: {
        "world": {"chirality": "right", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "down"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "right"}},
        "camera2": {"chirality": "left", "Z2": {"along": "axial", "points": "forward"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
    100: {
        "world": {"chirality": "right", "Xw": {"along": "axial", "points": "forward"},
                  "Zw": {"along": "normal", "points": "down"}},
        "camera1": {"chirality": "left", "Z1": {"along": "axial", "points": "forward"},
                    "X1": {"along": "lateral", "points": "left"}},
        "camera2": {"chirality": "right", "Z2": {"along": "normal", "points": "down"},
                    "X2": {"along": "lateral", "points": "right"}},
    },
}
DUMMY_BODY = {"chirality": "left", "Xb": {"along": "axial", "points": "forward"},
              "Zb": {"along": "normal", "points": "up"}}


def _m(matrix):
    return "[" + ", ".join("[" + ", ".join(sp.sstr(v) for v in row) + "]" for row in matrix.tolist()) + "]"


def _choice(prompt, options, correct, label):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return {"prompt": prompt, "label": label, "format_code": "无",
            "options": list(options), "answer": letters[options.index(correct)]}


def _matrix(prompt, matrix, label="A2"):
    return {"prompt": prompt, "label": label, "format_code": "矩阵", "options": [],
            "answer": _m(matrix)}


def _vector(prompt, vec, label="A2"):
    return {"prompt": prompt, "label": label, "format_code": "向量", "options": [],
            "answer": "[" + ", ".join(sp.sstr(v) for v in vec) + "]"}


def build(tier):
    spec = {"wind": SPECS[tier]["world"], "body": DUMMY_BODY,
            "camera1": SPECS[tier]["camera1"], "camera2": SPECS[tier]["camera2"]}
    scene = cm.Scene().declare(spec)
    cam1_s = scene.camera_senses(scene.cam1)
    cam2_s = scene.camera_senses(scene.cam2)
    psi1, th1, ph1 = scene.psi1, scene.theta1, scene.phi1
    psi2, th2, ph2 = scene.psi2, scene.theta2, scene.phi2

    qs = []

    def angle_process(cam, senses, ang, name, key):
        role = {"pan": "normal", "tilt": "lateral", "roll": "axial"}[key]
        toward = scene.toward_vector(scene.angle_toward[key], cam)
        bias = DIR_CN[DIR_NAME[toward]]
        axis = cm.axis_on(cam, role)[0]
        sign = senses[{"pan": 0, "tilt": 1, "roll": 2}[key]]
        matrix = cm.rot(role, sign * ang)
        qs.append(_choice(f"{name}>0 时，光轴向世界系的哪一方偏？", SIX, bias, "A1"))
        qs.append(_choice(f"{name} 相当于哪个坐标系平正时绕轴旋转？",
                          ["世界系", "相机1系", "相机2系"], _cam_label(cam), "C2"))
        qs.append(_choice(f"{name} 绕相机系的哪个坐标轴旋转？", AXIS_ALL, axis, "C1"))
        qs.append(_choice("绕该轴是正转还是反转？", ["正", "负"], "正" if sign > 0 else "负", "A1"))
        qs.append(_matrix(f"写出 {name} 对应的旋转矩阵。", matrix))

    def _cam_label(cam):
        return "相机1系" if cam is scene.cam1 else "相机2系"

    # camera1 姿态（与 CV 主数据逐空对应）
    qs.append(_choice("为把相机1姿态角计入成像几何，应进行哪种转换？",
                      ["相机1系向世界系", "世界系向相机1系"], "相机1系向世界系", "C2"))
    qs.append(_choice("该变换采用主动还是被动？", ["主动", "被动"], "被动", "C2"))
    qs.append(_choice("相机系是动轴还是定轴？", ["动轴", "定轴"], "动轴", "C2"))
    qs.append(_choice("采用内旋还是外旋？", ["内旋", "外旋"], "内旋", "C2"))
    for i in range(3):
        qs.append(_choice(f"相机1 姿态角顺序（第{i+1}个）？", CAM_EULER, CAM_EULER[i], "A1"))
    angle_process(scene.cam1, cam1_s, psi1, "相机1 方位角 ψ1", "pan")
    angle_process(scene.cam1, cam1_s, th1, "相机1 俯仰角 θ1", "tilt")
    angle_process(scene.cam1, cam1_s, ph1, "相机1 滚转角 φ1", "roll")
    qs.append(_matrix("写出 R_cg1 = R_ψ1·R_θ1·R_φ1·A1 中的 R_ψ1。", cm.rot("normal", cam1_s[0] * psi1)))
    qs.append(_matrix("写出 R_cg1 = R_ψ1·R_θ1·R_φ1·A1 中的 R_θ1。", cm.rot("lateral", cam1_s[1] * th1)))
    qs.append(_matrix("写出 R_cg1 = R_ψ1·R_θ1·R_φ1·A1 中的 R_φ1。", cm.rot("axial", cam1_s[2] * ph1)))
    r_cg1 = scene.camera_to_ground(scene.cam1, ("X1", "Y1", "Z1"), psi1, th1, ph1)
    qs.append(_matrix("写出相机1 到世界系的旋转 R_cg1（全展开）。", r_cg1))

    # camera2 姿态
    angle_process(scene.cam2, cam2_s, psi2, "相机2 方位角 ψ2", "pan")
    angle_process(scene.cam2, cam2_s, th2, "相机2 俯仰角 θ2", "tilt")
    angle_process(scene.cam2, cam2_s, ph2, "相机2 滚转角 φ2", "roll")
    qs.append(_matrix("写出 R_cg2 = R_ψ2·R_θ2·R_φ2 中的 R_ψ2。", cm.rot("normal", cam2_s[0] * psi2)))
    qs.append(_matrix("写出 R_cg2 = R_ψ2·R_θ2·R_φ2 中的 R_θ2。", cm.rot("lateral", cam2_s[1] * th2)))
    qs.append(_matrix("写出 R_cg2 = R_ψ2·R_θ2·R_φ2 中的 R_φ2。", cm.rot("axial", cam2_s[2] * ph2)))
    r_cg2 = scene.camera_to_ground(scene.cam2, ("X2", "Y2", "Z2"), psi2, th2, ph2)
    qs.append(_matrix("写出相机2 到世界系的旋转 R_cg2（全展开）。", r_cg2))

    # projection（与 CV 主数据逐空对应，点 = 世界原点）
    r_cw1 = sp.simplify(r_cg1.T)
    r_cw2 = sp.simplify(r_cg2.T)
    t_cw1 = sp.simplify(-r_cw1 * C1)
    t_cw2 = sp.simplify(-r_cw2 * C2)
    origin = sp.Matrix([0, 0, 0])
    p_c1 = sp.simplify(r_cw1 * origin + t_cw1)
    p_c2 = sp.simplify(r_cw2 * origin + t_cw2)

    def skew(v):
        px, py, pz = v[0], v[1], v[2]
        return sp.Matrix([[0, -pz, py], [pz, 0, -px], [-py, px, 0]])

    skew1 = sp.simplify(skew(p_c1))
    skew2 = sp.simplify(skew(p_c2))
    s1 = cm.camera_layout_permutation(scene.cam1, ("X1", "Y1", "Z1"))
    s2 = cm.camera_layout_permutation(scene.cam2, ("X2", "Y2", "Z2"))
    k1it = sp.simplify((K1 * s1).inv().T)
    k2it = sp.simplify((K2 * s2).inv().T)

    qs.append(_matrix("写出相机1 的有效内参逆转置 (K1·S1)^(-T)。", k1it))
    qs.append(_matrix("写出相机2 的有效内参逆转置 (K2·S2)^(-T)。", k2it))
    qs.append(_matrix("写出相机1 的位姿旋转 R_cw1（世界系到相机1系）。", r_cw1))
    qs.append(_matrix("写出相机2 的位姿旋转 R_cw2（世界系到相机2系）。", r_cw2))
    qs.append(_vector("写出相机1 的位姿平移 t_cw1 = −R_cw1·C1。", t_cw1))
    qs.append(_vector("写出相机2 的位姿平移 t_cw2 = −R_cw2·C2。", t_cw2))
    qs.append(_vector("写出世界原点在相机1 系中的坐标 P_c1 = R_cw1·O + t_cw1。", p_c1))
    qs.append(_vector("写出世界原点在相机2 系中的坐标 P_c2 = R_cw2·O + t_cw2。", p_c2))
    qs.append(_matrix("写出 P_c1 的反对称矩阵 [P_c1]_×。", skew1))
    qs.append(_matrix("写出 P_c2 的反对称矩阵 [P_c2]_×。", skew2))

    def line_formula(cam):
        c = cam
        correct = f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · R_cw{c} · d_w"
        options = [
            f"l = (K{c}·S{c})^(-T) · R_cw{c} · [P_c{c}]_× · d_w",
            f"l = (K{c}·S{c})^(-1) · [P_c{c}]_× · R_cw{c} · d_w",
            correct,
            f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · R_cw{c}^T · d_w",
            f"l = (K{c}·S{c})^(-T) · [P_c{c}]_× · d_w",
            f"l = [P_c{c}]_× · (K{c}·S{c})^(-T) · R_cw{c} · d_w",
        ]
        return _choice(f"相机{c}图像中的直线 l 应满足哪个公式？", options, correct, "A2")

    qs.append(line_formula("1"))
    qs.append(line_formula("2"))

    # extrinsic
    r21 = sp.simplify(r_cw2 * r_cw1.T)
    t21 = sp.simplify(t_cw2 - r21 * t_cw1)
    qs.append(_matrix("在计算总矩阵前，按刚选定的符号写出每个分矩阵的展开式。R_cw2的展开式为", r_cw2))
    qs.append(_matrix("R_cw1的展开式为", r_cw1))
    qs.append(_matrix("写出相机1 到相机2 的旋转 R21（全展开）。", r21))
    qs.append(_vector("写出相机1 到相机2 的平移 t21（全展开）。", t21))

    # epipolar
    qs.append(_choice("本质矩阵 E 的表达式是？",
                      ["E = skew(t21)*R21", "E = R21*skew(t21)", "E = skew(t21)+R21", "E = R21^T*skew(t21)"],
                      "E = skew(t21)*R21", "A2"))
    qs.append(_choice("基础矩阵 F 的表达式是？",
                      ["F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "F = (K1·S1)^(-T)*E*(K2·S2)^(-1)", "F = (K2·S2)^(-T)*E*K1", "F = E*(K1·S1)^(-1)*(K2·S2)^(-T)"],
                      "F = (K2·S2)^(-T)*E*(K1·S1)^(-1)", "A2"))
    qs.append(_choice("极线约束方程是？",
                      ["[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "[X1,Y1,1]*F*[X2,Y2,1]^T = 0",
                       "[X2,Y2,1]+F*[X1,Y1,1]^T = 0", "F*[X1,Y1,1]^T*[X2,Y2,1] = 0"],
                      "[X2,Y2,1]*F*[X1,Y1,1]^T = 0", "A2"))
    qs.append(_choice("三角化得到的点在哪个坐标系？", ["世界系", "相机1系", "相机2系"], "世界系", "A1"))
    qs.append(_choice("什么时候三角化会退化？",
                      ["基线为零、点在无穷远或深度为负", "基线越长", "内参已知", "像点越多"],
                      "基线为零、点在无穷远或深度为负", "A1"))

    SEGMENT_BOUNDS = [(45, "camera_attitude"), (57, "projection"), (61, "extrinsic"), (66, "epipolar")]
    OFFSET = {"camera_attitude": 12, "projection": 38, "extrinsic": 42, "epipolar": 42}

    def _segment_of(index):
        for bound, seg in SEGMENT_BOUNDS:
            if index <= bound:
                return seg
        return "epipolar"

    for i, item in enumerate(qs, start=1):
        seg = _segment_of(i)
        item["local_id"] = i
        item["id"] = str(i)
        item["process"] = "世界系到相机"
        item["segment"] = seg
        item["corresponds_to"] = str(i + OFFSET[seg])  # CV 主数据对应 local_id

    return {
        "sample_id": f"cvctrl-{tier:03d}", "tier": tier, "spec": SPECS[tier],
        "group": "control",
        "declaration": _decl(SPECS[tier], scene), "questions": qs,
    }


def _decl(spec, scene):
    def axes_text(names, frame):
        return "；".join(f"{axis} 指向{DIR_CN[DIR_NAME[frame[axis]]]}" for axis in names)

    chirality = {"right": "右", "left": "左"}
    return "\n".join([
        "固定约定",
        "- 坐标系转换采用被动变换。相机姿态旋转采用动轴内旋。最先旋转的角度，其矩阵写在连乘式最右侧。",
        "- 相机成像采用针孔模型 x = (K·S)·[R|t]·X。图像直线 l = (K·S)^(-T)·n，n 为光心与空间直线张成平面的法向。",
        "- 极线约束 x2^T·F·x1 = 0，F = (K2·S2)^(-T)·E·(K1·S1)^(-1)，E = [t21]_×·R21。",
        "",
        "绝对方向",
        "世界坐标系与相机系各有三根绝对轴。两套绝对轴的正方向定义相同：轴向正方向是前方，法向正方向是上方，侧向正方向是右侧。六个绝对方向是这三根轴上的两个指向。",
        "坐标轴声明只是指定每根坐标轴沿哪根绝对轴、指向哪一侧，可以与绝对正方向相同，也可以相反。",
        "两个系共用同一套指向关系：前方 × 右侧 = 下方。沿前方看去，右侧在右手侧，下方铅垂向下。这个关系不随定义改变。",
        "世界系的三轴六方向相对世界：",
        "- 轴向：世界前后轴。正方向朝前。后方是其反方向。",
        "- 法向：世界铅垂轴。正方向朝上，上方铅垂向上。下方铅垂向下。",
        "- 侧向：世界左右轴。正方向朝右。左侧是其反方向。",
        "相机的三轴六方向相对相机：",
        "- 轴向：光轴。正方向朝前，指向拍摄方向。后方是其反方向。",
        "- 法向：像面纵向。正方向朝上。下方是其反方向。",
        "- 侧向：像面横向。正方向朝右。左侧是其反方向。",
        "",
        "场景",
        "两台已标定、同步曝光的相机从不同方位拍摄同一场景。相机光轴沿相机轴向，像面横向沿相机侧向，像面纵向沿相机法向。",
        "其中有一类特殊任务：给定世界系中过世界原点、沿世界轴的直线，先确定相机姿态角，再把这些直线投影到两幅图像中，进而求解双相机外参与极线约束、三角化。",
        "",
        "角度",
        "相机方位角 ψ：光轴偏向世界右侧时为正。",
        "相机俯仰角 θ：光轴偏向世界上方时为正。",
        "相机滚转角 φ：像面右缘偏向相机下方时为正。",
        "相机按方位角、俯仰角、滚转角的顺序内旋。",
        "",
        "坐标系",
        f"世界系（{chirality[spec['world'].get('chirality','left')]}手系）：{axes_text(('Xw','Yw','Zw'), scene.wind)}。",
        f"相机1系（{chirality[spec['camera1'].get('chirality','left')]}手系）：{axes_text(('X1','Y1','Z1'), scene.cam1)}。",
        f"相机2系（{chirality[spec['camera2'].get('chirality','left')]}手系）：{axes_text(('X2','Y2','Z2'), scene.cam2)}。",
        "",
        "给定（符号）",
        "相机光心 C1=(x1,y1,z1)、C2=(x2,y2,z2)；内参 K1=[[fx1,0,cx1],[0,fy1,cy1],[0,0,1]]、K2=[[fx2,0,cx2],[0,fy2,cy2],[0,0,1]]。K 按 [侧向正方向（右）, 法向正方向（上）, 轴向正方向（前）] 作用。S 把该相机坐标变到这个顺序，含取反。有效内参为 K·S。",
        "世界原点 O=(0,0,0)。",
    ])


if __name__ == "__main__":
    import json
    for tier in (25, 50, 75, 100):
        sample = build(tier)
        out = __file__.replace("cv_ctrl_generate.py", f"sample_cvctrl_{tier}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False, indent=2)
        print(f"tier {tier}: {len(sample['questions'])} blanks -> {out}")
