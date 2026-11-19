# -*- coding: utf-8 -*-
"""Add the mechanism-angle definitions to all flight-mechanics declarations."""
from pathlib import Path

MECH = [
    '        "机构角",',
    '        "机构俯仰角 α_M：绕侧向轴旋转。",',
    '        "机构偏航角 β_M：绕法向轴旋转。",',
    '        "机构滚转角 γ_M：绕轴向轴旋转。",',
    '        "",',
    '        "坐标系",',
]
OLD = ['        lines.extend(["", "坐标系"])']

files = [
    r"C:\Users\minec\Desktop\ACL\workflow\domains\aerodynamics\dataset_generate.py",
    r"C:\Users\minec\Desktop\ACL\workflow\domains\aerodynamics\ground_track_air_dataset.py",
    r"C:\Users\minec\Desktop\ACL\workflow\domains\aerodynamics\body_air_dataset.py",
]

for path in files:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    old = OLD[0]
    if old not in text:
        # try alternate pattern
        old = 'lines.extend(["", "坐标系"])'
    new = "\n".join(MECH)
    if old in text:
        text = text.replace(old, new, 1)
        p.write_text(text, encoding="utf-8")
        print("patched", path)
    else:
        print("SKIP (pattern not found):", path)
