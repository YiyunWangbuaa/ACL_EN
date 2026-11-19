# -*- coding: utf-8 -*-
"""诊断：定位 CV 各套数据中「外参矩阵求解」错误的精确分布。"""
from __future__ import annotations

import json
import os
from collections import defaultdict

WORK = r"C:\Users\minec\Desktop\ACL\workflow"
DATA = {
    "cross": (os.path.join(WORK, "data", "archive", "wind_vision", "samples.jsonl"), "025-001"),
    "reconstruction": (os.path.join(WORK, "data", "vision", "reconstruction", "samples.jsonl"), "025-001"),
    "basic": (os.path.join(WORK, "data", "vision", "basic", "samples.jsonl"), "basic-02"),
    "basic_standard": (os.path.join(WORK, "data", "vision", "basic", "samples.jsonl"), "basic-01"),
}
RESULTS = os.path.join(WORK, "results", "cv_layers_flash_a")


def category_of(q: dict) -> str:
    prompt = q.get("prompt", "")
    fmt = q.get("format_code", "")
    if any(k in prompt for k in ("R_cw", "R_cg")):
        return "外参-旋转R_cw/R_cg"
    if any(k in prompt for k in ("t_cw",)):
        return "外参-平移t_cw"
    if any(k in prompt for k in ("R21", "t21")):
        return "外参-双目R21/t21"
    if any(k in prompt for k in ("R_ψ", "R_θ", "R_φ", "R_ψ1", "R_θ1", "R_φ1")):
        return "姿态-欧拉R_ψ/θ/φ"
    if "K" in prompt and ("^(-T)" in prompt or "^(-1)" in prompt or "内参" in prompt):
        return "投影-内参K"
    if "[P_c" in prompt or "反对称" in prompt:
        return "投影-反对称[P]_×"
    if "P_c" in prompt or "P_w" in prompt:
        return "投影-点P_c/P_w"
    if "l =" in prompt or "直线 l" in prompt or "图像直线" in prompt:
        return "投影-直线公式"
    if "d_w" in prompt or "方向向量" in prompt:
        return "直线-方向d_w"
    if "E =" in prompt or "本质矩阵" in prompt or "F =" in prompt or "基础矩阵" in prompt or "极线" in prompt:
        return "极线-E/F/约束"
    if "反推" in prompt or "世界坐标" in prompt and "图像" in prompt:
        return "反投影"
    return "定义/选择"


def analyze(group: str) -> dict:
    data_path, sample_id = DATA[group]
    rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
    sample = next(r for r in rows if r["id"] == sample_id)
    questions = sample["questions"]
    by_id = {q["id"]: q for q in questions}

    rec_path = os.path.join(RESULTS, group, "records.jsonl")
    records = [json.loads(line) for line in open(rec_path, encoding="utf-8") if line.strip()]

    # blank_id -> normalized
    result = {}
    for rec in records:
        for b in rec["blanks"]:
            result[b["blank_id"]] = bool(b.get("normalized"))

    per_cat = defaultdict(lambda: [0, 0])
    per_seg = defaultdict(lambda: [0, 0])
    per_fmt = defaultdict(lambda: [0, 0])
    for q in questions:
        bid = q["id"]
        if bid not in result:
            continue
        ok = result[bid]
        cat = category_of(q)
        per_cat[cat][0] += 1
        per_cat[cat][1] += (1 if ok else 0)
        seg = q.get("segment", "?")
        per_seg[seg][0] += 1
        per_seg[seg][1] += (1 if ok else 0)
        per_fmt[q.get("format_code", "?")][0] += 1
        per_fmt[q.get("format_code", "?")][1] += (1 if ok else 0)
    return {"cat": dict(per_cat), "seg": dict(per_seg), "fmt": dict(per_fmt)}


def pct(c, n):
    return f"{100 * c / n:.1f}%" if n else "-"


def cell(entry):
    total, correct = entry
    return f"{correct}/{total} = {pct(correct, total)}"


def main():
    all_cats = ["外参-旋转R_cw/R_cg", "外参-平移t_cw", "外参-双目R21/t21",
                "姿态-欧拉R_ψ/θ/φ", "投影-内参K", "投影-反对称[P]_×", "投影-点P_c/P_w",
                "投影-直线公式", "直线-方向d_w", "极线-E/F/约束", "反投影", "定义/选择"]
    results = {g: analyze(g) for g in DATA}

    print("## 各类正确率\n")
    print("| 类别 | cross | reconstruction | basic | basic_standard |")
    print("| --- | --- | --- | --- | --- |")
    for cat in all_cats:
        cells = [cell(results[g]["cat"].get(cat, [0, 0])) for g in DATA]
        print(f"| {cat} | " + " | ".join(cells) + " |")

    print("\n## 分段正确率\n")
    seg_order = ["frame", "camera_attitude", "body_attitude", "projection", "axis_wind",
                 "line_direction", "extrinsic", "epipolar", "backproject"]
    print("| 段 | cross | reconstruction | basic | basic_standard |")
    print("| --- | --- | --- | --- | --- |")
    for seg in seg_order:
        if not any(seg in results[g]["seg"] for g in DATA):
            continue
        cells = [cell(results[g]["seg"].get(seg, [0, 0])) for g in DATA]
        print(f"| {seg} | " + " | ".join(cells) + " |")

    print("\n## 按题型（格式）正确率\n")
    print("| 格式 | cross | reconstruction | basic | basic_standard |")
    print("| --- | --- | --- | --- | --- |")
    for fmt in ["无", "矩阵", "向量", "表达式"]:
        if not any(fmt in results[g]["fmt"] for g in DATA):
            continue
        cells = [cell(results[g]["fmt"].get(fmt, [0, 0])) for g in DATA]
        print(f"| {fmt} | " + " | ".join(cells) + " |")

    # 汇总：外参矩阵(旋转+平移+双目) vs 其他
    print("\n## 外参矩阵 vs 其他（汇总）\n")
    ext_cats = {"外参-旋转R_cw/R_cg", "外参-平移t_cw", "外参-双目R21/t21"}
    print("| 组 | 外参矩阵正确率 | 其他正确率 |")
    print("| --- | --- | --- |")
    for g in DATA:
        cat = results[g]["cat"]
        ext_c = sum(v[1] for k, v in cat.items() if k in ext_cats)
        ext_n = sum(v[0] for k, v in cat.items() if k in ext_cats)
        oth_c = sum(v[1] for k, v in cat.items() if k not in ext_cats)
        oth_n = sum(v[0] for k, v in cat.items() if k not in ext_cats)
        print(f"| {g} | {pct(ext_c, ext_n)} ({ext_c}/{ext_n}) | {pct(oth_c, oth_n)} ({oth_c}/{oth_n}) |")


if __name__ == "__main__":
    main()
