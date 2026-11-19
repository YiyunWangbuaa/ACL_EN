# -*- coding: utf-8 -*-
"""推理成本分析：传统飞行力学问题，正常表达 vs 异构表达。

对比 def_probe 各档（000=正常表达，025/050/075/100=异构表达）在
「地面系到机体系 / 地面系转航迹系和气流系 / 体轴系到气流系」三个传统
坐标系变换子过程上的推理 token、总 token、美元成本与延迟。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

BASE = r"C:\Users\minec\Desktop\ACL\workflow\results\flight_flash_think"
CONDITIONS = ["def_probe_A", "def_probe_B", "def_probe_E"]
TIER_ORDER = [0, 25, 50, 75, 100]
TIER_LABEL = {0: "正常表达(000)", 25: "异构25%", 50: "异构50%", 75: "异构75%", 100: "异构100%"}


def _reasoning_tokens(r: dict) -> int:
    u = r.get("usage") or {}
    ctd = u.get("completion_tokens_details") or {}
    return ctd.get("reasoning_tokens") or 0


def _completion_tokens(r: dict) -> int:
    return (r.get("usage") or {}).get("completion_tokens") or 0


def _total_tokens(r: dict) -> int:
    return (r.get("usage") or {}).get("total_tokens") or 0


def _cost_usd(r: dict) -> float:
    return (r.get("cost") or {}).get("usd") or 0.0


def _latency(r: dict) -> int:
    return r.get("latency_ms") or 0


def analyze(condition: str) -> dict:
    path = os.path.join(BASE, condition, "records.jsonl")
    records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    by_tier = defaultdict(lambda: {
        "correct": 0, "total": 0, "reasoning": 0, "completion": 0, "tokens": 0,
        "cost": 0.0, "latency": 0, "n_records": 0,
    })
    for r in records:
        t = r["tier"]
        d = by_tier[t]
        d["n_records"] += 1
        d["reasoning"] += _reasoning_tokens(r)
        d["completion"] += _completion_tokens(r)
        d["tokens"] += _total_tokens(r)
        d["cost"] += _cost_usd(r)
        d["latency"] += _latency(r)
        for b in r["blanks"]:
            d["total"] += 1
            if b.get("normalized"):
                d["correct"] += 1
    return by_tier


def _fmt(v: float, digits: int = 1) -> str:
    return f"{v:.{digits}f}"


def _load_jsonl(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def analyze_e2e() -> None:
    """端到端版：标准定义一条，加上 pilot_final-e2e 的四档异构。只计归一化正确率。"""
    records = _load_jsonl(os.path.join(BASE, "standard_final-e2e", "records.jsonl"))
    records += _load_jsonl(os.path.join(BASE, "pilot_final-e2e", "records.jsonl"))
    grouped = defaultdict(list)
    correct = defaultdict(int)
    total = defaultdict(int)
    for record in records:
        usage = record.get("usage") or {}
        grouped[record["tier"]].append((
            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0,
            usage.get("total_tokens") or 0,
            (record.get("cost") or {}).get("usd") or 0.0,
            record.get("latency_ms") or 0,
        ))
        for blank in record["blanks"]:
            total[record["tier"]] += 1
            if blank.get("normalized"):
                correct[record["tier"]] += 1

    def mean(tier: int, index: int) -> float:
        values = grouped[tier]
        return sum(item[index] for item in values) / len(values)

    print("# 端到端推理成本（标准定义 + 异构四档，归一化正确率）\n")
    print("| 档 | 调用数 | 归一化正确率 | 思考token/次 | 总token/次 | 美元/次 | 延迟ms/次 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    order = [tier for tier in TIER_ORDER if tier in grouped]
    for tier in order:
        label = "标准定义" if tier == 0 else f"异构{tier}%"
        acc = 100 * correct[tier] / max(total[tier], 1)
        print(
            f"| {label} | {len(grouped[tier])} | {acc:.0f}% | {mean(tier, 0):.0f} | "
            f"{mean(tier, 1):.0f} | {mean(tier, 2):.5f} | {mean(tier, 3):.0f} |"
        )

    baseline = 0 if 0 in grouped else 25
    base_label = "标准定义" if baseline == 0 else "异构25%"
    print(f"\n相对{base_label}的增幅：")
    print("| 档 | 思考token | 总token | 美元 | 延迟 |")
    print("| --- | --- | --- | --- | --- |")
    for tier in order:
        if tier == baseline:
            continue
        label = "标准定义" if tier == 0 else f"异构{tier}%"
        cells = [100 * (mean(tier, index) / mean(baseline, index) - 1) for index in range(4)]
        print(
            f"| {label} | {cells[0]:+.1f}% | {cells[1]:+.1f}% | {cells[2]:+.1f}% | {cells[3]:+.1f}% |"
        )


def main() -> None:
    print("# 推理成本分析：传统飞行力学问题，正常表达 vs 异构表达\n")
    print("数据：deepseek-flash，thinking=enabled，condition A/B/E（三种注入方式）\n")

    rows_all = defaultdict(list)
    for cond in CONDITIONS:
        by_tier = analyze(cond)
        # 打印该 condition 的表
        print(f"## condition {cond.split('_')[-1]}\n")
        print("| 档 | 正确率 | 每空思考token | 每空总token | 每空美元 | 每空延迟ms |")
        print("| --- | --- | --- | --- | --- | --- |")
        for t in TIER_ORDER:
            d = by_tier[t]
            acc = 100 * d["correct"] / max(d["total"], 1)
            rt = d["reasoning"] / max(d["total"], 1)
            tt = d["tokens"] / max(d["total"], 1)
            usd = d["cost"] / max(d["total"], 1)
            lat = d["latency"] / max(d["total"], 1)
            rows_all[t].append((acc, rt, tt, usd, lat))
            print(f"| {TIER_LABEL[t]} | {_fmt(acc)}% | {_fmt(rt)} | {_fmt(tt)} | {_fmt(usd, 5)} | {_fmt(lat, 0)} |")
        print()

    # 跨 condition 平均
    print("## 三条件平均\n")
    print("| 档 | 正确率 | 每空思考token | 每空总token | 每空美元 | 每空延迟ms |")
    print("| --- | --- | --- | --- | --- | --- |")
    for t in TIER_ORDER:
        vals = rows_all[t]
        n = len(vals)
        acc = sum(v[0] for v in vals) / n
        rt = sum(v[1] for v in vals) / n
        tt = sum(v[2] for v in vals) / n
        usd = sum(v[3] for v in vals) / n
        lat = sum(v[4] for v in vals) / n
        print(f"| {TIER_LABEL[t]} | {_fmt(acc)}% | {_fmt(rt)} | {_fmt(tt)} | {_fmt(usd, 5)} | {_fmt(lat, 0)} |")

    # 相对增幅
    print("\n## 相对正常表达(000)的增幅\n")
    print("| 档 | 正确率差 | 思考token增幅 | 总token增幅 | 美元增幅 |")
    print("| --- | --- | --- | --- | --- |")
    base_vals = rows_all[0]
    base_acc = sum(v[0] for v in base_vals) / len(base_vals)
    base_rt = sum(v[1] for v in base_vals) / len(base_vals)
    base_tt = sum(v[2] for v in base_vals) / len(base_vals)
    base_usd = sum(v[3] for v in base_vals) / len(base_vals)
    for t in TIER_ORDER:
        vals = rows_all[t]
        n = len(vals)
        acc = sum(v[0] for v in vals) / n
        rt = sum(v[1] for v in vals) / n
        tt = sum(v[2] for v in vals) / n
        usd = sum(v[3] for v in vals) / n
        print(f"| {TIER_LABEL[t]} | {_fmt(acc - base_acc)}pp | {_fmt(100 * (rt / base_rt - 1))}% | "
              f"{_fmt(100 * (tt / base_tt - 1))}% | {_fmt(100 * (usd / base_usd - 1))}% |")


if __name__ == "__main__":
    analyze_e2e()
    print()
    main()
