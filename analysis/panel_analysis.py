# -*- coding: utf-8 -*-
"""Aggregate the five-model panel (DeepSeek/Qwen/Kimi) A/B/C/D/E runs.

Run:
    python analysis/panel_analysis.py

Outputs:
    results/panel_stats.json
    results/panel_summary.md
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# model label -> {condition: run dir}
MODELS = [
    ("DeepSeek V4 Flash", {"A": "run_A", "B": "run_B", "C": "run_C", "D": "run_D", "E": "run_E"}),
    ("Qwen3.8 Flash", {"A": "run_A_qwen_flash", "B": "run_B_qwen_flash",
                       "C": "run_C_qwen_flash", "D": "run_D_qwen_flash", "E": "run_E_qwen_flash"}),
    ("Qwen3.8 27B", {"A": "run_A_qwen_27b", "B": "run_B_qwen_27b",
                     "C": "run_C_qwen_27b", "D": "run_D_qwen_27b", "E": "run_E_qwen_27b"}),
    ("Kimi K3", {"A": "run_A_kimi_k3", "B": "run_B_kimi_k3",
                 "C": "run_C_kimi_k3", "D": "run_D_kimi_k3", "E": "run_E_kimi_k3"}),
]
CONDITIONS = ["A", "B", "C", "D", "E"]
PAIRS = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "B"), ("E", "A")]
LABELS = ["C1", "C2", "A1", "A2"]
TIERS = [25, 50, 75, 100]
PROCESSES = ["风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点"]


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_ci(deltas: list[float], reps: int = 3000, seed: int = 7) -> tuple[float, float]:
    random.seed(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(reps):
        sample = [deltas[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def load():
    """Return {model: {condition: {key: item}}}, {model: {condition: meta}}."""
    data: dict[str, dict[str, dict[tuple, dict]]] = {}
    meta: dict[str, dict[str, dict]] = {}
    for model, conds in MODELS:
        data[model] = {}
        meta[model] = {}
        for cond, run in conds.items():
            path = RESULTS / run / "records.jsonl"
            if not path.is_file():
                continue
            items = {}
            cost = 0.0
            prompt_tokens = 0
            completion_tokens = 0
            calls = 0
            for line in path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                calls += 1
                usage = rec.get("usage") or {}
                prompt_tokens += usage.get("prompt_tokens", 0)
                completion_tokens += usage.get("completion_tokens", 0)
                cost += (rec.get("cost") or {}).get("usd", 0.0) or 0.0
                for b in rec["blanks"]:
                    key = (rec["sample_id"], rec["process"], b["blank_id"])
                    items[key] = {
                        "sample": rec["sample_id"],
                        "tier": rec["tier"],
                        "process": rec["process"],
                        "blank": b["blank_id"],
                        "label": b["label"],
                        "fmt": b["format_code"],
                        "exact": bool(b["exact"]),
                        "norm": bool(b["normalized"]),
                        "ek": b["error_kind"] or "",
                        "answer": b.get("answer", ""),
                        "pred": b.get("prediction", ""),
                        "options": b.get("options") or [],
                    }
            data[model][cond] = items
            meta[model][cond] = {
                "run": run,
                "calls": calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost": cost,
            }
    return data, meta


def summarize(items: dict[tuple, dict]) -> dict:
    rows = list(items.values())
    n = len(rows)
    exact = sum(1 for i in rows if i["exact"])
    norm = sum(1 for i in rows if i["norm"])
    labels = {}
    for label in LABELS:
        sub = [i for i in rows if i["label"] == label]
        labels[label] = {
            "n": len(sub),
            "exact": sum(1 for i in sub if i["exact"]),
            "norm": sum(1 for i in sub if i["norm"]),
        }
    tiers = {}
    for tier in TIERS:
        sub = [i for i in rows if i["tier"] == tier]
        tiers[tier] = {
            "n": len(sub),
            "exact": sum(1 for i in sub if i["exact"]),
            "norm": sum(1 for i in sub if i["norm"]),
        }
    processes = {}
    for proc in PROCESSES:
        sub = [i for i in rows if i["process"] == proc]
        processes[proc] = {
            "n": len(sub),
            "exact": sum(1 for i in sub if i["exact"]),
            "norm": sum(1 for i in sub if i["norm"]),
        }
    eks = Counter(i["ek"] for i in rows if not i["norm"])
    return {
        "n": n,
        "exact": exact,
        "norm": norm,
        "exact_rate": exact / n if n else 0.0,
        "norm_rate": norm / n if n else 0.0,
        "format": (n - exact) - (n - norm),
        "labels": labels,
        "tiers": tiers,
        "processes": processes,
        "error_kinds": dict(eks),
        "defs": eks.get("定义未用", 0),
        "other": eks.get("其他", 0),
    }


def paired(a: dict, b: dict, keys: list[tuple], key: str = "norm") -> dict:
    a_only = b_only = both = neither = 0
    for k in keys:
        ia = a.get(k)
        ib = b.get(k)
        if ia is None or ib is None:
            continue
        x, y = ia[key], ib[key]
        if x and y:
            both += 1
        elif x:
            a_only += 1
        elif y:
            b_only += 1
        else:
            neither += 1
    n = both + a_only + b_only + neither
    return {
        "n": n,
        "a": (both + a_only) / n if n else 0.0,
        "b": (both + b_only) / n if n else 0.0,
        "delta": (b_only - a_only) / n if n else 0.0,
        "a_only": a_only,
        "b_only": b_only,
        "p": mcnemar(b_only, a_only),
    }


def sample_ci(a: dict, b: dict, keys: list[tuple], key: str = "norm") -> list[float]:
    samples = sorted({k[0] for k in keys})
    deltas = []
    for s in samples:
        sub = [k for k in keys if k[0] == s]
        deltas.append(paired(a, b, sub, key)["delta"])
    lo, hi = cluster_ci(deltas)
    return [lo, hi]


def main() -> None:
    data, meta = load()
    stats: dict = {"models": {}, "cross": {}, "meta": meta}
    for model, conds in MODELS:
        m = {}
        for cond in conds:
            if cond not in data[model]:
                continue
            m[cond] = summarize(data[model][cond])
            m[cond]["cost"] = meta[model][cond]["cost"]
            m[cond]["prompt_tokens"] = meta[model][cond]["prompt_tokens"]
            m[cond]["completion_tokens"] = meta[model][cond]["completion_tokens"]
        stats["models"][model] = m

    # within-model paired comparisons
    for model, conds in MODELS:
        for a, b in PAIRS:
            ca = a if a in conds else ("A*" if a == "A" and "A*" in conds else None)
            cb = b if b in conds else ("A*" if b == "A" and "A*" in conds else None)
            if not ca or not cb:
                continue
            da = data[model].get(ca)
            db = data[model].get(cb)
            if not da or not db:
                continue
            keys = sorted(set(da) & set(db))
            p = paired(da, db, keys)
            p["ci"] = sample_ci(da, db, keys)
            stats["models"][model].setdefault("pairs", {})[f"{a}|{b}"] = p

    # cross-model, same condition
    available = {model: set(cond for cond in conds if cond in data[model]) for model, conds in MODELS}
    for cond in CONDITIONS:
        models = [m for m in available if cond in available[m]]
        rows = []
        for m in models:
            s = stats["models"][m][cond]
            rows.append({"model": m, "exact": s["exact_rate"], "norm": s["norm_rate"], "n": s["n"]})
        stats["cross"][cond] = rows

    out = RESULTS / "panel_stats.json"
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    lines = ["# 四模型面板汇总（A/B/C/D/E）", ""]
    lines.append("10 条 pilot 样本 × 193 空 = 1,930 空/条件；严格与归一化两套口径。")
    lines.append("")
    lines.append("## 严格正确率")
    lines.append("")
    header = "| 模型 | " + " | ".join(CONDITIONS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(CONDITIONS) + 1))
    for model, m in stats["models"].items():
        cells = []
        for cond in CONDITIONS:
            if cond in m:
                s = m[cond]
                cells.append(f'{s["exact"]}/{s["n"]}（{pct(s["exact_rate"])}）')
            else:
                cells.append("—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 条件内配对差（归一化，后者 − 前者）")
    lines.append("")
    lines.append("| 模型 | B−A | C−A | D−B | D−C | E−B | E−A |")
    lines.append("|---|---|---|---|---|---|---|")
    for model, m in stats["models"].items():
        pairs = m.get("pairs", {})
        cells = []
        for a, b in PAIRS:
            p = pairs.get(f"{a}|{b}")
            cells.append(f'{p["delta"]*100:+.1f}' if p else "—")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## 按条件跨模型")
    lines.append("")
    for cond in CONDITIONS:
        rows = stats["cross"].get(cond, [])
        if not rows:
            continue
        lines.append(f"### {cond}")
        lines.append("")
        for r in sorted(rows, key=lambda x: -x["norm"]):
            lines.append(f'- {r["model"]}：严格 {pct(r["exact"])}，归一化 {pct(r["norm"])}（{r["n"]} 空）')
        lines.append("")
    lines.append("## 成本（USD，按各 run 记录）")
    lines.append("")
    for model, m in stats["models"].items():
        total = sum(m[c]["cost"] for c in m if c != "pairs")
        calls = sum(meta[model][c]["calls"] for c in meta[model])
        lines.append(f"- {model}：${total:.2f}，{calls} 次调用")
    lines.append("")
    (RESULTS / "panel_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
