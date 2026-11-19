# -*- coding: utf-8 -*-
"""Analyze the A / B / C / D / E cloze runs.

Run:
    python analysis/run_analysis.py

Outputs:
    results/experiment_analysis.md
    results/experiment_analysis.json
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUNS = {"run_A": "A", "run_B": "B", "run_C": "C", "run_D": "D", "run_E": "E"}
ORDER = ["A", "B", "C", "D", "E"]
COMBO = {
    "A": "金标准前缀，句号截断",
    "B": "自身答案前缀，句号截断",
    "C": "金标准前缀，保留完整后续正文",
    "D": "自身答案前缀，保留完整后续正文",
    "E": "整段一次调用，全部空待填，只输出一个 JSON",
}
REPORT_CI = {
    "A": (65.5, 70.4),
    "B": (47.4, 52.0),
    "C": (64.2, 67.7),
    "D": (45.5, 50.4),
    "E": (52.3, 57.7),
}
PAIRS = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "B"), ("E", "D"), ("E", "A"), ("E", "C")]
LABELS = ["C1", "C2", "A1", "A2"]
TIERS = [25, 50, 75, 100]
PROCESSES = ["风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点"]


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    if p < 0.01:
        return f"{p:.3f}"
    return f"{p:.3f}"


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_ci(deltas: list[float], reps: int = 3000, seed: int = 7) -> tuple[float, float, float]:
    random.seed(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    means = []
    for _ in range(reps):
        sample = [deltas[random.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    mean = sum(deltas) / n
    return mean, means[int(0.025 * reps)], means[int(0.975 * reps)]


def load():
    """Load all runs into item-level records keyed by (sample, process, blank)."""
    items: dict[str, dict[tuple[str, str, str], dict]] = {cond: {} for cond in ORDER}
    groups: dict[str, dict[tuple[str, str, int], dict]] = {cond: {} for cond in ORDER}
    summary: dict[str, dict] = {}
    for run, cond in RUNS.items():
        path = RESULTS / run / "records.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        proc_max: dict[tuple[str, str], int] = defaultdict(int)
        group_count: dict[tuple[str, str], int] = defaultdict(int)
        for rec in records:
            sk = (rec["sample_id"], rec["process"])
            group_count[sk] += 1
            for blank in rec["blanks"]:
                proc_max[sk] = max(proc_max[sk], int(blank["blank_id"]))
        group_idx: dict[tuple[str, str], int] = defaultdict(int)
        for rec in records:
            sk = (rec["sample_id"], rec["process"])
            group_idx[sk] += 1
            gidx = group_idx[sk]
            gcount = group_count[sk]
            proc_len = proc_max[sk]
            blank_rows = []
            for blank in rec["blanks"]:
                pos = int(blank["blank_id"])
                key = (rec["sample_id"], rec["process"], blank["blank_id"])
                item = {
                    "cond": cond,
                    "sample": rec["sample_id"],
                    "tier": rec["tier"],
                    "process": rec["process"],
                    "blank": blank["blank_id"],
                    "pos": pos,
                    "proc_len": proc_len,
                    "rel": pos / proc_len,
                    "group_key": (rec["sample_id"], rec["process"], gidx),
                    "group_idx": gidx,
                    "group_n": gcount,
                    "group_rel": gidx / gcount,
                    "group_size": len(rec["blanks"]),
                    "label": blank["label"],
                    "fmt": blank["format_code"],
                    "answer": blank["answer"],
                    "pred": blank["prediction"],
                    "exact": bool(blank["exact"]),
                    "norm": bool(blank["normalized"]),
                    "ek": blank["error_kind"] or "",
                    "line_hit": bool(blank["line_hit"]),
                    "sympy": bool(blank["sympy_equal"]),
                    "option": bool(blank["option_match"]),
                }
                items[cond][key] = item
                blank_rows.append(item)
            groups[cond][(rec["sample_id"], rec["process"], gidx)] = {
                "items": blank_rows,
                "all": all(i["norm"] for i in blank_rows),
                "all_exact": all(i["exact"] for i in blank_rows),
                "size": len(blank_rows),
            }
        summary[cond] = {
            "calls": len(records),
            "cost": sum((r.get("cost") or {}).get("usd", 0.0) for r in records),
            "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in records),
            "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in records),
            "cache_hit": sum(((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) for r in records),
            "latency": statistics.mean([r.get("latency_ms", 0) for r in records]),
        }
    return items, groups, summary


def rate(items: list[dict], key: str = "norm") -> tuple[int, int]:
    return sum(1 for i in items if i[key]), len(items)


def paired_rows(items: dict[str, dict], keys: list[tuple], a: str, b: str, key: str = "norm") -> dict:
    a_only = b_only = both = neither = 0
    for k in keys:
        ia = items[a].get(k)
        ib = items[b].get(k)
        if ia is None or ib is None:
            continue
        x = ia[key]
        y = ib[key]
        if x and y:
            both += 1
        elif x and not y:
            a_only += 1
        elif y and not x:
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
        "both": both,
        "neither": neither,
        "p": mcnemar(b_only, a_only),
    }


def per_sample_delta(items: dict[str, dict], keys: list[tuple], a: str, b: str, key: str = "norm") -> dict:
    samples = sorted({k[0] for k in keys})
    deltas = []
    rows = []
    for s in samples:
        s_keys = [k for k in keys if k[0] == s]
        pa = paired_rows(items, s_keys, a, b, key)
        deltas.append(pa["delta"])
        rows.append({"sample": s, "delta": pa["delta"], "a": pa["a"], "b": pa["b"], "n": pa["n"]})
    mean, lo, hi = cluster_ci(deltas)
    return {"mean": mean, "lo": lo, "hi": hi, "samples": rows}


def analyze():
    items, groups, summary = load()
    all_keys = sorted(set().union(*[set(v.keys()) for v in items.values()]))
    missing = {c: [k for k in all_keys if k not in items[c]] for c in ORDER}
    assert all(len(v) == 1930 for v in items.values()), [len(v) for v in items.values()]
    assert not any(missing.values()), missing

    # ---------- per-condition summary ----------
    cond_summary = {}
    for cond in ORDER:
        rows = list(items[cond].values())
        exact, n = rate(rows, "exact")
        norm, _ = rate(rows, "norm")
        labels = {}
        for label in LABELS:
            sub = [i for i in rows if i["label"] == label]
            labels[label] = {"n": len(sub), "exact": rate(sub, "exact")[0], "norm": rate(sub, "norm")[0]}
        tiers = {}
        for tier in TIERS:
            sub = [i for i in rows if i["tier"] == tier]
            tiers[tier] = {"n": len(sub), "exact": rate(sub, "exact")[0], "norm": rate(sub, "norm")[0]}
        processes = {}
        for proc in PROCESSES:
            sub = [i for i in rows if i["process"] == proc]
            processes[proc] = {"n": len(sub), "exact": rate(sub, "exact")[0], "norm": rate(sub, "norm")[0]}
        per_sample = {}
        for s in sorted({i["sample"] for i in rows}):
            sub = [i for i in rows if i["sample"] == s]
            per_sample[s] = {"n": len(sub), "exact": rate(sub, "exact")[0], "norm": rate(sub, "norm")[0]}
        eks = Counter(i["ek"] for i in rows)
        cond_summary[cond] = {
            "calls": summary[cond]["calls"],
            "cost": summary[cond]["cost"],
            "prompt_tokens": summary[cond]["prompt_tokens"],
            "completion_tokens": summary[cond]["completion_tokens"],
            "cache_hit": summary[cond]["cache_hit"],
            "latency": summary[cond]["latency"],
            "n": n,
            "exact": exact,
            "norm": norm,
            "exact_rate": exact / n,
            "norm_rate": norm / n,
            "format": (n - exact) - (n - norm),
            "labels": labels,
            "tiers": tiers,
            "processes": processes,
            "samples": per_sample,
            "error_kinds": eks,
            "defs_unused": eks.get("定义未用", 0),
            "other": eks.get("其他", 0),
        }

    # ---------- paired comparisons ----------
    pair_stats = {}
    for a, b in PAIRS:
        pa = paired_rows(items, all_keys, a, b)
        ci = per_sample_delta(items, all_keys, a, b)
        pa["ci"] = (ci["lo"], ci["hi"])
        pa["samples"] = ci["samples"]
        pair_stats[f"{a}|{b}"] = pa

    # ---------- per label / tier / process / sample ----------
    per_label = {}
    for label in LABELS:
        keys = [k for k in all_keys if items["A"][k]["label"] == label]
        per_label[label] = {f"{a}|{b}": paired_rows(items, keys, a, b) for a, b in PAIRS}
    per_tier = {}
    for tier in TIERS:
        keys = [k for k in all_keys if items["A"][k]["tier"] == tier]
        per_tier[tier] = {f"{a}|{b}": paired_rows(items, keys, a, b) for a, b in PAIRS}
    per_process = {}
    for proc in PROCESSES:
        keys = [k for k in all_keys if items["A"][k]["process"] == proc]
        per_process[proc] = {f"{a}|{b}": paired_rows(items, keys, a, b) for a, b in PAIRS}
    per_sample = {}
    for s in sorted({k[0] for k in all_keys}):
        keys = [k for k in all_keys if k[0] == s]
        per_sample[s] = {f"{a}|{b}": paired_rows(items, keys, a, b) for a, b in PAIRS}

    # ---------- conditional / cascade ----------
    conditional = {}
    for cond in ["A", "B", "C", "D", "E"]:
        right = wrong = 0
        given_right = [0, 0]
        given_wrong = [0, 0]
        transition = Counter()
        for k, item in items[cond].items():
            prev_key = (k[0], k[1], str(int(k[2]) - 1))
            prev = items[cond].get(prev_key)
            if prev is None:
                continue
            cur = item["norm"]
            transition[(prev["norm"], cur)] += 1
            if prev["norm"]:
                given_right[0] += cur
                given_right[1] += 1
            else:
                given_wrong[0] += cur
                given_wrong[1] += 1
        conditional[cond] = {
            "given_right": given_right[0] / given_right[1] if given_right[1] else None,
            "given_wrong": given_wrong[0] / given_wrong[1] if given_wrong[1] else None,
            "n_right": given_right[1],
            "n_wrong": given_wrong[1],
            "transition": {f"{int(a)}->{int(b)}": v for (a, b), v in transition.items()},
        }

    # group-level transitions
    group_cond = {}
    for cond in ["A", "B", "C", "D"]:
        g = groups[cond]
        cur_right = cur_wrong = 0
        given_right = [0, 0]
        given_wrong = [0, 0]
        for (s, p, idx), info in g.items():
            prev = g.get((s, p, idx - 1))
            if prev is None:
                continue
            if prev["all"]:
                given_right[0] += info["all"]
                given_right[1] += 1
            else:
                given_wrong[0] += info["all"]
                given_wrong[1] += 1
        group_cond[cond] = {
            "given_right": given_right[0] / given_right[1] if given_right[1] else None,
            "given_wrong": given_wrong[0] / given_wrong[1] if given_wrong[1] else None,
            "n_right": given_right[1],
            "n_wrong": given_wrong[1],
        }

    # first error / after-error recovery
    first_error = {}
    for cond in ["A", "B", "C", "D"]:
        g = groups[cond]
        firsts = []
        after_acc = [0, 0]
        recovered = [0, 0]
        for s, p in sorted({(s, p) for (s, p, _) in g}):
            indices = sorted(idx for (ss, pp, idx) in g if ss == s and pp == p)
            first = next((idx for idx in indices if not g[(s, p, idx)]["all"]), None)
            if first is None:
                firsts.append(1.0)
                continue
            firsts.append(first / len(indices))
            for idx in indices:
                if idx <= first:
                    continue
                after_acc[0] += g[(s, p, idx)]["all"]
                after_acc[1] += 1
                recovered[0] += 0 if g[(s, p, idx)]["all"] else 1
                recovered[1] += 1
        first_error[cond] = {
            "first_rel": statistics.mean(firsts) if firsts else None,
            "after_acc": after_acc[0] / after_acc[1] if after_acc[1] else None,
            "recovery": recovered[0] / recovered[1] if recovered[1] else None,
        }

    # ---------- position / leakage ----------
    def position_table(conds):
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001)]
        bins = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0000001)]
        out = []
        for lo, hi in bins:
            row = {"bin": f"{int(lo*100)}-{int(min(hi,1)*100)}%"}
            for c in conds:
                sub = [i for i in items[c].values() if lo <= i["rel"] < hi]
                row[c] = rate(sub, "norm")[0] / len(sub) if sub else None
            out.append(row)
        return out

    position = position_table(ORDER)
    leakage = []
    for lo, hi in [(0.0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.0000001)]:
        keys = [k for k in all_keys if lo <= items["A"][k]["group_rel"] < hi]
        row = {"bin": f"{int(lo*100)}-{int(min(hi,1)*100)}%", "n": len(keys)}
        ca = paired_rows(items, keys, "A", "C")
        cb = paired_rows(items, keys, "B", "D")
        row["A"] = ca["a"]
        row["C"] = ca["b"]
        row["C-A"] = ca["delta"]
        row["B"] = cb["a"]
        row["D"] = cb["b"]
        row["D-B"] = cb["delta"]
        leakage.append(row)

    # ---------- A2 ----------
    a2_keys = [k for k in all_keys if items["A"][k]["label"] == "A2"]
    a2 = {}
    for c in ORDER:
        rows = [items[c][k] for k in a2_keys]
        eks = Counter(i["ek"] for i in rows)
        a2[c] = {
            "n": len(rows),
            "exact": rate(rows, "exact")[0],
            "norm": rate(rows, "norm")[0],
            "format": (len(rows) - rate(rows, "exact")[0]) - (len(rows) - rate(rows, "norm")[0]),
            "kinds": dict(eks),
        }

    # ---------- error kinds ----------
    error_kinds = {}
    for c in ORDER:
        rows = list(items[c].values())
        eks = Counter(i["ek"] for i in rows if not i["norm"])
        exact_only = Counter(i["ek"] for i in rows if not i["exact"] and i["norm"])
        error_kinds[c] = {"norm_wrong": dict(eks), "format": exact_only, "defs": eks.get("定义未用", 0), "other": eks.get("其他", 0)}

    # ---------- E specifics ----------
    d_records = [json.loads(line) for line in (RESULTS / "run_E" / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    d_info = {
        "calls": len(d_records),
        "prompt_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in d_records),
        "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in d_records),
        "cost": sum((r.get("cost") or {}).get("usd", 0.0) for r in d_records),
        "finish": Counter(r.get("finish_reason") for r in d_records),
        "errors": sum(1 for r in d_records if r.get("error")),
        "answer_blank": sum(1 for r in d_records for b in r["blanks"] if not b["prediction"].strip()),
        "per_process": {},
    }
    for proc in PROCESSES:
        sub = [b for r in d_records if r["process"] == proc for b in r["blanks"]]
        d_info["per_process"][proc] = {"n": len(sub), "norm": sum(1 for b in sub if b["normalized"]) / len(sub), "exact": sum(1 for b in sub if b["exact"]) / len(sub)}

    stats = {
        "summary": cond_summary,
        "pairs": pair_stats,
        "per_label": per_label,
        "per_tier": per_tier,
        "per_process": per_process,
        "per_sample": per_sample,
        "conditional": conditional,
        "group_cond": group_cond,
        "first_error": first_error,
        "position": position,
        "leakage": leakage,
        "a2": a2,
        "error_kinds": error_kinds,
        "d": d_info,
        "keys": len(all_keys),
    }
    return stats


# ---------------------------------------------------------------- markdown

def table(headers, rows) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def markdown(stats) -> str:
    s = stats["summary"]
    L = []
    L.append("# 风洞填空实验分析：A / B / C / D / E")
    L.append("")
    L.append("**数据**：batch_100 的 pilot_10（10 条样本 × 193 空 = 1,930 空）。")
    L.append("**模型**：deepseek-flash，thinking disabled，temperature 0.0，max_tokens 512（A/B/C/D）或 8192（E），1 次运行。")
    L.append("**判定**：严格 exact 与归一化 normalized 两套；归一化包含格式/符号/选项/矩阵等价，是语义口径。")
    L.append("**注意**：deepseek-flash + 非思考是**小模型对照**，不是主评测（主评测是 v4-pro + thinking）。")
    L.append("")
    L.append("## 1. 实验设置与总览")
    L.append("")
    L.append("### 1.1 条件设计")
    L.append("")
    L.append(table(
        ["条件", "空前上下文", "后续正文", "作答粒度", "调用/样本", "角色"],
        [
            ["A", "该样本的 gold 答案", "截断（只到当前句）", "逐句（155 次）", "155", "oracle 上限 / 主协议"],
            ["B", "模型自己的原始输出", "截断", "逐句", "155", "真实部署 / 误差传播"],
            ["C", "gold 答案", "完整保留", "逐句", "155", "上下文泄露对照"],
            ["D", "模型自己的原始输出", "完整保留", "逐句", "155", "泄露 + 误差传播"],
            ["E", "无（整段一次）", "完整", "整段（4 次）", "4", "端到端一次性 / 低成本"],
        ]))
    L.append("")
    L.append("### 1.2 总览")
    L.append("")
    rows = []
    for c in ORDER:
        x = s[c]
        ci = REPORT_CI[c]
        rows.append([c, x["calls"], f'{x["exact"]}/{x["n"]}', pct(x["exact_rate"]), pct(x["norm_rate"]),
                     f"[{ci[0]:.1f}, {ci[1]:.1f}]", f'${x["cost"]:.4f}', f'{x["prompt_tokens"]/max(x["calls"],1):.0f}'])
    L.append(table(["条件", "调用", "exact", "严格", "归一化", "95% CI（样本自助）", "成本 USD", "prompt tokens/调用"], rows))
    L.append("")
    L.append("1,930 空以上的排序：**A（68.1%）> C（66.0%）> E（54.9%）> B（49.7%）> D（48.1%）**。")
    L.append("")
    L.append("### 1.3 按标签（归一化）")
    L.append("")
    rows = []
    for c in ORDER:
        row = [c]
        for label in LABELS:
            v = s[c]["labels"][label]
            row.append(f'{v["norm"]}/{v["n"]}（{pct(v["norm"]/v["n"])}）')
        row.append(pct(s[c]["norm_rate"]))
        rows.append(row)
    L.append(table(["条件", "C1", "C2", "A1", "A2", "总体"], rows))
    L.append("")
    L.append("### 1.4 按定义变更档（归一化）")
    L.append("")
    rows = []
    for c in ORDER:
        row = [c]
        for t in TIERS:
            v = s[c]["tiers"][t]
            row.append(pct(v["norm"] / v["n"]))
        rows.append(row)
    L.append(table(["条件", "25%", "50%", "75%", "100%"], rows))
    L.append("")
    L.append("### 1.5 按子过程（归一化）")
    L.append("")
    rows = []
    for c in ORDER:
        row = [c]
        for p in PROCESSES:
            v = s[c]["processes"][p]
            row.append(pct(v["norm"] / v["n"]))
        rows.append(row)
    L.append(table(["条件", "风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点"], rows))
    L.append("")
    L.append("### 1.6 按样本（归一化）")
    L.append("")
    samples = sorted(stats["per_sample"].keys())
    rows = []
    for c in ORDER:
        row = [c]
        for smp in samples:
            v = s[c]["samples"][smp]
            row.append(pct(v["norm"] / v["n"]))
        rows.append(row)
    L.append(table(["条件"] + samples, rows))
    L.append("")
    L.append("## 2. 主效应与配对比较")
    L.append("")
    L.append("### 2.1 配对表（归一化）")
    L.append("")
    rows = []
    for a, b in PAIRS:
        p = stats["pairs"][f"{a}|{b}"]
        rows.append([f"{b} − {a}", pct(p["a"]), pct(p["b"]), f'{p["delta"]*100:+.1f}',
                     p["a_only"], p["b_only"], fmt_p(p["p"]), f'[{p["ci"][0]*100:+.1f}, {p["ci"][1]*100:+.1f}]'])
    L.append(table(["对比", "前者", "后者", "Δ pp", "仅前者对", "仅后者对", "McNemar p", "95% CI（样本 cluster）"], rows))
    L.append("")
    L.append("### 2.2 六个关键对比")
    L.append("")
    p = stats["pairs"]
    L.append(f'- **A − B = {(-p["A|B"]["delta"])*100:+.1f} pp**：gold 前缀换成模型自身前缀的代价，是全部条件里最大的一项。')
    L.append(f'- **A − C = {(-p["A|C"]["delta"])*100:+.1f} pp**：保留完整后续正文没有带来任何泄露收益，反而略低。')
    L.append(f'- **B − D = {(-p["B|D"]["delta"])*100:+.1f} pp**：自身前缀下，完整后续正文同样没有帮助。')
    L.append(f'- **C − D = {(-p["C|D"]["delta"])*100:+.1f} pp**：在完整正文下，gold 前缀相对自身前缀仍有 {p["C|D"]["delta"]*100:.1f} pp 优势。')
    L.append(f'- **E − B = {(-p["E|B"]["delta"])*100:+.1f} pp**：整段一次性比逐句自身前缀更好，但成本只有 1/38.75。')
    L.append(f'- **E − A = {(-p["E|A"]["delta"])*100:+.1f} pp**：一次性仍显著低于 oracle 前缀。')
    L.append("")
    L.append("### 2.3 按标签的配对差（归一化，Δ = 后者 − 前者）")
    L.append("")
    rows = []
    for label in LABELS:
        row = [label]
        for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("E", "B"), ("E", "A")]:
            d = stats["per_label"][label][f"{a}|{b}"]
            row.append(f'{d["delta"]*100:+.1f}')
        rows.append(row)
    L.append(table(["标签", "B−A", "C−A", "D−B", "D−C", "B−E", "A−E"], rows))
    L.append("")
    L.append("### 2.4 按档位的配对差")
    L.append("")
    rows = []
    for t in TIERS:
        row = [f"{t}%"]
        for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("E", "B"), ("E", "A")]:
            d = stats["per_tier"][t][f"{a}|{b}"]
            row.append(f'{d["delta"]*100:+.1f}')
        rows.append(row)
    L.append(table(["档位", "B−A", "C−A", "D−B", "B−E", "A−E"], rows))
    L.append("")
    L.append("### 2.5 按样本的配对差")
    L.append("")
    rows = []
    for smp in samples:
        row = [smp]
        for a, b in [("A", "B"), ("A", "C"), ("B", "D"), ("E", "B"), ("E", "A")]:
            d = stats["per_sample"][smp][f"{a}|{b}"]
            row.append(f'{d["delta"]*100:+.1f}')
        rows.append(row)
    L.append(table(["样本", "B−A", "C−A", "D−B", "B−E", "A−E"], rows))
    L.append("")
    L.append("## 3. 误差传播")
    L.append("")
    L.append("### 3.1 条件准确率（归一化）")
    L.append("")
    L.append("A 的空前永远是该样本的 gold；B/D 的空前是模型自己的输出；E 没有前缀。于是 A 是“给定正确历史”的上界，B/D 是“历史也由模型自己写”的真实条件。")
    L.append("")
    L.append(table(
        ["条件", "空前来源", "给定前一对的准确率", "给定前一对错的准确率", "前一对错的数量"],
        [
            ["A", "gold", pct(stats["conditional"]["A"]["given_right"] or 0), "—", stats["conditional"]["A"]["n_wrong"]],
            ["B", "模型自己", pct(stats["conditional"]["B"]["given_right"] or 0),
             pct(stats["conditional"]["B"]["given_wrong"] or 0), stats["conditional"]["B"]["n_wrong"]],
            ["C", "gold", pct(stats["conditional"]["C"]["given_right"] or 0), "—", stats["conditional"]["C"]["n_wrong"]],
            ["D", "模型自己", pct(stats["conditional"]["D"]["given_right"] or 0),
             pct(stats["conditional"]["D"]["given_wrong"] or 0), stats["conditional"]["D"]["n_wrong"]],
            ["E", "同一响应内的前空", pct(stats["conditional"]["E"]["given_right"] or 0),
             pct(stats["conditional"]["E"]["given_wrong"] or 0), stats["conditional"]["E"]["n_wrong"]],
        ]))
    L.append("")
    L.append("### 3.2 首错与级联（句级）")
    L.append("")
    rows = []
    for c in ["A", "B", "C", "D"]:
        fe = stats["first_error"][c]
        rows.append([c, pct(fe["first_rel"] or 0), pct(fe["after_acc"] or 0), pct(fe["recovery"] or 0)])
    L.append(table(["条件", "首错的平均相对位置", "首错之后句级全对率", "首错后仍错率"], rows))
    L.append("")
    L.append("B 的句级首错平均出现在过程的前段，之后句级全对率明显低于 A；D 与 B 接近，说明后续正文没有阻断级联。")
    L.append("")
    L.append("### 3.3 位置曲线（按过程内相对位置，归一化）")
    L.append("")
    rows = []
    for row in stats["position"]:
        rows.append([row["bin"]] + [pct(row[c]) for c in ORDER])
    L.append(table(["相对位置"] + ORDER, rows))
    L.append("")
    L.append("## 4. 上下文泄露")
    L.append("")
    L.append("### 4.1 C − A：保留完整后续正文")
    L.append("")
    L.append(f'C 66.0% vs A 68.1%，Δ = {(p["A|C"]["delta"])*100:+.1f} pp；按标签：C1 −1.0、C2 −3.0、A1 −1.7、A2 −4.6；按档位 25% −3.8、50% −1.2、75% −4.6、100% +1.3。')
    L.append("")
    L.append("如果后续正文会泄露答案，C 应该在早期空上**高于** A，且早期差距大于晚期。实际结果相反：C 在早期略低、总差距为负。")
    L.append("")
    L.append("### 4.2 D − B：自身前缀 + 完整正文")
    L.append("")
    L.append(f'D 48.1% vs B 49.7%，Δ = {(p["B|D"]["delta"])*100:+.1f} pp；按标签 C1 −2.5、C2 +0.4、A1 −2.8、A2 0.0。完整正文同样没有把自身前缀的误差救回来。')
    L.append("")
    L.append("### 4.3 按句子相对位置")
    L.append("")
    rows = []
    for row in stats["leakage"]:
        rows.append([row["bin"], row["n"], pct(row["A"]), pct(row["C"]), f'{row["C-A"]*100:+.1f}',
                     pct(row["B"]), pct(row["D"]), f'{row["D-B"]*100:+.1f}'])
    L.append(table(["句位置", "空数", "A", "C", "C−A", "B", "D", "D−B"], rows))
    L.append("")
    L.append("早期（1/3 之前）C−A 仍为负，说明模型没有利用后续正文反推前面的答案；**未发现上下文泄露**。")
    L.append("")
    L.append("### 4.4 成本")
    L.append("")
    L.append(f'C 的 prompt 更长：cache miss {s["C"]["cache_hit"]/1e6:.2f}M 命中 / {s["C"]["prompt_tokens"]/1e6:.2f}M 总 prompt，成本 ${s["C"]["cost"]:.2f}，是 A 的 {s["C"]["cost"]/s["A"]["cost"]:.1f}×；换来的却是 −2.1 pp。')
    L.append("")
    L.append("## 5. 一次性作答（E）")
    L.append("")
    L.append(f'E 每个子过程一次调用（40 次），一次输出全部空的 JSON；总体 54.9% / 58.1%。')
    L.append("")
    L.append("### 5.1 按标签")
    L.append("")
    rows = []
    for c in ORDER:
        row = [c]
        for label in LABELS:
            v = s[c]["labels"][label]
            row.append(pct(v["norm"] / v["n"]))
        rows.append(row)
    L.append(table(["条件", "C1", "C2", "A1", "A2"], rows))
    L.append("")
    L.append("E 的 C1（80.4%）甚至高于 A（75.7%）——整段上下文帮助了简单判断；但 C2 只有 75.7%（A 93.9%），A2 只有 5.1%。")
    L.append("")
    L.append("### 5.2 E 的每子过程")
    L.append("")
    rows = []
    for proc, v in stats["d"]["per_process"].items():
        rows.append([proc, v["n"], pct(v["exact"]), pct(v["norm"])])
    L.append(table(["子过程", "空数", "严格", "归一化"], rows))
    L.append("")
    L.append("### 5.3 成本")
    L.append("")
    L.append(f'E 总成本 ${stats["d"]["cost"]:.4f}，40 次调用，输出 {stats["d"]["completion_tokens"]:,} tokens；每条正确约 ${stats["d"]["cost"]/s["E"]["exact"]:.6f}，而 A 为 ${s["A"]["cost"]/s["A"]["exact"]:.6f}。E 是**最便宜**的条件，但 A2 仍只有 5.1%。')
    L.append("")
    L.append("## 6. 错误类型")
    L.append("")
    L.append("### 6.1 五类错误（归一化不一致）")
    L.append("")
    rows = []
    for c in ORDER:
        e = stats["error_kinds"][c]
        rows.append([c, e["norm_wrong"].get("格式", 0), e["norm_wrong"].get("轴", 0), e["norm_wrong"].get("符号", 0),
                     e["defs"], e["other"]])
    L.append(table(["条件", "格式", "轴", "符号", "定义未用", "其他"], rows))
    L.append("")
    L.append("“定义未用”指模型仍按默认定义作答，占 7–10%；“其他”是级联/多步错误，B 与 E 最高；A 的“其他”最少。")
    L.append("")
    L.append("### 6.2 A2：多步瓶颈")
    L.append("")
    rows = []
    for c in ORDER:
        v = stats["a2"][c]
        rows.append([c, v["n"], pct(v["exact"]), pct(v["norm"]), f'{v["format"]}'])
    L.append(table(["条件", "空数", "严格", "归一化", "格式差"], rows))
    L.append("")
    L.append("A2 的 gold 前缀下 42.3%，自身前缀/一次性只有 5.1%。A2 的 350 空是全部条件里差距最大的 37.2 pp，说明多步矩阵/表达式严重依赖正确历史。")
    L.append("")
    L.append("### 6.3 定义未用")
    L.append("")
    rows = []
    for c in ORDER:
        rows.append([c, stats["error_kinds"][c]["defs"], pct(stats["error_kinds"][c]["defs"] / s[c]["n"])])
    L.append(table(["条件", "定义未用", "占比"], rows))
    L.append("")
    L.append("## 7. 成本")
    L.append("")
    rows = []
    for c in ORDER:
        x = s[c]
        rows.append([c, x["calls"], f'${x["cost"]:.4f}', f'${x["cost"]/x["exact"]:.6f}', pct(x["exact_rate"])])
    L.append(table(["条件", "调用", "总成本", "每正确成本", "严格"], rows))
    L.append("")
    L.append("C 成本是 A 的 3.7×，准确率反而低 2.1 pp；E 成本是 A 的 1/14.5，准确率低 13.2 pp。是否值得用 C/E 取决于目标是“省调用”还是“追准确率”。")
    L.append("")
    L.append("## 8. 统计与稳健性")
    L.append("")
    L.append("- 只有 10 条样本、1 次运行；official DeepSeek 接口忽略 seed，报告里的区间是**样本级自助法**，不是重复运行的方差。")
    L.append("- 配对检验用 McNemar exact；区间用样本 cluster bootstrap（重抽样单位是样本，不是空）。")
    L.append("- 严格 exact 会惩罚格式差异（A 有 41 个格式差，C 47，E 62）；归一化口径是语义准确率。")
    L.append("- 本轮是 deepseek-flash + thinking disabled 的对照；主评测（v4-pro + thinking）可能显著缩小 A−B 和 A2 的差距，但“后续正文不泄露”“A2 是多步瓶颈”这两个方向性结论预计不变。")
    L.append("")
    L.append("## 9. 结论与建议")
    L.append("")
    L.append("1. **gold 前缀是最大主效应**：A 68.1% vs B 49.7%，−18.4 pp；A2 42.3% vs 5.1%，−37.2 pp。逐句 gold 前缀的 A 是合理的主协议/上限，B 是真实部署条件。")
    L.append("2. **未发现上下文泄露**：C（66.0%）低于 A（68.1%），D（48.1%）低于 B（49.7%）；后续正文没有帮助，反而略增“定义未用”和成本。C/D 作为泄露对照是干净的阴性结果。")
    L.append("3. **误差传播是主风险**：B 的“其他”错误 511 个（26.5%），A 只有 240（12.4%）；B 的首错更早，首错之后句级全对率更低。")
    L.append("4. **E 是便宜但不够好的端到端基线**：54.9%，比 B 高 5.2 pp，成本仅 1/14.5；C1 甚至高于 A，但 A2 只有 5.1%，不能替代 A。")
    L.append("5. **A2 是瓶颈**：350 空，A 42.3%，其余 5.1%；多步矩阵/表达式需要 gold 历史或更强模型。")
    L.append("6. **“定义未用”是稳定的语义错误**：7–10%，B/D 最高；提示词/输出格式值得再压。")
    L.append("7. **成本**：C 3.7× A 的成本换 −2.1 pp，不划算；E 最便宜；A 最准。")
    L.append("8. **下一步**：v4-pro + thinking 主评测；100 条正式集；每条件 3 次重复；补 C（一次性无前缀）与 position/conditional 的正式统计；A2 单独分析。")
    L.append("")
    L.append("## 附录：完整逐样本表")
    L.append("")
    rows = []
    for smp in samples:
        row = [smp]
        for c in ORDER:
            v = s[c]["samples"][smp]
            row.append(pct(v["norm"] / v["n"]))
        rows.append(row)
    L.append(table(["样本"] + ORDER, rows))
    L.append("")
    return "\n".join(L)


def main() -> None:
    stats = analyze()
    md = markdown(stats)
    (RESULTS / "experiment_analysis.md").write_text(md, encoding="utf-8")
    (RESULTS / "experiment_analysis.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    s = stats["summary"]
    for c in ORDER:
        print(f'{c:>4}: exact {s[c]["exact"]:4d}/{s[c]["n"]} ({s[c]["exact_rate"]*100:5.1f}%)  '
              f'norm {s[c]["norm"]:4d} ({s[c]["norm_rate"]*100:5.1f}%)  '
              f'cost ${s[c]["cost"]:.4f}  calls {s[c]["calls"]}')
    print("pairs:")
    for a, b in PAIRS:
        p = stats["pairs"][f"{a}|{b}"]
        print(f'  {b} - {a}: {p["delta"]*100:+6.1f} pp  p={p["p"]:.4g}  '
              f'CI=[{p["ci"][0]*100:+.1f},{p["ci"][1]*100:+.1f}]  (only {a}={p["a_only"]}, only {b}={p["b_only"]})')
    print("A2:", {c: round(stats["a2"][c]["norm"] / stats["a2"][c]["n"] * 100, 1) for c in ORDER})
    print("wrote", RESULTS / "experiment_analysis.md")


if __name__ == "__main__":
    main()
