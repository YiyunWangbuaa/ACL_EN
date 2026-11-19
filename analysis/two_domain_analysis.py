# -*- coding: utf-8 -*-
"""Cross-domain analysis of the five injection conditions.

Domains: wind tunnel (196 blanks/sample) and flight mechanics (104 blanks/sample).
Model: deepseek-flash. Thinking on for both; a thinking-off flight run is included
as a secondary control.

Conditions: A gold-trunc, B self-trunc, C gold-keep, D self-keep, E proc-e2e.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "two_domain_analysis.json"

RUNS = {
    "wind_thinking": {
        "A": "results/wind_flash_think/pilot_gold-trunc",
        "B": "results/wind_flash_think/pilot_self-trunc",
        "C": "results/wind_flash_think/pilot_gold-keep",
        "D": "results/wind_flash_think/pilot_self-keep",
        "E": "results/wind_flash_think/pilot_proc-e2e",
    },
    "flight_thinking": {
        "A": "results/flight_flash_think/pilot_gold-trunc",
        "B": "results/flight_flash_think/pilot_self-trunc",
        "C": "results/flight_flash_think/pilot_gold-keep",
        "D": "results/flight_flash_think/pilot_self-keep",
        "E": "results/flight_flash_think/pilot_proc-e2e",
    },
    "flight_nothink": {
        "A": "results/flight_flash/run_A",
        "B": "results/flight_flash/run_B",
        "C": "results/flight_flash/run_C",
        "D": "results/flight_flash/run_D",
        "E": "results/flight_flash/run_E",
    },
}
LABELS = ("C1", "C2", "A1", "A2")
PAIRS = (("A", "B"), ("A", "C"), ("C", "D"), ("B", "D"), ("A", "E"), ("D", "E"))


def load(path: Path) -> dict:
    """Return {key: dict} for every blank in a run."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    blanks = {}
    meta = {"blanks": 0, "reasoning_tokens": 0, "completion_tokens": 0,
            "prompt_tokens": 0, "calls": 0, "latency": 0, "truncated": 0}
    for row in rows:
        usage = row.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        meta["calls"] += 1
        meta["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        meta["completion_tokens"] += int(usage.get("completion_tokens") or 0)
        meta["reasoning_tokens"] += int(details.get("reasoning_tokens") or 0)
        meta["latency"] += int(row.get("latency_ms") or 0)
        meta["truncated"] += 1 if row.get("truncated") else 0
        for blank in row.get("blanks") or []:
            key = (row["sample_id"], row["process"], str(blank["blank_id"]))
            blanks[key] = {
                "tier": row.get("tier"),
                "process": row["process"],
                "label": blank.get("label"),
                "exact": bool(blank.get("exact")),
                "norm": bool(blank.get("normalized", blank.get("exact"))),
            }
    meta["blanks"] = len(blanks)
    return {"blanks": blanks, "meta": meta}


def rate(blanks: dict, field: str = "exact") -> float:
    if not blanks:
        return 0.0
    return sum(1 for item in blanks.values() if item[field]) / len(blanks)


def rate_by(blanks: dict, key: str, field: str = "exact") -> dict:
    totals: dict = {}
    hits: dict = {}
    for item in blanks.values():
        value = item[key]
        totals[value] = totals.get(value, 0) + 1
        hits[value] = hits.get(value, 0) + (1 if item[field] else 0)
    return {value: hits[value] / totals[value] for value in sorted(totals)}


def count_by(blanks: dict, key: str) -> dict:
    counts: dict = {}
    for item in blanks.values():
        counts[item[key]] = counts.get(item[key], 0) + 1
    return dict(sorted(counts.items()))


def label_contrast(a: dict, b: dict) -> dict:
    out = {}
    for label in LABELS:
        keys = [k for k in set(a) & set(b) if a[k]["label"] == label]
        if not keys:
            continue
        out[label] = (sum(1 for k in keys if a[k]["exact"]) - sum(1 for k in keys if b[k]["exact"])) / len(keys)
    return out


def rate_group(blanks: dict, labels: tuple[str, ...], field: str = "exact") -> float:
    subset = [item for item in blanks.values() if item["label"] in labels]
    if not subset:
        return 0.0
    return sum(1 for item in subset if item[field]) / len(subset)


def mcnemar(a: dict, b: dict) -> dict:
    """Paired exact test on the shared keys."""
    keys = set(a) & set(b)
    only_a = sum(1 for key in keys if a[key]["exact"] and not b[key]["exact"])
    only_b = sum(1 for key in keys if b[key]["exact"] and not a[key]["exact"])
    n = only_a + only_b
    if n == 0:
        return {"delta": rate({k: a[k] for k in keys}) - rate({k: b[k] for k in keys}),
                "only_a": 0, "only_b": 0, "p": 1.0, "n": 0}
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / (2 ** n)
    delta = (sum(1 for k in keys if a[k]["exact"]) - sum(1 for k in keys if b[k]["exact"])) / len(keys)
    return {"delta": delta, "only_a": only_a, "only_b": only_b, "p": min(1.0, 2 * tail), "n": len(keys)}


def cluster_ci(a: dict, b: dict, reps: int = 3000, seed: int = 7) -> tuple[float, float]:
    samples: dict = {}
    for key in set(a) & set(b):
        samples.setdefault(key[0], []).append((a[key]["exact"], b[key]["exact"]))
    names = list(samples)
    rng = random.Random(seed)
    means = []
    for _ in range(reps):
        drawn = [names[rng.randrange(len(names))] for _ in names]
        pairs = [pair for name in drawn for pair in samples[name]]
        means.append(sum(int(x) - int(y) for x, y in pairs) / len(pairs))
    means.sort()
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def main() -> None:
    report = {}
    loaded = {}
    for domain, conditions in RUNS.items():
        loaded[domain] = {cond: load(ROOT / path / "records.jsonl") for cond, path in conditions.items()}
        report[domain] = {
            "conditions": {
                cond: {
                    "exact": rate(data["blanks"]),
                    "norm": rate(data["blanks"], "norm"),
                    "blanks": len(data["blanks"]),
                    "labels": rate_by(data["blanks"], "label"),
                    "label_counts": count_by(data["blanks"], "label"),
                    "tiers": rate_by(data["blanks"], "tier"),
                    "processes": rate_by(data["blanks"], "process"),
                    "definition_items": rate_group(data["blanks"], ("C1", "C2")),
                    "derivation_items": rate_group(data["blanks"], ("A1", "A2")),
                    "meta": data["meta"],
                }
                for cond, data in loaded[domain].items()
            },
            "pairs": {},
        }
        for left, right in PAIRS:
            test = mcnemar(loaded[domain][left]["blanks"], loaded[domain][right]["blanks"])
            low, high = cluster_ci(loaded[domain][left]["blanks"], loaded[domain][right]["blanks"])
            report[domain]["pairs"][f"{left}|{right}"] = {
                "delta": test["delta"], "p": test["p"],
                "ci": [low, high],
                "only_left": test["only_a"], "only_right": test["only_b"],
                "by_label": label_contrast(loaded[domain][left]["blanks"],
                                           loaded[domain][right]["blanks"]),
            }

    # difference in differences between domains for each contrast
    report["did"] = {}
    for left, right in PAIRS:
        wind = report["wind_thinking"]["pairs"][f"{left}|{right}"]["delta"]
        flight = report["flight_thinking"]["pairs"][f"{left}|{right}"]["delta"]
        report["did"][f"{left}|{right}"] = wind - flight

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for domain in RUNS:
        print(f"\n== {domain} ==")
        for cond, data in report[domain]["conditions"].items():
            print(f"  {cond}: exact {data['exact']:.1%}  norm {data['norm']:.1%}  "
                  f"def {data['definition_items']:.1%} der {data['derivation_items']:.1%}  "
                  f"labels " +
                  " ".join(f"{k}={v:.0%}" for k, v in data["labels"].items()))
            print("       processes " + " ".join(f"{k}={v:.0%}" for k, v in data["processes"].items())
                  + f"  | reasoning {data['meta']['reasoning_tokens']/max(data['meta']['calls'],1):.0f} tok/call"
                  f"  latency {data['meta']['latency']/max(data['meta']['calls'],1)/1000:.1f}s")
        for pair, data in report[domain]["pairs"].items():
            print(f"  {pair}: {data['delta']*100:+.1f} pp  p={data['p']:.2g}  "
                  f"CI[{data['ci'][0]*100:+.1f},{data['ci'][1]*100:+.1f}]")
    print("\n== difference in differences (wind - flight) ==")
    for pair, value in report["did"].items():
        print(f"  {pair}: {value*100:+.1f} pp")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
