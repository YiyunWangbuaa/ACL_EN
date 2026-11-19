# -*- coding: utf-8 -*-
"""Per-condition output-cost share, and a 160-sample projection."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stats = json.loads((ROOT / "results" / "panel_stats.json").read_text(encoding="utf-8"))
PRICES = {
    "DeepSeek V4 Flash": (0.15, 0.60),
    "Qwen3.8 Flash": (0.15, 0.47),
    "Qwen3.8 27B": (0.15, 1.875),
    "Kimi K3": (0.192, 15.0),
}
MODELS = list(PRICES)
CONDS = ["A", "B", "C", "D", "E"]


def cost(P, C, pin, pout):
    return P / 1e6 * pin + C / 1e6 * pout


print("per-condition output-cost share w_out (and $ per 1,000 blanks)")
for m in MODELS:
    pin, pout = PRICES[m]
    parts = []
    for c in CONDS:
        meta = stats["meta"][m][c]
        P, C = meta["prompt_tokens"], meta["completion_tokens"]
        base = cost(P, C, pin, pout)
        w = C / 1e6 * pout / base
        parts.append(f"{c}:{w * 100:4.1f}%")
    print(f"  {m:20s} " + "  ".join(parts))

print("\nunit cost per 1,000 blanks (5-condition panel = 9,650 blanks)")
for m in MODELS:
    pin, pout = PRICES[m]
    meta = stats["meta"][m]
    base = sum(cost(v["prompt_tokens"], v["completion_tokens"], pin, pout) for v in meta.values())
    print(f"  {m:20s} ${base:6.2f} per 9,650 blanks = ${base / 9.65:5.2f}/1k blanks")

print("\n160-sample projection (16x the 10-sample panel, 154,400 blanks)")
for m in MODELS:
    pin, pout = PRICES[m]
    meta = stats["meta"][m]
    P = sum(v["prompt_tokens"] for v in meta.values())
    C = sum(v["completion_tokens"] for v in meta.values())
    base = cost(P, C, pin, pout) * 16
    row = [f"{base:7.1f}"]
    for rho in (5, 20, 100):
        row.append(f"{cost(P, C * rho, pin, pout) * 16:7.1f}")
    print(f"  {m:20s} base={row[0]}  rho=5:{row[1]}  rho=20:{row[2]}  rho=100:{row[3]}")
print("\nnot in these totals: timeout and 429 retries. 40 workers on long outputs add calls.")
print("reasoning_content is not written back; that would multiply input and dwarf w_out.")
