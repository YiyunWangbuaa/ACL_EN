# -*- coding: utf-8 -*-
"""Estimate the cost of enabling chain-of-thought for the 5-condition panel.

Thinking only adds reasoning tokens, which providers bill as output tokens, so
    cost(rho) = P*p_in + rho*C*p_out
with P/C the measured prompt/completion tokens and rho the output inflation.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
stats = json.loads((ROOT / "results" / "panel_stats.json").read_text(encoding="utf-8"))

# USD per 1M tokens; Kimi K3 has no DashScope list price in the registry, use the
# OpenRouter list price recorded earlier (0.192 / 15).
PRICES = {
    "DeepSeek V4 Flash": (0.15, 0.60, "off-peak 0.15/0.60"),
    "Qwen3.8 Flash": (0.15, 0.47, "0.15/0.47"),
    "Qwen3.8 27B": (0.15, 1.875, "0.15/1.875"),
    "Kimi K3": (0.192, 15.0, "OpenRouter 0.192/15"),
}
MODELS = ["DeepSeek V4 Flash", "Qwen3.8 Flash", "Qwen3.8 27B", "Kimi K3"]
RHOS = [5, 10, 20, 50, 100]


def cost(P, C, pin, pout):
    return P / 1e6 * pin + C / 1e6 * pout


print(f"{'model':20s} {'calls':>6s} {'P':>12s} {'C':>9s} {'P/call':>7s} {'C/call':>7s} "
      f"{'base$':>7s} {'w_out':>6s} " + " ".join(f"x{r:<4d}" for r in RHOS))
rows = []
for m in MODELS:
    meta = stats["meta"][m]
    calls = sum(v["calls"] for v in meta.values())
    P = sum(v["prompt_tokens"] for v in meta.values())
    C = sum(v["completion_tokens"] for v in meta.values())
    pin, pout, note = PRICES[m]
    base = cost(P, C, pin, pout)
    w = (C / 1e6 * pout) / base
    out = [cost(P, C * r, pin, pout) for r in RHOS]
    print(f"{m:20s} {calls:6d} {P:12,d} {C:9,d} {P / calls:7.0f} {C / calls:7.1f} "
          f"{base:7.2f} {w:5.1%} " + " ".join(f"{c:6.1f}" for c in out))
    rows.append((m, calls, P, C, base, w, out))

print("\nrelative increase vs base:")
for m, calls, P, C, base, w, out in rows:
    print(f"{m:20s} " + " ".join(f"+{(c / base - 1) * 100:5.0f}%" for c in out))

print("\nabsolute delta (USD):")
for m, calls, P, C, base, w, out in rows:
    print(f"{m:20s} " + " ".join(f"+{c - base:5.1f}" for c in out))

# empirical anchor: DeepSeek V4 Pro pilot with thinking on
pro_entry = stats["meta"].get("DeepSeek V4 Pro (A, pilot)", {})
pro = pro_entry.get("A*")
ds = stats["meta"]["DeepSeek V4 Flash"]["A"]
if pro:
    print(f"\nPro thinking pilot: calls={pro['calls']} P/call={pro['prompt_tokens'] / pro['calls']:.0f} "
          f"C/call={pro['completion_tokens'] / pro['calls']:.0f} cost=${pro['cost']:.2f} "
          f"blanks=134 cost/blank=${pro['cost'] / 134:.4f}")
    print(f"Flash no-thinking A: C/call={ds['completion_tokens'] / ds['calls']:.1f} "
          f"cost/blank=${ds['cost'] / 1930:.5f}")
    print(f"ratio of output per call: "
          f"{pro['completion_tokens'] / pro['calls'] / (ds['completion_tokens'] / ds['calls']):.0f}x")
    print(f"ratio of cost per blank: "
          f"{(pro['cost'] / 134) / (ds['cost'] / 1930):.0f}x")
else:
    print("\nPro thinking pilot is no longer under results/, so the 328x anchor is skipped.")
print("\nnot in these totals:")
print("- retries after timeouts and HTTP 429. Long outputs at 40 workers add billed calls.")
print("- reasoning_content is not fed back. Putting it in the next prompt would multiply input tokens.")
