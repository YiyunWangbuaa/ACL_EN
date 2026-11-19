# -*- coding: utf-8 -*-
"""Print every number quoted in the report narrative, so text claims can be checked."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "results" / "panel_stats.json").read_text(encoding="utf-8"))
MODELS = ["Kimi K3", "Qwen3.8 Flash", "Qwen3.8 27B", "DeepSeek V4 Flash"]
CONDS = ["A", "B", "C", "D", "E"]


def r(m, c, k="norm_rate"):
    return d["models"][m][c][k]


print("== norm rates ==")
for m in MODELS:
    print(f"{m:20s}", " ".join(f"{c}:{r(m, c) * 100:5.1f}" for c in CONDS))

print("\n== gap to 2nd place (norm) ==")
for c in CONDS:
    vals = sorted(((r(m, c), m) for m in MODELS), reverse=True)
    print(f"{c:3s} top={vals[0][1]}:{vals[0][0] * 100:.1f} 2nd={vals[1][1]}:{vals[1][0] * 100:.1f} "
          f"gap={100 * (vals[0][0] - vals[1][0]):.1f}")

print("\n== A2 / C1 / C2 label rates ==")
for m in MODELS:
    lab = d["models"][m]["A"]["labels"]
    print(f"{m:20s}", " ".join(f"{k}:{100 * lab[k]['norm'] / lab[k]['n']:5.1f}" for k in ("C1", "C2", "A1", "A2")))

print("\n== process rates (A) ==")
for m in MODELS:
    pr = d["models"][m]["A"]["processes"]
    print(f"{m:20s}", " ".join(f"{k[:6]}:{100 * v['norm'] / v['n']:5.1f}" for k, v in pr.items()))

print("\n== tier rates (A) ==")
for m in MODELS:
    t = d["models"][m]["A"]["tiers"]
    print(f"{m:20s}", " ".join(f"{k}:{100 * v['norm'] / v['n']:5.1f}" for k, v in t.items()))

print("\n== defs / format ==")
for m in MODELS:
    a = d["models"][m]["A"]
    print(f"{m:20s} defs={a['defs']} ({100 * a['defs'] / a['n']:.1f}%) other={a['other']} format={a['format']}",
          "kinds=", a["error_kinds"])

print("\n== A|B flips ==")
for m in MODELS:
    p = d["models"][m]["pairs"]["A|B"]
    print(f"{m:20s} a_only={p['a_only']:4d} b_only={p['b_only']:4d} delta={100 * p['delta']:+.1f} "
          f"ci=({100 * p['ci'][0]:+.1f},{100 * p['ci'][1]:+.1f}) p={p['p']:.3g}")

print("\n== format loss (norm-exact) ==")
for m in MODELS:
    print(f"{m:20s}", " ".join(f"{c}:{100 * (r(m, c) - r(m, c, 'exact_rate')):.1f}" for c in CONDS))

print("\n== pairs ==")
for m in MODELS:
    print(f"{m:20s}", " ".join(f"{k}:{100 * v['delta']:+.1f}" for k, v in d["models"][m]["pairs"].items()))

print("\n== cost / tokens ==")
for m in MODELS:
    meta = d["meta"][m]
    pt = sum(v["prompt_tokens"] for v in meta.values())
    ct = sum(v["completion_tokens"] for v in meta.values())
    cost = sum(v["cost"] for v in meta.values())
    print(f"{m:20s} ptok={pt:,} ctok={ct:,} cost=${cost:.2f} calls={sum(v['calls'] for v in meta.values())}")
ds = d["meta"]["DeepSeek V4 Flash"]
print("C/A prompt ratio", ds["C"]["prompt_tokens"] / ds["A"]["prompt_tokens"],
      "cost ratio", ds["C"]["cost"] / ds["A"]["cost"],
      "D/A prompt", ds["D"]["prompt_tokens"] / ds["A"]["prompt_tokens"])
print("total prompt", sum(sum(v["prompt_tokens"] for v in d["meta"][m].values()) for m in MODELS))
