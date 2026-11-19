# -*- coding: utf-8 -*-
"""Per-label accuracy for every model and condition.

panel_stats.json only stores the label breakdown for condition A, so the
label x condition grid is re-derived here from the 20 run directories.

Run:
    python analysis/label_by_condition.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MODELS = {
    "Kimi K3": {"A": "run_A_kimi_k3", "B": "run_B_kimi_k3", "C": "run_C_kimi_k3",
                "D": "run_D_kimi_k3", "E": "run_E_kimi_k3"},
    "Qwen3.8 Flash": {"A": "run_A_qwen_flash", "B": "run_B_qwen_flash",
                      "C": "run_C_qwen_flash", "D": "run_D_qwen_flash",
                      "E": "run_E_qwen_flash"},
    "Qwen3.8 27B": {"A": "run_A_qwen_27b", "B": "run_B_qwen_27b",
                    "C": "run_C_qwen_27b", "D": "run_D_qwen_27b",
                    "E": "run_E_qwen_27b"},
    "DeepSeek V4 Flash": {"A": "run_A", "B": "run_B", "C": "run_C",
                          "D": "run_D", "E": "run_E"},
}
CONDS = ["A", "B", "C", "D", "E"]
LABELS = ["C1", "C2", "A1", "A2"]


def collect():
    grid = defaultdict(lambda: {"n": 0, "exact": 0, "norm": 0})
    for model, conds in MODELS.items():
        for cond, dirname in conds.items():
            path = RES / dirname / "records.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    for b in rec.get("blanks", []):
                        key = (model, cond, b.get("label"))
                        cell = grid[key]
                        cell["n"] += 1
                        cell["exact"] += int(bool(b.get("exact")))
                        cell["norm"] += int(bool(b.get("normalized")))
    out = {"labels": LABELS, "conditions": CONDS, "models": {}, "meta": {}}
    for model in MODELS:
        out["models"][model] = {}
        for cond in CONDS:
            out["models"][model][cond] = {}
            for label in LABELS:
                cell = grid[(model, cond, label)]
                out["models"][model][cond][label] = {
                    "n": cell["n"],
                    "exact": cell["exact"],
                    "norm": cell["norm"],
                    "exact_rate": cell["exact"] / cell["n"] if cell["n"] else None,
                    "norm_rate": cell["norm"] / cell["n"] if cell["n"] else None,
                }
    out["meta"] = {
        "blanks_per_label": {label: grid[("Kimi K3", "A", label)]["n"] for label in LABELS},
        "total_blanks": sum(v["n"] for v in grid.values()),
        "source": "results/run_*/records.jsonl",
    }
    (RES / "label_by_condition.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    for label in LABELS:
        print(f"\n== {label} (n={out['meta']['blanks_per_label'][label]} blanks/condition) ==")
        print(f"{'model':20s} " + " ".join(f"{c:>7s}" for c in CONDS))
        for model in MODELS:
            cells = [out["models"][model][c][label] for c in CONDS]
            print(f"{model:20s} " + " ".join(
                "   —   " if cell["n"] == 0 else f"{cell['norm_rate'] * 100:6.1f}%" for cell in cells))
    return out


if __name__ == "__main__":
    collect()
