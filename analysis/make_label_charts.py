# -*- coding: utf-8 -*-
"""Four grouped bar charts: label accuracy x condition x model.

Input : results/label_by_condition.json
Output: report_build/figs/label_C1.png ... label_A2.png

Run:
    python analysis/make_label_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "label_by_condition.json").read_text(encoding="utf-8"))
OUT = ROOT / "report_build" / "figs"
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ["A", "B", "C", "D", "E"]
MODELS = ["Kimi K3", "Qwen3.8 Flash", "Qwen3.8 27B", "DeepSeek V4 Flash"]
COLORS = ["#1F3A5F", "#2E74B5", "#6FA8DC", "#B0302E"]
LABEL_TITLE = {
    "C1": "C1 轴/符号选择（530 空/条件）",
    "C2": "C2 方向或正负判断（230 空/条件）",
    "A1": "A1 短答：表达式/矩阵（820 空/条件）",
    "A2": "A2 长答构造：矩阵/公式（350 空/条件）",
}

# register a CJK font so labels render instead of boxes
for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        try:
            font_manager.fontManager.addfont(candidate)
            family = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.sans-serif"] = [family]
            break
        except Exception:
            continue
plt.rcParams["axes.unicode_minus"] = False


def draw(label: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 3.15), dpi=220)
    n_models = len(MODELS)
    width = 0.19
    xs = range(len(CONDS))
    for i, model in enumerate(MODELS):
        vals, ns = [], []
        for cond in CONDS:
            cell = DATA["models"][model][cond][label]
            vals.append(cell["norm_rate"] * 100)
            ns.append(cell["n"])
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar([x + offset for x in xs], vals, width=width * 0.92,
                      color=COLORS[i], label=model, zorder=3,
                      edgecolor="white", linewidth=0.4)
        for bar, value in zip(bars, vals):
            cx = bar.get_x() + bar.get_width() / 2
            if value >= 30:
                # keep the number inside the bar so neighbouring bars stay legible
                ax.text(cx, value - 1.6, f"{value:.1f}", ha="center", va="top",
                        fontsize=4.6, color="white", rotation=90, zorder=4)
            else:
                ax.text(cx, value + 1.4, f"{value:.1f}", ha="center", va="bottom",
                        fontsize=4.6, color="#333333", rotation=90, zorder=4)

    ax.set_xticks(list(xs))
    ax.set_xticklabels(CONDS, fontsize=9)
    ax.set_xlabel("条件", fontsize=9)
    ax.set_ylabel("归一化正确率（%）", fontsize=9)
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, color="#C9D3DE", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_title(LABEL_TITLE[label], fontsize=9.5, color="#1F3A5F", pad=20, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False,
              fontsize=7.4, handlelength=1.2, columnspacing=1.4)
    fig.tight_layout()
    path = OUT / f"label_{label}.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for label in ("C1", "C2", "A1", "A2"):
        p = draw(label)
        print("wrote", p)
