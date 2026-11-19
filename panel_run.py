# -*- coding: utf-8 -*-
"""按模型顺序跑 A、B、C、D。额度或限流中断时停在当前条件，不开始下一个。

已完成的目录会跳过。中断后重新执行本脚本，会从断点那一句继续。
E 组（端到端）已经单独跑完，这里不重复调用。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DOC = RESULTS / "qwen_kimi_compare.md"

MODELS = (
    ("qwen3.8-flash", "qwen_flash", "千问 Flash"),
    ("qwen3.8-27b", "qwen_27b", "千问 27B"),
    ("kimi-k3", "kimi_k3", "Kimi K3"),
)
CONDITIONS = (
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("D", "D"),
)
DONE_E = {
    "qwen_flash": RESULTS / "run_E_qwen_flash",
    "qwen_27b": RESULTS / "run_E_qwen_27b",
    "kimi_k3": RESULTS / "run_E_kimi_k3",
}
EXPECTED = 1550


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _is_complete(out: Path) -> bool:
    checkpoint = _read_json(out / "checkpoint.json")
    meta = _read_json(out / "meta.json")
    if checkpoint.get("status") == "complete" and checkpoint.get("completed_sentences") == EXPECTED:
        return True
    return meta.get("status") == "complete" and meta.get("sentences") == EXPECTED


def _cell(out: Path) -> str:
    checkpoint = _read_json(out / "checkpoint.json")
    meta = _read_json(out / "meta.json")
    source = meta if meta.get("exact") is not None else checkpoint
    exact = source.get("exact")
    normalized = source.get("normalized")
    blanks = source.get("blanks") or source.get("completed_blanks") or 0
    if exact is None or not blanks:
        if checkpoint.get("status") == "stopped":
            done = checkpoint.get("completed_sentences") or 0
            return f"中断 {done}/{EXPECTED} 句"
        return "未跑"
    status = checkpoint.get("status") or meta.get("status") or ""
    text = f"{exact}/{blanks}（{exact / blanks:.1%}）"
    if status == "stopped":
        text += "，未完成"
    return text


def _write_doc(note: str) -> None:
    lines = [
        "# 千问与 Kimi 的 A / B / C / D / E 对照",
        "",
        "思考关闭，温度 0，每个条件 1 次。A、B、C、D 每句一次调用，最多 512 token；E 每个子过程一次调用，最多 8192 token。",
        "样本是 pilot_10，共 10 条、1930 个空。空号按四个子过程连续编号，从 1 到 193。",
        "DeepSeek 已完成的 A–D 使用的是每个子过程从 1 重新编号的旧提示，分数可以对照方向，不能当作同一提示的重复。",
        "",
        "## 严格正确",
        "",
        "| 模型 | A | B | C | D | E |",
        "|---|---|---|---|---|---|",
    ]
    for _model, slug, label in MODELS:
        cells = []
        for flag, _name in CONDITIONS:
            cells.append(_cell(RESULTS / f"run_{flag}_{slug}"))
        cells.append(_cell(DONE_E[slug]))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "E 列来自已经跑完的 `run_E_qwen_flash`、`run_E_qwen_27b`、`run_E_kimi_k3`。",
        "",
        "## 条件含义",
        "",
        "- A：空前填标准答案，当前句结束就截断。它是这一套协议里的参照。",
        "- B：空前填模型自己的上一句原文。前一句错了，后面会接着错。",
        "- C：空前仍是标准答案，但后面的正文还在。用来看后文会不会把答案漏出来。",
        "- D：空前是模型自己的原文，后文也保留。",
        "- E：一个子过程只调用一次，一次写出该过程的全部空。",
        "",
        "DeepSeek Flash 在旧编号上的严格正确率是 A 68.1%、B 49.7%、C 66.0%、D 48.1%、E 54.9%。",
        "那一轮的主要差别是 A 到 B：没有标准答案垫在前面时，推导题 A2 掉得最多。",
        "",
        "## 断点",
        "",
    ])
    panel = _read_json(RESULTS / "panel_checkpoint.json")
    latest = panel.get("latest") if isinstance(panel.get("latest"), dict) else {}
    if latest.get("status") == "stopped":
        ids = "、".join(str(item) for item in (latest.get("blank_ids") or []))
        lines.extend([
            f"最近一次停在 **{latest.get('model')}** 的条件 **{latest.get('condition')}**。",
            f"位置：样本 {latest.get('sample_id')}，过程 {latest.get('process')}，空 {ids}。",
            f"原因：{latest.get('kind')}。已完成 {latest.get('completed_sentences')}/{latest.get('expected_sentences')} 句。",
            "这一句没有写入记录。重新执行 `python panel_run.py` 会从这里继续，不会把失败算进分数。",
            "",
            "```",
            str(latest.get("resume_command") or ""),
            "```",
            "",
            f"接口返回：{latest.get('error')}",
            "",
        ])
    elif note:
        lines.extend([note, ""])
    else:
        lines.append("当前没有未完成的断点。")
        lines.append("")
    lines.extend([
        "## 结果怎么读",
        "",
        "同一模型里，A 高于 B，说明错误会沿子过程传下去。C 接近 A，说明看见后文并没有把空填对。D 接近 B，说明后文也补不回已经走错的前缀。",
        "E 通常落在 A 和 B 之间：一次生成全部空，没有逐句的标准答案，也没有把上一句的错误再送回去。",
        "模型之间比的是同一条件下的严格正确率。Kimi K3 的 E 组已经高于两个千问；A–D 跑完后才能判断这是不是端到端才有的差别。",
        "",
    ])
    DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _load_env()
    os.environ["DEEPSEEK_BASE_URL"] = os.environ.get(
        "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    os.environ["PYTHONIOENCODING"] = "utf-8"
    _write_doc("队列已开始。完成一个条件就会更新这张表。")
    for model, slug, label in MODELS:
        for flag, name in CONDITIONS:
            out = RESULTS / f"run_{flag}_{slug}"
            if _is_complete(out):
                print(f"跳过已完成：{label} {name} -> {out}", flush=True)
                continue
            print(f"开始：{label} {name} -> {out}", flush=True)
            completed = subprocess.run(
                [
                    sys.executable, "cloze_eval.py",
                    "--model", model,
                    "--thinking", "disabled",
                    "--condition", flag,
                    "--runs", "1",
                    "--workers", "4",
                    "--out", str(out),
                ],
                cwd=ROOT,
                env=os.environ.copy(),
            )
            _write_doc("")
            if completed.returncode != 0:
                print(f"停在 {label} {name}，退出码 {completed.returncode}。后续条件未开始。", flush=True)
                raise SystemExit(completed.returncode)
            print(f"完成：{label} {name}", flush=True)
    _write_doc("A、B、C、D 都已跑完。")
    print("全部条件完成。", flush=True)


if __name__ == "__main__":
    main()
