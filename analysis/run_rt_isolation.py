# -*- coding: utf-8 -*-
"""Build the three R_t conditions and refuse the run when a probe answer leaks."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from cloze_eval import _repair_phrase, contexts_for, load_samples  # noqa: E402
from domains.wind_tunnel.dataset_generate import PROCESS_TEXT  # noqa: E402
from domains.wind_tunnel.rt_isolation import (  # noqa: E402
    COMPONENT_IDS,
    DEEP_ANCHOR,
    MATRICES,
    PLACEHOLDER,
    PROBE_MARK,
    PROCESS,
    RT_ID,
    STAGES,
    build_conditions,
    isolate_stage,
)

DATA = EVAL_ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = EVAL_ROOT / "results" / "rt_isolation" / "audit.md"
SAMPLE = "body-unmarked"


def _chain(process: str) -> dict:
    row = load_samples(DATA, {SAMPLE})[0]
    chain = next(item for item in contexts_for(row) if item["process"] == process)
    if not chain.get("process_text"):
        chain["process_text"] = PROCESS_TEXT[process]
    chain["declaration"] = row["declaration"]
    return chain


def _frozen(chain: dict) -> dict[str, str]:
    memory = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            memory[str(blank["blank_id"])] = _repair_phrase(blank)
    return memory


def _report(prompts: dict[str, str], stage_counts: list[tuple[str, int]]) -> str:
    lines = [
        "# 分矩阵探针隔离",
        "",
        "定义稿是体轴左手系·无标记。这里检查的是提示隔离，没有调用模型。",
        "每张矩阵先写出名称，浅层探针在名称之后、原题之前。深层探针在符号选定之后。",
        "探针一律写出该矩阵的展开式，并附上矩阵格式。探针答案不写回。",
        "",
        "| 过程 | 矩阵 | 名称出现在 | 浅层锚点 | 原题 | 深层锚点 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in MATRICES:
        lines.append(
            f"| {PROCESS} | {item['name']} | {item['named']} | ID{item['shallow_anchor']} | ID{item['blank']} | ID{DEEP_ANCHOR} |"
        )
    for stage in STAGES:
        for item in stage["matrices"]:
            lines.append(
                f"| {stage['process']} | {item['name']} | {item['named']} | ID{item['shallow_anchor']} | ID{item['blank']} | ID{stage['deep_anchor']} |"
            )
    lines += [
        "",
        f"风轴系到体轴系隔离通过，共 {len(prompts)} 份提示。被遮蔽的原答案是 ID{'、ID'.join(COMPONENT_IDS)}。",
        f"占位符是“{PLACEHOLDER}”。总矩阵是 ID{RT_ID}。探针答案带“{PROBE_MARK}”。",
        "写回控制只把 R_γ_M 的深层探针答案追加到原始前缀上，其余探针答案仍然不进入。",
    ]
    for process, count in stage_counts:
        lines.append(f"{process} 隔离通过，共 {count} 份提示。")
    lines.append("天平系到体轴系的总矩阵是 ID104。天平系到风轴系没有单独的总矩阵空。力矩参考点没有分矩阵探针。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    chain = _chain(PROCESS)
    frozen = _frozen(chain)
    held = {
        item["key"]: {
            "shallow": f"{PROBE_MARK}浅层{item['name']}",
            "deep": f"{PROBE_MARK}深层{item['name']}",
        }
        for item in MATRICES
    }
    prompts = build_conditions(chain, frozen, held)
    stage_counts = []
    for stage in STAGES:
        later = _chain(stage["process"])
        stage_prompts = isolate_stage(later, _frozen(later), stage)
        stage_counts.append((stage["process"], len(stage_prompts)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(_report(prompts, stage_counts), encoding="utf-8")
    print(f"isolated {len(prompts)} prompts")
    for process, count in stage_counts:
        print(f"isolated {process} {count} prompts")
    print(OUT)


if __name__ == "__main__":
    main()
