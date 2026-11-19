# -*- coding: utf-8 -*-
"""Model-prefix shallow and rederive probes, built from a fresh self-trunc walk.

The walk starts at the first sentence of each probed process and stops once the
deep anchor has been answered. Probe calls use that memory. Their answers are
not written back. Deep-original is not run.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from cloze_eval import (  # noqa: E402
    SELF_TRUNC,
    THINKING_MAX_TOKENS,
    _remember,
    contexts_for,
    job_for,
    load_env_file,
    load_samples,
    score_one,
    standard_answers,
)
from cloze_score import judge  # noqa: E402
from domains.wind_tunnel.rt_isolation import (  # noqa: E402
    COMPONENT_IDS,
    DEEP_ANCHOR,
    MATRICES,
    PLACEHOLDER,
    PROCESS,
    STAGES,
    _copy,
    _probe_prompt,
    _question,
)
from semi_auto_eval import make_client  # noqa: E402

DATA = EVAL_ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = EVAL_ROOT / "results" / "rt_low_body_model"
SAMPLE = "body-unmarked"
MODEL = "deepseek-flash"
SEED = 20261102

TARGETS = (
    {
        "process": PROCESS,
        "deep_anchor": DEEP_ANCHOR,
        "hide": COMPONENT_IDS,
        "matrices": MATRICES,
    },
    *STAGES,
)


def _chain(row: dict, process: str) -> dict:
    chain = next(item for item in contexts_for(row) if item["process"] == process)
    chain["declaration"] = row["declaration"]
    return chain


def _extract(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    start = raw.find("[[")
    if start < 0:
        return raw
    depth = 0
    for index, char in enumerate(raw[start:], start):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return raw[start:index + 1]
    return raw[start:]


def _same(left: str, right: str) -> bool:
    return bool(left) and bool(right) and judge(left, right, "矩阵")["normalized"]


def _label(prediction: str, gold: str, standard: str) -> str:
    hit_gold = judge(prediction, gold, "矩阵")["normalized"]
    same = judge(gold, standard, "矩阵")["normalized"] if standard else False
    hit_standard = judge(prediction, standard, "矩阵")["normalized"] if standard else False
    if same:
        return "契约与标准相同" if hit_gold else "两者都不是"
    if hit_gold:
        return "契约"
    if hit_standard:
        return "标准"
    return "两者都不是"


def _gold(chain: dict, blank_id: str) -> str:
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            if str(blank["blank_id"]) == str(blank_id):
                return blank["answer"]
    raise KeyError(blank_id)


def _append(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _walk(client, chain: dict, anchor: str, standard: dict, flow_path: Path, lock: threading.Lock) -> dict[str, str]:
    memory: dict[str, str] = {}
    done = _done_ids(flow_path)
    reached = False
    for index, sentence in enumerate(chain["sentences"], start=1):
        ids = [str(item) for item in sentence["blank_ids"]]
        key = f"{chain['process']}:{','.join(ids)}"
        if key in done:
            saved = next(
                json.loads(line) for line in flow_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line)["id"] == key
            )
            for blank in saved["blanks"]:
                _remember(memory, blank, set())
        else:
            job = job_for(
                chain, sentence, SELF_TRUNC, memory, 0, SEED, THINKING_MAX_TOKENS, None, standard,
            )
            result = score_one(client, MODEL, 0.0, "enabled", job)
            for blank in result["blanks"]:
                _remember(memory, blank, set())
            _append(flow_path, {
                "id": key,
                "process": chain["process"],
                "blank_ids": ids,
                "blanks": result["blanks"],
                "finish_reason": result.get("finish_reason") or "",
                "reasoning_effort": result.get("reasoning_effort") or "",
            }, lock)
            print(
                f"flow {chain['process']} {index} ID{ids[-1]} {result.get('finish_reason')}",
                flush=True,
            )
        if anchor in ids:
            reached = True
            break
    if not reached:
        raise RuntimeError(f"{chain['process']} 没有走到 ID{anchor}")
    return memory


def _probe_row(chain: dict, standard: dict, item: dict, kind: str, prediction: str, finish: str, effort: str) -> dict:
    gold = _gold(chain, item["blank"])
    std = standard.get((chain["process"], item["blank"]), "")
    return {
        "id": f"{chain['process']}:{item['name']}:{kind}",
        "process": chain["process"],
        "name": item["name"],
        "kind": kind,
        "blank_id": item["blank"],
        "prediction": prediction,
        "label": _label(prediction, gold, std),
        "gold": gold,
        "standard": std,
        "finish_reason": finish,
        "reasoning_effort": effort,
    }


def _run_process(client, row: dict, spec: dict, standard: dict, flow_path: Path, call_path: Path, lock: threading.Lock) -> list[dict]:
    chain = _chain(row, spec["process"])
    memory = _walk(client, chain, spec["deep_anchor"], standard, flow_path, lock)
    masked = _copy(memory)
    hide_ids = spec["hide"] if "hide" in spec else tuple(item["blank"] for item in spec["matrices"])
    for blank_id in hide_ids:
        if blank_id not in masked:
            raise KeyError(f"{spec['process']} 的模型前缀缺少 ID{blank_id}")
        masked[blank_id] = PLACEHOLDER
    done = _done_ids(call_path)
    found = []
    for item in spec["matrices"]:
        question = _question(item["name"])
        calls = (
            ("shallow", _probe_prompt(chain, item["shallow_anchor"], memory, question)),
            ("deep_rederive", _probe_prompt(chain, spec["deep_anchor"], masked, question)),
        )
        for kind, prompt in calls:
            probe_id = f"{spec['process']}:{item['name']}:{kind}"
            if probe_id in done:
                continue
            if PLACEHOLDER in prompt and kind == "shallow":
                raise RuntimeError(f"{probe_id} 的浅层前缀被隐去了矩阵")
            job = {
                "sample_id": SAMPLE,
                "tier": chain.get("tier", ""),
                "process": spec["process"],
                "blank_ids": [item["blank"]],
                "blanks": [],
                "prompt": prompt,
                "max_tokens": THINKING_MAX_TOKENS,
                "stop": None,
                "seed": SEED,
            }
            result = score_one(client, MODEL, 0.0, "enabled", job)
            record = _probe_row(
                chain, standard, item, kind, _extract(result.get("response") or ""),
                result.get("finish_reason") or "", result.get("reasoning_effort") or "",
            )
            record["response"] = result.get("response") or ""
            _append(call_path, record, lock)
            print(f"{probe_id} {record['label']} {record['finish_reason']}", flush=True)
            found.append(record)
    return found


def _report(rows: list[dict]) -> str:
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["process"], row["name"]), {})[row["kind"]] = row
    lines = [
        "# 体轴左手系·无标记，模型前缀，推理强度 low",
        "",
        "这是对照，不是主评测。模型 deepseek-flash，思考开启，推理强度 low。",
        "每个环节从第一句重答到深层锚点。前缀是模型自己的答案。浅层和重推导的答案不写回。",
        "原流程深层没有再测。",
        "",
        "| 环节 | 矩阵 | 浅层 | 重推导 | 两次是否同一张矩阵 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(grouped):
        pair = grouped[key]
        shallow = pair.get("shallow")
        rederive = pair.get("deep_rederive")
        if shallow is None or rederive is None:
            continue
        same = "同一张" if _same(shallow["prediction"], rederive["prediction"]) else "不一致"
        lines.append(
            f"| {key[0]} | {key[1]} | {shallow['label']} | {rederive['label']} | {same} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    load_env_file(EVAL_ROOT / ".env")
    os.environ["DEEPSEEK_REASONING_EFFORT"] = "low"
    row = load_samples(DATA, {SAMPLE})[0]
    standard = standard_answers()
    OUT.mkdir(parents=True, exist_ok=True)
    flow_path = OUT / "flow.jsonl"
    call_path = OUT / "calls.jsonl"
    lock = threading.Lock()
    client = make_client(timeout=1800.0)
    print(f"model prefix from scratch, effort=low, processes={len(TARGETS)}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [
            pool.submit(_run_process, client, row, spec, standard, flow_path, call_path, lock)
            for spec in TARGETS
        ]
        for future in as_completed(futures):
            future.result()
    rows = [json.loads(line) for line in call_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = _report(rows)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
