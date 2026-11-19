# -*- coding: utf-8 -*-
"""One hetero definition, thinking on, reasoning effort low.

The frozen prefix is generator gold, fixed before any probe. Probe answers are
scored and stored here. They are not written into later prompts. The single
write-back call is a control: it appends the R_γ_M deep-rederive answer to the
original R_t prefix after that probe has already been scored.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from cloze_eval import THINKING_MAX_TOKENS, _repair_phrase, contexts_for, load_env_file, load_samples  # noqa: E402
from cloze_score import judge  # noqa: E402
from domains.wind_tunnel.rt_isolation import (  # noqa: E402
    MATRICES,
    PROBE_MARK,
    PROCESS,
    RT_ID,
    STAGES,
    _inject_control,
    build_conditions,
    isolate_stage,
)
from semi_auto_eval import call_model, make_client  # noqa: E402

DATA = EVAL_ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = EVAL_ROOT / "results" / "rt_low_body"
SAMPLE = "body-unmarked"
STANDARD = "standard-default-unmarked"
MODEL = "deepseek-flash"
EFFORT = "low"
WORKERS = 4
SEED = 20261102


def _chain(row: dict, process: str) -> dict:
    chain = next(item for item in contexts_for(row) if item["process"] == process)
    if not chain.get("process_text"):
        raise KeyError(f"{SAMPLE} 缺少{process}题面")
    chain["declaration"] = row["declaration"]
    return chain


def _frozen(chain: dict) -> dict[str, str]:
    memory = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            memory[str(blank["blank_id"])] = _repair_phrase(blank)
    return memory


def _gold(chain: dict, blank_id: str) -> str:
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            if str(blank["blank_id"]) == str(blank_id):
                return _repair_phrase(blank)
    raise KeyError(blank_id)


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


def _jobs(contract: dict, standard: dict) -> list[dict]:
    jobs = []
    wind = _chain(contract, PROCESS)
    frozen = _frozen(wind)
    held = {
        item["key"]: {"shallow": f"{PROBE_MARK}浅层{item['name']}", "deep": f"{PROBE_MARK}深层{item['name']}"}
        for item in MATRICES
    }
    prompts = build_conditions(wind, frozen, held)
    prompts.pop("rt_writeback_control")
    for item in MATRICES:
        gold = _gold(wind, item["blank"])
        standard_gold = _gold(_chain(standard, PROCESS), item["blank"])
        for kind, key in (
            ("shallow", f"shallow_{item['key']}"),
            ("deep_original", f"deep_original_{item['key']}"),
            ("deep_rederive", f"deep_rederive_{item['key']}"),
        ):
            jobs.append({
                "id": f"{PROCESS}:{item['name']}:{kind}",
                "process": PROCESS,
                "name": item["name"],
                "kind": kind,
                "blank_id": item["blank"],
                "prompt": prompts[key],
                "gold": gold,
                "standard": standard_gold,
            })
    total_gold = _gold(wind, RT_ID)
    total_standard = _gold(_chain(standard, PROCESS), RT_ID)
    for kind, key in (("total_original", "rt_original"), ("total_rederive", "rt_rederive")):
        jobs.append({
            "id": f"{PROCESS}:R_t:{kind}",
            "process": PROCESS,
            "name": "R_t",
            "kind": kind,
            "blank_id": RT_ID,
            "prompt": prompts[key],
            "gold": total_gold,
            "standard": total_standard,
        })
    for stage in STAGES:
        chain = _chain(contract, stage["process"])
        std_chain = _chain(standard, stage["process"])
        stage_prompts = isolate_stage(chain, _frozen(chain), stage)
        for item in stage["matrices"]:
            gold = _gold(chain, item["blank"])
            standard_gold = _gold(std_chain, item["blank"])
            for kind, key in (
                ("shallow", f"shallow_{item['key']}"),
                ("deep_original", f"deep_original_{item['key']}"),
                ("deep_rederive", f"deep_rederive_{item['key']}"),
            ):
                jobs.append({
                    "id": f"{stage['process']}:{item['name']}:{kind}",
                    "process": stage["process"],
                    "name": item["name"],
                    "kind": kind,
                    "blank_id": item["blank"],
                    "prompt": stage_prompts[key],
                    "gold": gold,
                    "standard": standard_gold,
                })
        if stage.get("total_id"):
            gold = _gold(chain, stage["total_id"])
            standard_gold = _gold(std_chain, stage["total_id"])
            for kind, key in (("total_original", "total_original"), ("total_rederive", "total_rederive")):
                jobs.append({
                    "id": f"{stage['process']}:R:{kind}",
                    "process": stage["process"],
                    "name": "R",
                    "kind": kind,
                    "blank_id": stage["total_id"],
                    "prompt": stage_prompts[key],
                    "gold": gold,
                    "standard": standard_gold,
                })
    return jobs


def _done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            found.add(json.loads(line)["id"])
    return found


def _call(client, job: dict) -> dict:
    captured: dict = {}
    response = ""
    reasoning = ""
    last_error = ""
    for attempt in range(4):
        try:
            response, reasoning = call_model(
                client,
                MODEL,
                job["prompt"],
                0.0,
                "enabled",
                captured=captured,
                max_tokens=THINKING_MAX_TOKENS,
                stop=None,
                seed=SEED,
                reasoning_effort=EFFORT,
            )
            last_error = ""
            break
        except Exception as exc:  # noqa: BLE001 - retry a transient API error
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(20, 2 ** attempt))
    prediction = _extract(response)
    row = {
        "id": job["id"],
        "process": job["process"],
        "name": job["name"],
        "kind": job["kind"],
        "blank_id": job["blank_id"],
        "prediction": prediction,
        "label": _label(prediction, job["gold"], job["standard"]) if not last_error else "调用失败",
        "gold": job["gold"],
        "standard": job["standard"],
        "response": response,
        "reasoning_chars": len(reasoning or ""),
        "finish_reason": captured.get("finish_reason") or "",
        "reasoning_effort": captured.get("reasoning_effort") or "",
        "model": MODEL,
        "error": last_error,
    }
    if PROBE_MARK in job["prompt"]:
        raise RuntimeError(f"{job['id']} 的提示里带了探针记录")
    return row


def _write(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _report(rows: list[dict]) -> str:
    order = {"shallow": 0, "deep_original": 1, "deep_rederive": 2, "total_original": 3, "total_rederive": 4, "writeback": 5}
    rows = sorted(rows, key=lambda item: (item["process"], order.get(item["kind"], 9), int(item["blank_id"])))
    lines = [
        "# 体轴左手系·无标记，推理强度 low",
        "",
        "这是对照，不是主评测。模型 deepseek-flash，思考开启，推理强度 low。",
        "冻结前缀在探针之前取定，内容是生成器金标。探针答案只记在本目录，不写回后续提示。",
        "写回控制只把 R_γ_M 的深层重推导答案追加到原始总矩阵前缀上，不并入主结果。",
        "",
        "| 环节 | 矩阵 | 条件 | 判定 | 结束 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['process']} | {row['name']} | {row['kind']} | {row['label']} | {row.get('finish_reason') or row.get('error') or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    load_env_file(EVAL_ROOT / ".env")
    contract = load_samples(DATA, {SAMPLE})[0]
    standard = load_samples(DATA, {STANDARD})[0]
    jobs = _jobs(contract, standard)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "calls.jsonl"
    done = _done(path)
    pending = [job for job in jobs if job["id"] not in done]
    client = make_client(timeout=1800.0)
    lock = threading.Lock()
    print(f"pending {len(pending)} / {len(jobs)} effort={EFFORT}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_call, client, job): job["id"] for job in pending}
        for future in as_completed(futures):
            row = future.result()
            _write(path, row, lock)
            print(f"{row['id']} {row['label']} {row['finish_reason']}", flush=True)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    gamma = next((row for row in rows if row["id"] == f"{PROCESS}:R_γ_M:deep_rederive"), None)
    writeback_id = f"{PROCESS}:R_γ_M:writeback"
    if gamma and gamma["label"] != "调用失败" and writeback_id not in {row["id"] for row in rows}:
        wind_jobs = {job["id"]: job for job in jobs}
        original = wind_jobs[f"{PROCESS}:R_t:total_original"]
        control = dict(original)
        control["id"] = writeback_id
        control["kind"] = "writeback"
        control["prompt"] = _inject_control(original["prompt"], gamma["prediction"])
        row = _call(client, control)
        _write(path, row, lock)
        rows.append(row)
        print(f"{row['id']} {row['label']} {row['finish_reason']}", flush=True)
    report = _report(rows)
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
