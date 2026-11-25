# -*- coding: utf-8 -*-
"""Session control/fork for flight endpoints (ID45 / ID81 kernels).

Default is --control-only. Walks the owning session through --target.
CoT is not written back.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_robotics_one import (  # noqa: E402
    FLOW,
    _gold_turn,
    _system_message,
    _turn_question,
    load_flow,
)
from cloze_eval import (  # noqa: E402
    RunStop,
    SESSION_MODEL,
    _resolve_max_tokens,
    score_one,
    transcript,
)
from domains.aerodynamics.manuscript import gold_scored, questions  # noqa: E402
from domains.aerodynamics.t_final import minimal_fixes  # noqa: E402
from semi_auto_eval import load_env_file, make_client, resolve_api_config  # noqa: E402

FORK_CONDITION = "session-fork"
_EXPECTED_KERNEL = {
    "45": ("42", "43"),
    "81": ("79", "80"),
}


def _session_for(target_id: str) -> str:
    number = int(target_id)
    if number <= 59:
        return "坐标转换"
    if number <= 102:
        return "气动力"
    return "参考点"


def _kernel_for(target_id: str) -> tuple[str, ...]:
    fixes = minimal_fixes(target_id)
    if not fixes:
        raise RuntimeError(f"ID{target_id} 没有修复核")
    return tuple(fixes[0])


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _slim(blank: dict) -> dict:
    return {
        "blank_id": str(blank["blank_id"]),
        "answer": blank.get("answer") or "",
        "prediction": blank.get("prediction") or "",
        "exact": bool(blank.get("exact")),
        "normalized": bool(blank.get("normalized")),
        "error_kind": blank.get("error_kind") or "",
        "options": [str(item) for item in (blank.get("options") or [])],
    }


def _gold_json(node: str) -> str:
    row = questions()[node]
    scored = gold_scored()[node]
    if row["options"]:
        return "{" + f"{node}:{scored}" + "}"
    escaped = scored.replace("\\", "\\\\").replace('"', '\\"')
    return "{" + f'{node}:"{escaped}"' + "}"


def _sentences_through(sample: dict, stop_id: str) -> list[tuple[dict, int]]:
    """(chain, sentence_index) pairs up to and including stop_id."""
    session = _session_for(stop_id)
    found = []
    for chain in sample["chains"]:
        if chain.get("session") != session:
            continue
        for index, sentence in enumerate(chain["sentences"]):
            item = sentence["item"]
            if item is None:
                continue
            found.append((chain, index))
            if str(item["id"]) == stop_id:
                return found
    raise RuntimeError(f"{session} 没有 ID{stop_id}")


def check_shape(target_id: str, kernel: tuple[str, ...]) -> None:
    expected = _EXPECTED_KERNEL.get(target_id)
    if expected is not None and kernel != expected:
        raise RuntimeError(f"ID{target_id} 的核应为 {expected}，得到 {kernel}")
    for node in kernel:
        _gold_json(node)
    print(f"形状检查通过：飞行力学 ID{target_id} 会话 {_session_for(target_id)} 核 {kernel}", flush=True)


def _job(chain: dict, index: int, messages: list[dict], run: int, seed: int, max_tokens: int, condition: str) -> dict:
    item = chain["sentences"][index]["item"]
    question = "\n".join([
        "填写当前这一句。",
        "",
        "【子过程】",
        chain["process"],
        "",
        "【问题】",
        _turn_question(chain, index),
    ])
    blank = {
        "blank_id": item["id"],
        "label": item["label"],
        "format_code": item["format_code"],
        "answer": item["answer"],
        "options": list(item["options"]),
        "process": chain["process"],
    }
    history = [*messages, {"role": "user", "content": question}]
    return {
        "sample_id": "flow-standard",
        "tier": 0,
        "process": chain["process"],
        "group": "flight",
        "blank_ids": [item["id"]],
        "blanks": [blank],
        "condition": condition,
        "run": run,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": None,
        "standard": {(chain["process"], item["id"]): item["answer"]},
        "question": question,
        "messages": history,
        "prompt": transcript(history),
        "session": chain.get("session") or _session_for(str(item["id"])),
        "system": messages[0]["content"],
    }


def run_trial(spec: str, sample: dict, client, model: str, thinking: str, effort: str, run: int, seed: int, max_tokens: int, out: Path, control_only: bool, target_id: str, kernel_ids: tuple[str, ...]) -> dict:
    del effort
    run_dir = out / f"run_{run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = _sentences_through(sample, target_id)
    first_chain = steps[0][0]
    messages = [{"role": "system", "content": _system_message(spec, first_chain)}]
    for turn_index, (chain, sentence_index) in enumerate(steps[:-1]):
        path = run_dir / f"turn_{turn_index:02d}.json"
        item = chain["sentences"][sentence_index]["item"]
        job = _job(chain, sentence_index, messages, run, seed, max_tokens, SESSION_MODEL)
        if path.is_file() and not _load(path).get("error"):
            record = _load(path)
            messages.append({"role": "user", "content": record["question"]})
            messages.append({"role": "assistant", "content": record["committed"]})
            continue
        print(f"第 {run + 1} 趟 提问 ID{item['id']}", flush=True)
        result = score_one(client, model, 0.0, thinking, job)
        reply = result.get("response") or ""
        if result.get("reasoning") and len(result["reasoning"]) > 40 and result["reasoning"] in job["question"]:
            raise RuntimeError("思维链被写回了用户消息")
        record = {
            "blank_ids": [item["id"]],
            "question": job["question"],
            "committed": reply,
            "response": reply,
            "error": result.get("error") or "",
            "unanswered": bool(result.get("unanswered")),
            "blanks": [_slim(blank) for blank in result.get("blanks") or []],
        }
        _save(path, record)
        if record["error"] or record["unanswered"]:
            return {"run": run, "status": "failed", "where": [item["id"]]}
        messages.append({"role": "user", "content": job["question"]})
        messages.append({"role": "assistant", "content": reply})
    kernel = {}
    for node in kernel_ids:
        for path in sorted(run_dir.glob("turn_*.json")):
            data = _load(path)
            for blank in data.get("blanks") or []:
                if str(blank.get("blank_id")) == node:
                    kernel[node] = _slim(blank)
        if node not in kernel:
            raise RuntimeError(f"核槽 ID{node} 没有进入会话")
    wrong_ids = [node for node in kernel_ids if not kernel[node].get("normalized")]
    chain, sentence_index = steps[-1]
    control_path = run_dir / "control.json"
    if control_path.is_file() and not _load(control_path).get("error"):
        control = _load(control_path)
    else:
        print(f"第 {run + 1} 趟 对照提问 ID{target_id}（错槽 {','.join(wrong_ids) or '无'}）", flush=True)
        job = _job(chain, sentence_index, messages, run, seed, max_tokens, SESSION_MODEL)
        result = score_one(client, model, 0.0, thinking, job)
        blank = {}
        for item in result.get("blanks") or []:
            if str(item.get("blank_id")) == target_id:
                blank = _slim(item)
        control = {
            "blank_ids": [target_id],
            "question": job["question"],
            "committed": result.get("response") or "",
            "response": result.get("response") or "",
            "error": result.get("error") or "",
            "unanswered": bool(result.get("unanswered")),
            "blanks": [_slim(item) for item in result.get("blanks") or []],
            "blank": blank,
            "wrong_ids": wrong_ids,
        }
        _save(control_path, control)
    if control.get("error") or control.get("unanswered"):
        return {"run": run, "status": "failed", "where": [target_id], "wrong_ids": wrong_ids}
    control_blank = control.get("blank") or {}
    if not wrong_ids:
        status = "residue" if not control_blank.get("normalized") else "clean"
        print(f"第 {run + 1} 趟 核槽都是金标，状态 {status}", flush=True)
        return {"run": run, "status": status, "wrong_ids": [], "control": control_blank}
    if control_only:
        print(f"第 {run + 1} 趟 对照已记下，错槽 {','.join(wrong_ids)}，暂不分叉", flush=True)
        return {"run": run, "status": "control", "wrong_ids": wrong_ids, "control": control_blank}
    raise RuntimeError("本轮默认不分叉；去掉 --control-only 前先确认样本量")


def collect(out: Path, runs: int) -> list[dict]:
    rows = []
    for run in range(runs):
        run_dir = out / f"run_{run:02d}"
        control_path = run_dir / "control.json"
        if not control_path.is_file():
            rows.append({"run": run, "status": "failed"})
            continue
        control = _load(control_path)
        blank = control.get("blank") or {}
        wrong_ids = list(control.get("wrong_ids") or [])
        if not wrong_ids:
            status = "residue" if not blank.get("normalized") else "clean"
        elif (run_dir / "fork.json").is_file():
            status = "kept"
        else:
            status = "control"
        rows.append({"run": run, "status": status, "wrong_ids": wrong_ids, "control": blank})
    return rows


def render(model: str, thinking: str, effort: str, runs: int, rows: list[dict], target_id: str, kernel_ids: tuple[str, ...]) -> str:
    waiting = [row for row in rows if row["status"] == "control"]
    residue = [row for row in rows if row["status"] == "residue"]
    clean = [row for row in rows if row["status"] == "clean"]
    failed = [row for row in rows if row["status"] == "failed"]
    kept = [row for row in rows if row["status"] == "kept"]
    session = _session_for(target_id)
    lines = [
        f"# 会话分叉：飞行力学 ID{target_id}",
        "",
        f"样本 sample_01，会话 {session} 走到 ID{target_id}。模型 {model}，思考 {thinking}，强度 {effort}。",
        f"核槽 {list(kernel_ids)}。对照是 session-model；分叉只换错槽。思维链不写回。",
        "",
        f"计划 {runs} 趟，完成记录 {len(rows)} 趟：分叉 {len(kept)}，待分叉 {len(waiting)}，"
        f"公式层残留 {len(residue)}，干净 {len(clean)}，调用失败 {len(failed)}。",
        "",
        "| 槽 | 对照答错 |",
        "| --- | ---: |",
    ]
    finished = [row for row in rows if row["status"] != "failed"]
    for node in kernel_ids:
        wrong = sum(1 for row in finished if node in row.get("wrong_ids", []))
        denom = len(finished)
        cell = f"{wrong}/{denom} = {wrong / denom:.0%}" if denom else "—"
        lines.append(f"| ID{node} | {cell} |")
    target_wrong = sum(1 for row in finished if not (row.get("control") or {}).get("normalized"))
    denom = len(finished)
    lines.append(f"| ID{target_id} | {target_wrong}/{denom} = {target_wrong / denom:.0%} |" if denom else f"| ID{target_id} | — |")
    lines.append("")
    base = len(residue) + len(clean)
    if base:
        lines.append(f"核槽全对时 ID{target_id} 仍错：{len(residue)}/{base} = {len(residue) / base:.0%}。")
    else:
        lines.append(f"核槽全对时 ID{target_id} 仍错：—（没有核槽全对的趟）。")
    lines.append("")
    lines.append(f"| 趟 | 状态 | 错的核槽 | 对照 ID{target_id} |")
    lines.append("| ---: | --- | --- | --- |")
    labels = {"control": "待分叉", "kept": "已分叉", "residue": "残留", "clean": "干净", "failed": "失败"}
    for row in rows:
        wrong = "、".join(f"ID{item}" for item in row.get("wrong_ids") or []) or "—"
        pred = (row.get("control") or {}).get("prediction") or "—"
        lines.append(f"| {row['run'] + 1} | {labels.get(row['status'], row['status'])} | {wrong} | {pred} |")
    if waiting and not kept:
        lines.append("")
        lines.append("对照已记下，分叉还没跑。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="45")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--control-only", action="store_true", default=True)
    parser.add_argument("--allow-fork", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    target_id = str(args.target)
    kernel_ids = _kernel_for(target_id)
    out = args.out or (ROOT / "results" / f"fork_flight_id{target_id}" / "sample_01")
    report = args.report or (ROOT / "analysis" / f"会话分叉_飞行力学_ID{target_id}.md")
    control_only = False if args.allow_fork else True
    check_shape(target_id, kernel_ids)
    if args.check:
        print("检验通过。", flush=True)
        return
    if not args.report_only:
        load_env_file(ROOT / ".env")
        os.environ["DEEPSEEK_REASONING_EFFORT"] = args.effort
        spec, samples = load_flow(FLOW)
        sample = samples[0]
        max_tokens = _resolve_max_tokens(0, args.thinking, announce=False)
        client = make_client(timeout=1800.0 if args.thinking == "enabled" else 600.0)
        model = args.model or resolve_api_config()[2]
        meta = {
            "sample": "sample_01",
            "model": model,
            "thinking": args.thinking,
            "effort": args.effort,
            "kernel": list(kernel_ids),
            "target": target_id,
            "session": _session_for(target_id),
            "runs": args.runs,
        }
        meta_path = out / "meta.json"
        if meta_path.is_file():
            saved = _load(meta_path)
            keys = ("sample", "model", "thinking", "effort", "kernel", "target")
            if {key: saved.get(key) for key in keys} != {key: meta[key] for key in keys}:
                raise SystemExit("已有结果的条件与这次不一致")
        _save(meta_path, meta)
        print(
            f"样本 sample_01 模型 {model} 思考 {args.thinking} 强度 {args.effort} "
            f"核 {kernel_ids} 趟数 {args.runs} {'只跑对照' if control_only else '对照后分叉'}",
            flush=True,
        )
        try:
            for run in range(args.runs):
                run_trial(
                    spec, sample, client, model, args.thinking, args.effort,
                    run, 20261102 + run, max_tokens, out, control_only, target_id, kernel_ids,
                )
        except RunStop as exc:
            print(f"调用中断：{exc}", flush=True)
    rows = collect(out, args.runs)
    text = render(args.model, args.thinking, args.effort, args.runs, rows, target_id, kernel_ids)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
