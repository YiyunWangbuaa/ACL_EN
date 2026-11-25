# -*- coding: utf-8 -*-
"""Session control for vision basic endpoint ID41 (kernel 37–40).

Default --control-only. Walks basic theory through --target. CoT is not written back.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_robotics_one import (  # noqa: E402
    SLOT_RE,
    _options,
    _system_message,
    _turn_question,
)
from cloze_eval import (  # noqa: E402
    RunStop,
    SESSION_MODEL,
    _resolve_max_tokens,
    score_one,
)
from domains.computer_vision.manuscript import gold_scored, questions  # noqa: E402
from domains.computer_vision.t_final import minimal_fixes  # noqa: E402
from semi_auto_eval import load_env_file, make_client, resolve_api_config  # noqa: E402

BASIC = ROOT / "data" / "vision" / "review" / "scene" / "sample_03.md"
_SPEC = "\n".join([
    "选择题只回答一个选项字母。",
    "矩阵写成 [[a, b, c], [d, e, f], [g, h, i]]。",
    "向量写成 [a, b, c]。",
])
_EXPECTED = {"41": ("37", "38", "39", "40"), "47": ("41",), "53": ("41", "49")}


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


def _sentences_by_slot(body: str) -> list[dict]:
    """One turn per blank. Vision review packs several blanks into one period."""
    pieces = []
    cursor = 0
    for match in SLOT_RE.finditer(body):
        chunk = body[cursor:match.end()]
        # Keep trailing punctuation that belongs to this blank if present.
        end = match.end()
        while end < len(body) and body[end] in "。；;，,":
            end += 1
            break
        chunk = body[cursor:end]
        options = _options(match.group(2))
        item = {
            "id": match.group(1),
            "options": options,
            "format_code": match.group(3),
            "label": match.group(4),
            "answer": match.group(5),
            "span": (match.start() - cursor, match.end() - cursor),
        }
        pieces.append({"text": chunk, "item": item})
        cursor = end
    tail = body[cursor:].strip()
    if tail and SLOT_RE.search(tail):
        raise RuntimeError(f"未消化的空：{tail[:80]}")
    return pieces


def _chains_by_slot(title: str, process: str, declaration: str) -> tuple[list[dict], list[str]]:
    parts = re.split(r"\n### ", process)
    chains = []
    seen = []
    for part in parts[1:]:
        name, _, body = part.partition("\n")
        sentences = _sentences_by_slot(body.strip() + ("\n" if not body.endswith("\n") else ""))
        ids = [item["item"]["id"] for item in sentences if item["item"] is not None]
        if not ids:
            raise RuntimeError(f"{title} / {name.strip()} 没有空。")
        seen.extend(ids)
        chains.append({
            "process": name.strip(),
            "sentences": sentences,
            "ids": ids,
            "declaration": declaration,
            "prior": "",
            "session": title,
        })
    return chains, seen


def load_basic(path: Path, section: str = "theory") -> tuple[str, dict]:
    raw = path.read_text(encoding="utf-8")
    if "# 前置环节：基础理论" in raw and "# 物体与双目" in raw:
        theory, scene = raw.split("# 前置环节：基础理论", 1)[1].split("# 物体与双目", 1)
        raw = theory if section == "theory" else scene
    declaration = raw.split("## 定义声明", 1)[1].split("## 完整过程", 1)[0]
    process = raw.split("## 完整过程", 1)[1]
    if "\n### " in process:
        process = process[process.index("\n### "):]
    chains, seen = _chains_by_slot("基础理论", process, declaration.strip())
    expected = [str(number) for number in range(1, len(seen) + 1)]
    if seen != expected:
        raise RuntimeError(f"视觉题号是 {seen[:8]}…{seen[-4:]}，应当是 1 到 {len(seen)}")
    sample = {
        "sample_id": "basic-01",
        "title": "基础理论",
        "declaration": declaration.strip(),
        "prior": "",
        "chains": chains,
        "group": "vision",
    }
    return _SPEC, sample


def _sentences_through(sample: dict, stop_id: str) -> list[tuple[dict, int]]:
    found = []
    for chain in sample["chains"]:
        for index, sentence in enumerate(chain["sentences"]):
            item = sentence["item"]
            if item is None:
                continue
            found.append((chain, index))
            if str(item["id"]) == stop_id:
                return found
    raise RuntimeError(f"基础理论没有 ID{stop_id}")


def _job(chain, index, messages, run, seed, max_tokens, condition) -> dict:
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
        "sample_id": "basic-01",
        "tier": 0,
        "process": chain["process"],
        "group": "vision",
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
        "prompt": question,
        "session": "基础理论",
        "system": messages[0]["content"],
    }


def run_trial(spec, sample, client, model, thinking, effort, run, seed, max_tokens, out, control_only, target, kernel) -> dict:
    del effort
    run_dir = out / f"run_{run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = _sentences_through(sample, target)
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
    kernel_rows = {}
    for node in kernel:
        for path in sorted(run_dir.glob("turn_*.json")):
            data = _load(path)
            for blank in data.get("blanks") or []:
                if str(blank.get("blank_id")) == node:
                    kernel_rows[node] = _slim(blank)
        if node not in kernel_rows:
            raise RuntimeError(f"核槽 ID{node} 没有进入会话")
    wrong_ids = [node for node in kernel if not kernel_rows[node].get("normalized")]
    chain, sentence_index = steps[-1]
    control_path = run_dir / "control.json"
    if control_path.is_file() and not _load(control_path).get("error"):
        control = _load(control_path)
    else:
        print(f"第 {run + 1} 趟 对照提问 ID{target}（错槽 {','.join(wrong_ids) or '无'}）", flush=True)
        job = _job(chain, sentence_index, messages, run, seed, max_tokens, SESSION_MODEL)
        result = score_one(client, model, 0.0, thinking, job)
        blank = {}
        for item in result.get("blanks") or []:
            if str(item.get("blank_id")) == target:
                blank = _slim(item)
        control = {
            "blank_ids": [target],
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
        return {"run": run, "status": "failed", "where": [target], "wrong_ids": wrong_ids}
    control_blank = control.get("blank") or {}
    if not wrong_ids:
        status = "residue" if not control_blank.get("normalized") else "clean"
        return {"run": run, "status": status, "wrong_ids": [], "control": control_blank}
    if control_only:
        print(f"第 {run + 1} 趟 对照已记下，错槽 {','.join(wrong_ids)}，暂不分叉", flush=True)
        return {"run": run, "status": "control", "wrong_ids": wrong_ids, "control": control_blank}
    raise RuntimeError("本轮默认不分叉")


def collect(out: Path, runs: int) -> list[dict]:
    rows = []
    for run in range(runs):
        control_path = out / f"run_{run:02d}" / "control.json"
        if not control_path.is_file():
            rows.append({"run": run, "status": "failed"})
            continue
        control = _load(control_path)
        blank = control.get("blank") or {}
        wrong_ids = list(control.get("wrong_ids") or [])
        if not wrong_ids:
            status = "residue" if not blank.get("normalized") else "clean"
        else:
            status = "control"
        rows.append({"run": run, "status": status, "wrong_ids": wrong_ids, "control": blank})
    return rows


def render(model, thinking, effort, runs, rows, target, kernel) -> str:
    waiting = [row for row in rows if row["status"] == "control"]
    residue = [row for row in rows if row["status"] == "residue"]
    clean = [row for row in rows if row["status"] == "clean"]
    failed = [row for row in rows if row["status"] == "failed"]
    lines = [
        f"# 会话分叉：机器视觉 ID{target}",
        "",
        f"样本 basic/sample_01，走到 ID{target}。模型 {model}，思考 {thinking}，强度 {effort}。",
        f"核槽 {list(kernel)}。对照 session-model；暂不分叉。",
        "",
        f"计划 {runs} 趟，完成 {len(rows)}：待分叉 {len(waiting)}，残留 {len(residue)}，干净 {len(clean)}，失败 {len(failed)}。",
        "",
        "| 槽 | 对照答错 |",
        "| --- | ---: |",
    ]
    finished = [row for row in rows if row["status"] != "failed"]
    for node in kernel:
        wrong = sum(1 for row in finished if node in row.get("wrong_ids", []))
        denom = len(finished)
        lines.append(f"| ID{node} | {wrong}/{denom} = {wrong / denom:.0%} |" if denom else f"| ID{node} | — |")
    target_wrong = sum(1 for row in finished if not (row.get("control") or {}).get("normalized"))
    denom = len(finished)
    lines.append(f"| ID{target} | {target_wrong}/{denom} = {target_wrong / denom:.0%} |" if denom else f"| ID{target} | — |")
    lines.append("")
    base = len(residue) + len(clean)
    if base:
        lines.append(f"核槽全对时 ID{target} 仍错：{len(residue)}/{base} = {len(residue) / base:.0%}。")
    else:
        lines.append(f"核槽全对时 ID{target} 仍错：—（没有核槽全对的趟）。")
    lines.append("")
    lines.append(f"| 趟 | 状态 | 错的核槽 | 对照 ID{target} |")
    lines.append("| ---: | --- | --- | --- |")
    labels = {"control": "待分叉", "residue": "残留", "clean": "干净", "failed": "失败"}
    for row in rows:
        wrong = "、".join(f"ID{item}" for item in row.get("wrong_ids") or []) or "—"
        pred = (row.get("control") or {}).get("prediction") or "—"
        lines.append(f"| {row['run'] + 1} | {labels.get(row['status'], row['status'])} | {wrong} | {pred} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="41")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--control-only", action="store_true", default=True)
    parser.add_argument("--allow-fork", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = str(args.target)
    kernel = tuple(minimal_fixes(target)[0])
    expected = _EXPECTED.get(target)
    if expected is not None and kernel != expected:
        raise RuntimeError(f"ID{target} 核应为 {expected}，得到 {kernel}")
    for node in kernel:
        _ = gold_scored()[node]
        _ = questions()[node]
    out = args.out or (ROOT / "results" / f"fork_vision_id{target}" / "sample_01")
    report = args.report or (ROOT / "analysis" / f"会话分叉_机器视觉_ID{target}.md")
    control_only = False if args.allow_fork else True
    print(f"形状检查通过：机器视觉 ID{target} 核 {kernel}", flush=True)
    if args.check:
        return
    load_env_file(ROOT / ".env")
    os.environ["DEEPSEEK_REASONING_EFFORT"] = args.effort
    spec, sample = load_basic(BASIC)
    max_tokens = _resolve_max_tokens(0, args.thinking, announce=False)
    client = make_client(timeout=1800.0 if args.thinking == "enabled" else 600.0)
    model = args.model or resolve_api_config()[2]
    meta = {
        "sample": "basic/sample_01",
        "model": model,
        "thinking": args.thinking,
        "effort": args.effort,
        "kernel": list(kernel),
        "target": target,
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
        f"样本 basic 模型 {model} 思考 {args.thinking} 强度 {args.effort} "
        f"核 {kernel} 趟数 {args.runs} {'只跑对照' if control_only else '对照后分叉'}",
        flush=True,
    )
    try:
        for run in range(args.runs):
            run_trial(
                spec, sample, client, model, args.thinking, args.effort,
                run, 20261102 + run, max_tokens, out, control_only, target, kernel,
            )
    except RunStop as exc:
        print(f"调用中断：{exc}", flush=True)
    rows = collect(out, args.runs)
    text = render(args.model, args.thinking, args.effort, args.runs, rows, target, kernel)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
