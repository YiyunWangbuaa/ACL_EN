# -*- coding: utf-8 -*-
"""Session control for robotics theory endpoints (default F11 / public 11).

Default --control-only. Walks forward kinematics through the public ID of --target.
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
    _system_message,
    _turn_question,
    load_samples,
)
from cloze_eval import (  # noqa: E402
    RunStop,
    SESSION_MODEL,
    _resolve_max_tokens,
    score_one,
)
from domains.robotics.manuscript import gold_scored, questions  # noqa: E402
from domains.robotics.t_final import minimal_fixes  # noqa: E402
from semi_auto_eval import load_env_file, make_client, resolve_api_config  # noqa: E402

THEORY = ROOT / "data" / "robotics" / "review" / "theory" / "sample_01.md"
_EXPECTED = {"F11": ("F7",), "F4": ("F3",), "I10": ("I8", "I9")}


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


def _to_public(node: str) -> str:
    return node[1:] if node[:1] in {"F", "I"} else node


def _session_sample(target: str, samples: list[dict]) -> dict:
    if target.startswith("I"):
        return next(item for item in samples if "inverse" in item["title"].lower())
    return next(item for item in samples if "forward" in item["title"].lower())


def _sentences_through(sample: dict, stop_public: str) -> list[tuple[dict, int]]:
    found = []
    for chain in sample["chains"]:
        for index, sentence in enumerate(chain["sentences"]):
            item = sentence["item"]
            if item is None:
                continue
            found.append((chain, index))
            if str(item["id"]) == stop_public:
                return found
    raise RuntimeError(f"{sample['title']} has no ID{stop_public}")


def _job(chain: dict, index: int, messages: list[dict], run: int, seed: int, max_tokens: int, condition: str) -> dict:
    item = chain["sentences"][index]["item"]
    question = "\n".join([
        "Fill the current sentence.",
        "",
        "[Subprocess]",
        chain["process"],
        "",
        "[Question]",
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
        "sample_id": "theory-standard",
        "tier": 0,
        "process": chain["process"],
        "group": "robotics",
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
        "session": chain.get("session") or "",
        "system": messages[0]["content"],
    }


def run_trial(spec, sample, client, model, thinking, effort, run, seed, max_tokens, out, control_only, target, kernel) -> dict:
    del effort
    public_target = _to_public(target)
    public_kernel = tuple(_to_public(node) for node in kernel)
    run_dir = out / f"run_{run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    steps = _sentences_through(sample, public_target)
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
        print(f"run {run + 1} asking ID{item['id']}", flush=True)
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
    for node in public_kernel:
        for path in sorted(run_dir.glob("turn_*.json")):
            data = _load(path)
            for blank in data.get("blanks") or []:
                if str(blank.get("blank_id")) == node:
                    kernel_rows[node] = _slim(blank)
        if node not in kernel_rows:
            raise RuntimeError(f"kernel slot ID{node} did not enter the session")
    wrong_ids = [node for node in public_kernel if not kernel_rows[node].get("normalized")]
    chain, sentence_index = steps[-1]
    control_path = run_dir / "control.json"
    if control_path.is_file() and not _load(control_path).get("error"):
        control = _load(control_path)
    else:
        print(f"run {run + 1} control question ID{public_target}(wrong slots {','.join(wrong_ids) or 'none'})", flush=True)
        job = _job(chain, sentence_index, messages, run, seed, max_tokens, SESSION_MODEL)
        result = score_one(client, model, 0.0, thinking, job)
        blank = {}
        for item in result.get("blanks") or []:
            if str(item.get("blank_id")) == public_target:
                blank = _slim(item)
        control = {
            "blank_ids": [public_target],
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
        return {"run": run, "status": "failed", "where": [public_target], "wrong_ids": wrong_ids}
    control_blank = control.get("blank") or {}
    if not wrong_ids:
        status = "residue" if not control_blank.get("normalized") else "clean"
        return {"run": run, "status": status, "wrong_ids": [], "control": control_blank}
    if control_only:
        print(f"run {run + 1} control recorded, wrong slots {','.join(wrong_ids)}, not forking yet", flush=True)
        return {"run": run, "status": "control", "wrong_ids": wrong_ids, "control": control_blank}
    raise RuntimeError("this round does not fork by default")


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
    public_target = _to_public(target)
    public_kernel = [_to_public(node) for node in kernel]
    waiting = [row for row in rows if row["status"] == "control"]
    residue = [row for row in rows if row["status"] == "residue"]
    clean = [row for row in rows if row["status"] == "clean"]
    failed = [row for row in rows if row["status"] == "failed"]
    lines = [
        f"# Session fork: robotics theory {target}",
        "",
        f"Sample theory/sample_01, through public id ID{public_target}. Model {model}, thinking {thinking}, effort {effort}.",
        f"Kernel slots {list(kernel)} (session ids {[f'ID{n}' for n in public_kernel]}). Control is session-model; not forking yet.",
        "",
        f"Planned {runs} runs, completed {len(rows)}: waiting to fork {len(waiting)}, residue {len(residue)}, clean {len(clean)}, failed {len(failed)}.",
        "",
        "| slot | control wrong |",
        "| --- | ---: |",
    ]
    finished = [row for row in rows if row["status"] != "failed"]
    for node in public_kernel:
        wrong = sum(1 for row in finished if node in row.get("wrong_ids", []))
        denom = len(finished)
        lines.append(f"| ID{node} | {wrong}/{denom} = {wrong / denom:.0%} |" if denom else f"| ID{node} | — |")
    target_wrong = sum(1 for row in finished if not (row.get("control") or {}).get("normalized"))
    denom = len(finished)
    lines.append(f"| ID{public_target} | {target_wrong}/{denom} = {target_wrong / denom:.0%} |" if denom else f"| ID{public_target} | — |")
    lines.append("")
    base = len(residue) + len(clean)
    if base:
        lines.append(f"When all kernel slots are correct, ID{public_target} is still wrong: {len(residue)}/{base} = {len(residue) / base:.0%}.")
    else:
        lines.append(f"When all kernel slots are correct, ID{public_target} is still wrong: —(no run has every kernel slot correct).")
    lines.append("")
    lines.append(f"| run | status | wrong kernel slots | control ID{public_target} |")
    lines.append("| ---: | --- | --- | --- |")
    labels = {"control": "waiting to fork", "residue": "residue", "clean": "clean", "failed": "failed"}
    for row in rows:
        wrong = ", ".join(f"ID{item}" for item in row.get("wrong_ids") or []) or "—"
        pred = (row.get("control") or {}).get("prediction") or "—"
        lines.append(f"| {row['run'] + 1} | {labels.get(row['status'], row['status'])} | {wrong} | {pred} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="F11")
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
        raise RuntimeError(f"{target} kernel should be {expected}, got {kernel}")
    for node in kernel:
        if node not in questions():
            raise RuntimeError(node)
        _ = gold_scored()[node]
    out = args.out or (ROOT / "results" / f"fork_robotics_{target.lower()}" / "sample_01")
    report = args.report or (ROOT / "doc" / f"session_fork_robotics_{target}.md")
    control_only = False if args.allow_fork else True
    print(f"shape check passed: robotics {target} kernel {kernel}", flush=True)
    if args.check:
        return
    load_env_file(ROOT / ".env")
    os.environ["DEEPSEEK_REASONING_EFFORT"] = args.effort
    spec, samples = load_samples(THEORY)
    sample = _session_sample(target, samples)
    max_tokens = _resolve_max_tokens(0, args.thinking, announce=False)
    client = make_client(timeout=1800.0 if args.thinking == "enabled" else 600.0)
    model = args.model or resolve_api_config()[2]
    meta = {
        "sample": "theory/sample_01",
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
            raise SystemExit("saved results were run under different conditions")
    _save(meta_path, meta)
    print(
        f"sample theory model {model} thinking {args.thinking} effort {args.effort} "
        f"kernel {kernel} runs {args.runs} {'control only' if control_only else 'fork after control'}",
        flush=True,
    )
    try:
        for run in range(args.runs):
            run_trial(
                spec, sample, client, model, args.thinking, args.effort,
                run, 20261102 + run, max_tokens, out, control_only, target, kernel,
            )
    except RunStop as exc:
        print(f"call interrupted: {exc}", flush=True)
    rows = collect(out, args.runs)
    text = render(args.model, args.thinking, args.effort, args.runs, rows, target, kernel)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
