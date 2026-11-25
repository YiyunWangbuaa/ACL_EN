# -*- coding: utf-8 -*-
"""Session control for robotics kinematics (default F45). Control-only by default."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.fork_run_robotics import (  # noqa: E402
    _job,
    _load,
    _save,
    _sentences_through,
    _session_sample,
    _slim,
    _to_public,
    collect,
    render,
    run_trial,
)
from analysis.run_robotics_one import load_samples  # noqa: E402
from cloze_eval import RunStop, _resolve_max_tokens  # noqa: E402
from domains.robotics.kin_manuscript import gold_scored, questions  # noqa: E402
from domains.robotics.kin_t_final import minimal_fixes  # noqa: E402
from semi_auto_eval import load_env_file, make_client, resolve_api_config  # noqa: E402

KIN = ROOT / "data" / "robotics" / "review" / "kinematics" / "sample_01.md"
_EXPECTED = {
    "F45": ("F41", "F42", "F43", "F44"),
    "F48": ("F45", "F47"),
    "I16": ("I9", "I14"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="F45")
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
        raise RuntimeError(f"{target} 核应为 {expected}，得到 {kernel}")
    for node in kernel:
        _ = gold_scored()[node]
        _ = questions()[node]
    out = args.out or (ROOT / "results" / f"fork_kinematics_{target.lower()}" / "sample_01")
    report = args.report or (ROOT / "analysis" / f"会话分叉_机器人学运动学_{target}.md")
    control_only = False if args.allow_fork else True
    print(f"形状检查通过：运动学 {target} 核 {kernel}", flush=True)
    if args.check:
        return
    load_env_file(ROOT / ".env")
    os.environ["DEEPSEEK_REASONING_EFFORT"] = args.effort
    spec, samples = load_samples(KIN)
    sample = _session_sample(target, samples)
    max_tokens = _resolve_max_tokens(0, args.thinking, announce=False)
    client = make_client(timeout=1800.0 if args.thinking == "enabled" else 600.0)
    model = args.model or resolve_api_config()[2]
    meta = {
        "sample": "kinematics/sample_01",
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
        f"样本 kinematics 模型 {model} 思考 {args.thinking} 强度 {args.effort} "
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
    text = text.replace("机器人学理论", "机器人学运动学").replace("theory/sample_01", "kinematics/sample_01")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
