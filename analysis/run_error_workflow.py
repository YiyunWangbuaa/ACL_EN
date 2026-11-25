# -*- coding: utf-8 -*-
"""Run error-first diagnosis on stored cloze records. No model calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.contract_executor import generated_answers  # noqa: E402
from domains.wind_tunnel.contract_executor import _model  # noqa: E402
from domains.wind_tunnel.error_workflow import (  # noqa: E402
    audit,
    diagnose,
    load_observations,
    render_report,
    write_run,
)
from domains.wind_tunnel.contract_verifiers import check_verifiers  # noqa: E402
from domains.wind_tunnel.fault_diagnosis import annotate, render as render_layers  # noqa: E402

DATA = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
LIVE = ROOT / "results" / "contract_consistency_live"
OUT = ROOT / "results" / "contract_consistency" / "error_workflow"
LAYER_OUT = ROOT / "results" / "contract_consistency" / "fault_diagnosis"
RUNS = (
    ("body-rule · model前缀", "body-rule", "self-trunc", LIVE / "body" / "model" / "records.jsonl"),
    ("body-rule · gold前缀", "body-rule", "gold-trunc", LIVE / "body" / "gold" / "records.jsonl"),
    ("roll-rule · model前缀", "roll-rule", "self-trunc", LIVE / "roll" / "model" / "records.jsonl"),
    ("roll-rule · gold前缀", "roll-rule", "gold-trunc", LIVE / "roll" / "gold" / "records.jsonl"),
)


def _specs() -> dict[str, dict]:
    wanted = {item[1] for item in RUNS} | {"standard-default"}
    found = {}
    for line in DATA.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in wanted:
            found[row["id"]] = row["spec"]
    missing = wanted - set(found)
    if missing:
        raise RuntimeError(f"样本缺少规格：{sorted(missing)}")
    return found


def main() -> None:
    specs = _specs()
    answers = {sample_id: generated_answers(spec) for sample_id, spec in specs.items()}
    standard = answers["standard-default"]
    for sample_id, spec in specs.items():
        check_verifiers(_model(spec), answers[sample_id])
        print(f"{sample_id} 验证器金标复核通过", flush=True)
    loaded = []
    for name, sample_id, condition, path in RUNS:
        observations = load_observations(path, sample_id, condition, answers[sample_id], standard)
        cases = diagnose(_model(specs[sample_id]), observations, answers[sample_id])
        loaded.append((name, sample_id, condition, observations, cases))
    by_key = {(sample_id, condition): observations for _name, sample_id, condition, observations, _cases in loaded}
    rendered = []
    layered = []
    OUT.mkdir(parents=True, exist_ok=True)
    for name, sample_id, condition, observations, cases in loaded:
        other_condition = "gold-trunc" if condition == "self-trunc" else "self-trunc"
        annotate(
            _model(specs[sample_id]),
            observations,
            cases,
            answers[sample_id],
            by_key[(sample_id, other_condition)],
        )
        summary = audit(observations, cases)
        write_run(OUT / f"{sample_id}_{condition}", observations, cases, summary)
        rendered.append((name, summary, cases))
        layered.append((name, observations, cases))
        print(f"{name} 错题 {summary['wrong']} 登记 {summary['registered']} 未解决 {summary['unresolved']}", flush=True)
    report = OUT / "report.md"
    report.write_text(render_report(rendered), encoding="utf-8")
    LAYER_OUT.mkdir(parents=True, exist_ok=True)
    layer_report = LAYER_OUT / "report.md"
    layer_report.write_text(render_layers(layered), encoding="utf-8")
    print(report, flush=True)
    print(layer_report, flush=True)


if __name__ == "__main__":
    main()
