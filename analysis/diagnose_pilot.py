# -*- coding: utf-8 -*-
"""Diagnose one session-model trajectory per domain against that sample's gold."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_score import resolve_choice

PILOT = ROOT / "results" / "validate"


def _records(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("branch") == "probe":
            continue
        if row.get("condition") not in {None, "", "session-model"}:
            continue
        rows.append(row)
    return rows


def _phrase(raw: str, options: list[str]) -> str:
    usable = [item for item in options if item not in {"", "none"}]
    text = str(raw or "").strip()
    if not usable:
        return text
    _letter, body = resolve_choice(text, usable)
    return body or text


def _sheet(rows: list[dict], node_of) -> dict[str, str]:
    found = {}
    for row in rows:
        for blank in row.get("blanks") or []:
            node = node_of(row, blank)
            options = [str(item) for item in (blank.get("options") or [])]
            found[node] = _phrase(str(blank.get("prediction") or ""), options)
    return found


def _summary(report: dict, cases) -> dict:
    states = Counter(report.get("states", {}).values())
    chains = Counter(
        item.get("chain_class") or ""
        for item in report.get("chain_attributions") or []
        if item.get("chain_class")
    )
    classes = Counter(case.final_class for case in cases)
    targets = []
    for item in report.get("targets") or []:
        targets.append({
            "node": item.get("node"),
            "reading": item.get("reading"),
            "chain_class": item.get("chain_class"),
            "matches_gold": item.get("matches_gold"),
            "causal": item.get("causal_candidates") or item.get("causal"),
            "associative": item.get("associative_candidates") or item.get("associative"),
            "nearest_slots": item.get("nearest_slots") or [],
            "side_paths": item.get("side_paths") or [],
        })
    return {
        "status": report.get("status"),
        "diagnosis_cardinality": report.get("diagnosis_cardinality"),
        "repair_cardinality": report.get("repair_cardinality"),
        "truncated": report.get("truncated"),
        "states": dict(states),
        "chain_classes": dict(chains),
        "local_classes": dict(classes),
        "conflict_count": len(report.get("conflicts") or []),
        "diagnosis_count": len(report.get("minimal_diagnoses") or []),
        "targets": targets,
        "note": report.get("note"),
    }


def _write(domain: str, report: dict, cases, meta: dict) -> None:
    out = PILOT / domain
    out.mkdir(parents=True, exist_ok=True)
    (out / "conflict_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    brief = _summary(report, cases)
    brief["sample"] = meta
    (out / "diagnosis_summary.json").write_text(
        json.dumps(brief, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        f"# {meta['domain']} {meta['sample_id']} {meta['factor']}",
        "",
        f"- condition: session-model (earlier turns are the model's own answers)",
        f"- exact {meta.get('exact')} / normalized {meta.get('normalized')} / blanks {meta.get('blanks')}",
        f"- diagnosis status: {brief['status']}",
        f"- diagnosis cardinality: {brief['diagnosis_cardinality']}",
        f"- repair cardinality: {brief['repair_cardinality']}",
        f"- conflict sets {brief['conflict_count']}, minimal hitting sets {brief['diagnosis_count']}",
        "- search truncated: " + (
            "yes. The whole-sheet hitting set was not covered within cardinality 4. An empty minimal hitting set does not mean there is no fault."
            if brief["truncated"] else "no."
        ),
        "",
        "## Node states",
        "",
    ]
    for name, count in sorted(brief["states"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Answer classes", ""])
    for name, count in sorted(brief["chain_classes"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Local classes", ""])
    for name, count in sorted(brief["local_classes"].items()):
        lines.append(f"- {name}: {count}")
    lines.extend(["", "## Terminals", ""])
    for item in brief["targets"]:
        nearest = ", ".join(item.get("nearest_slots") or [])
        lines.append(
            f"- {item['node']} {item.get('reading') or ''} {item.get('chain_class') or ''} "
            f"matches gold={item.get('matches_gold')}"
            + (f" nearest wrong slots={nearest}" if nearest else "")
        )
    (out / "diagnosis.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(domain, brief["status"], "exact", meta.get("exact"), "/", meta.get("blanks"), flush=True)


def _meta(domain: str, sample_id: str, factor: str) -> dict:
    path = PILOT / domain / "meta.json"
    stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return {
        "domain": domain,
        "sample_id": sample_id,
        "factor": factor,
        "condition": stored.get("condition"),
        "model": stored.get("model"),
        "thinking": stored.get("thinking"),
        "exact": stored.get("exact"),
        "normalized": stored.get("normalized"),
        "blanks": stored.get("blanks"),
    }


def _cases(view, answers):
    from domains.observation import diagnose

    return diagnose(view, answers, prefix="model")


def wind() -> None:
    from body_frame_model import BodyFrameModel
    from domains.wind_tunnel.conflict_diagnosis import explain_observations
    from domains.wind_tunnel.contract_executor import generated_answers
    from domains.wind_tunnel.error_workflow import diagnose as local_diagnose
    from domains.wind_tunnel.error_workflow import load_observations

    rows = []
    for line in (ROOT / "data" / "wind_tunnel" / "single" / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    by_id = {row["id"]: row for row in rows}
    sample = by_id["S01"]
    from domains.wind_tunnel import slot_signature

    slot_signature._rows()[sample["id"]] = sample
    standard = generated_answers(by_id["000-001"]["spec"])
    gold = generated_answers(sample["spec"])
    model = BodyFrameModel().declare(sample["spec"])
    observations = load_observations(
        PILOT / "wind_tunnel" / "records.jsonl",
        "S01",
        "session-model",
        gold,
        standard,
    )
    report = explain_observations(model, observations, gold, baseline=standard)
    cases = local_diagnose(model, observations, gold)
    _write("wind_tunnel", report, cases, _meta("wind_tunnel", "S01", "theoretical angle of attack alpha"))


def _retarget_flight() -> None:
    import domains.aerodynamics.manuscript as manuscript

    manuscript.SAMPLE_ID = "02"
    manuscript.record.cache_clear()
    manuscript.questions.cache_clear()
    manuscript.gold_texts.cache_clear()
    manuscript.gold_scored.cache_clear()


def flight() -> None:
    _retarget_flight()
    from domains.aerodynamics.aligned import view
    from domains.aerodynamics.fault_diagnosis import _baseline_gold
    from domains.aerodynamics.signed_lists import T_FINAL
    from domains.conflict_diagnosis import explain

    built = view()
    answers = _sheet(_records(PILOT / "flight" / "records.jsonl"), lambda _row, blank: str(blank["blank_id"]))
    report = explain(built, answers, T_FINAL, prefix="model", baseline=_baseline_gold())
    _write("flight", report, _cases(built, answers), _meta("flight", "02", "yaw"))


def _retarget_scene() -> None:
    import domains.computer_vision.scene_manuscript as manuscript

    manuscript.REVIEW = ROOT / "data" / "vision" / "single"
    manuscript.SAMPLE_ID = "02"
    manuscript.questions.cache_clear()
    manuscript.gold_texts.cache_clear()
    manuscript.gold_scored.cache_clear()


def vision() -> None:
    _retarget_scene()
    from domains.computer_vision.scene_aligned import view
    from domains.computer_vision.scene_fault_diagnosis import _baseline_gold
    from domains.computer_vision.scene_signed_lists import T_FINAL
    from domains.conflict_diagnosis import explain

    built = view()
    answers = _sheet(_records(PILOT / "cv" / "records.jsonl"), lambda _row, blank: str(blank["blank_id"]))
    report = explain(built, answers, T_FINAL, prefix="model", baseline=_baseline_gold())
    _write("cv", report, _cases(built, answers), _meta("cv", "02", "world frame"))


def _retarget_kin() -> None:
    import domains.robotics.kin_manuscript as manuscript

    manuscript.REVIEW = ROOT / "data" / "robotics" / "single"
    manuscript.SAMPLE_ID = "02"
    manuscript.questions.cache_clear()
    manuscript.gold_texts.cache_clear()
    manuscript.gold_scored.cache_clear()


def robotics() -> None:
    _retarget_kin()
    from domains.conflict_diagnosis import explain
    from domains.robotics.kin_aligned import view
    from domains.robotics.kin_fault_diagnosis import _baseline_gold
    from domains.robotics.kin_signed_lists import T_FINAL

    def node_of(row, blank):
        process = str(row.get("process") or "")
        number = str(blank["blank_id"])
        if process.startswith("inverse kinematics"):
            return f"I{number}"
        return f"F{number}"

    built = view()
    answers = _sheet(_records(PILOT / "robotics" / "records.jsonl"), node_of)
    report = explain(built, answers, T_FINAL, prefix="model", baseline=_baseline_gold())
    _write("robotics", report, _cases(built, answers), _meta("robotics", "02", "joint angle q1"))


def main() -> None:
    global PILOT
    if len(sys.argv) > 1:
        PILOT = Path(sys.argv[1]).resolve()
    wind()
    flight()
    vision()
    robotics()


if __name__ == "__main__":
    main()
