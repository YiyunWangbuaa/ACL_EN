# -*- coding: utf-8 -*-
"""Collect statistics and representative snippets for the Word report.

Run with the project Python (needs sympy + the workspace generators):
    python report_build/stats.py
"""

from __future__ import annotations

import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloze_eval import jobs_for, load_samples  # noqa: E402
from domains.wind_tunnel.dataset_generate import PRIOR, _mechanism_phrases, _render_markdown, build_record  # noqa: E402
from body_frame_model import BodyFrameModel  # noqa: E402


def length_stats(values: list[int]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "max": max(values),
        "p25": ordered[int(len(ordered) * 0.25)],
        "p75": ordered[int(len(ordered) * 0.75)],
        "sum": sum(values),
    }


def count_by(items, key):
    out: dict[str, int] = {}
    for item in items:
        k = item[key]
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> None:
    data_path = ROOT / "data" / "wind_tunnel" / "batch" / "samples.jsonl"
    lines = [line for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [json.loads(line) for line in lines]
    with_meta = []
    for line, row in zip(lines, rows):
        with_meta.append({
            "row": row,
            "line_chars": len(line),
            "line_bytes": len(line.encode("utf-8")),
            "declaration_chars": len(row["declaration"]),
            "spec_chars": len(json.dumps(row["spec"], ensure_ascii=False)),
            "questions_chars": len(json.dumps(row["questions"], ensure_ascii=False)),
            "answers_chars": sum(len(q["answer"]) for q in row["questions"]),
        })

    all_questions = [q for row in rows for q in row["questions"]]
    label_counts = count_by(all_questions, "label")
    format_counts = count_by(all_questions, "format_code")
    process_counts = count_by(all_questions, "process")
    option_counts = {
        "有选项": sum(1 for q in all_questions if q.get("options")),
        "无选项": sum(1 for q in all_questions if not q.get("options")),
    }
    per_process_labels: dict[str, dict[str, int]] = {}
    for process in process_counts:
        per_process_labels[process] = count_by(
            [q for q in all_questions if q["process"] == process], "label"
        )
    per_tier = []
    for tier in (25, 50, 75, 100):
        subset = [row for row in rows if row["tier"] == tier]
        per_tier.append({
            "tier": tier,
            "samples": len(subset),
            "changed": sorted({row["changed_count"] for row in subset}),
            "row_chars": length_stats([len(line) for line, row in zip(lines, rows) if row["tier"] == tier]),
            "declaration_chars": length_stats([len(row["declaration"]) for row in subset]),
        })

    # Sentence-level jobs: calls per sample and prompt lengths.
    pilot_ids = json.loads((ROOT / "data" / "wind_tunnel" / "batch" / "pilot_10.json").read_text(encoding="utf-8"))["ids"]
    pilot_rows = load_samples(data_path, set(pilot_ids))
    order = {sample_id: index for index, sample_id in enumerate(pilot_ids)}
    pilot_rows.sort(key=lambda row: order[row["id"]])

    sample_jobs = []
    prompt_lengths: list[int] = []
    prompt_by_process: dict[str, list[int]] = {}
    group_sizes: list[int] = []
    multi_blanks = 0
    total_calls = 0
    total_prompt_chars = 0
    example_prompt = ""
    example_group = None
    example_sample = ""
    all_jobs_by_sample: dict[str, int] = {}
    for row in rows:
        jobs = jobs_for(row)
        all_jobs_by_sample[row["id"]] = len(jobs)
        total_calls += len(jobs)
        for job in jobs:
            prompt_lengths.append(len(job["prompt"]))
            total_prompt_chars += len(job["prompt"])
            prompt_by_process.setdefault(job["process"], []).append(len(job["prompt"]))
            group_sizes.append(len(job["blank_ids"]))
            if len(job["blank_ids"]) > 1:
                multi_blanks += 1
        sample_jobs.append({"id": row["id"], "tier": row["tier"], "calls": len(jobs)})

    # Representative prompts: one single-blank and one multi-blank sentence.
    single_prompt = ""
    single_answers: list[dict] = []
    single_sample = ""
    multi_prompt = ""
    multi_answers: list[dict] = []
    multi_sample = ""
    for row in pilot_rows:
        jobs = jobs_for(row)
        if not single_prompt:
            job = next(j for j in jobs if len(j["blank_ids"]) == 1)
            single_prompt = job["prompt"]
            single_answers = [
                {"id": b["blank_id"], "label": b["label"], "format": b["format_code"], "answer": b["answer"], "options": b.get("options") or []}
                for b in job["blanks"]
            ]
            single_sample = row["id"]
        if not multi_prompt:
            job = next((j for j in jobs if len(j["blank_ids"]) >= 3), None)
            if job is not None:
                multi_prompt = job["prompt"]
                multi_answers = [
                    {"id": b["blank_id"], "label": b["label"], "format": b["format_code"], "answer": b["answer"], "options": b.get("options") or []}
                    for b in job["blanks"]
                ]
                multi_sample = row["id"]
        if single_prompt and multi_prompt:
            break
    example_prompt = multi_prompt
    example_group = None
    example_sample = multi_sample

    # Full source record for the 100% sample 100-001.
    full_row = next(row for row in rows if row["id"] == "100-001")
    full_record = build_record({
        "id": full_row["id"],
        "title": full_row["title"],
        "change": full_row["change"],
        "spec": deepcopy(full_row["spec"]),
    })
    full_json = json.dumps(full_record, ensure_ascii=False)
    markdown = _render_markdown(full_record)
    process_lengths = {name: len(text) for name, text in full_record["processes"].items()}

    # Other corpora.
    def corpus_stats(path: Path) -> dict:
        samples = json.loads(path.read_text(encoding="utf-8"))
        questions = [q for sample in samples for q in sample["questions"]]
        return {
            "samples": len(samples),
            "questions": len(questions),
            "labels": count_by(questions, "label"),
            "declaration_chars": length_stats([len(s["declaration"]) for s in samples]),
            "process_chars": length_stats([sum(len(v) for v in s["processes"].values()) for s in samples]),
            "sample_json_chars": length_stats([len(json.dumps(s, ensure_ascii=False)) for s in samples]),
        }

    review = corpus_stats(ROOT / "data" / "wind_tunnel" / "review" / "samples.json")
    stress = corpus_stats(ROOT / "data" / "wind_tunnel" / "stress" / "samples.json")

    pilot_records = ROOT / "results" / "pilot_10" / "records.jsonl"
    pilot_result = {}
    if pilot_records.is_file():
        recs = [json.loads(line) for line in pilot_records.read_text(encoding="utf-8").splitlines() if line.strip()]
        pilot_result = {
            "records": len(recs),
            "samples": len({r["sample_id"] for r in recs}),
            "exact": sum(bool(r.get("exact")) for r in recs),
            "labels": count_by(recs, "label"),
        }

    # Theoretical definition space.
    factors = 16
    scalars = 13
    tiers = (4, 8, 12, 16)
    from math import comb

    def tiered_count(k: int) -> int:
        total = 0
        for frames in range(0, 4):
            if k - frames < 0 or k - frames > scalars:
                continue
            total += comb(3, frames) * comb(scalars, k - frames) * (47 ** frames)
        return total

    tiered = {str(k): tiered_count(k) for k in tiers}
    tiered_sum = sum(tiered.values())
    full_space = (2 ** scalars) * (48 ** 3)
    blanks_full = full_space * 193
    blanks_tiered = tiered_sum * 193
    mean_calls = total_calls / len(rows)

    stats = {
        "data_file": str(data_path),
        "data_bytes": data_path.stat().st_size,
        "rows": len(rows),
        "row_chars": length_stats([m["line_chars"] for m in with_meta]),
        "row_bytes": length_stats([m["line_bytes"] for m in with_meta]),
        "declaration_chars": length_stats([m["declaration_chars"] for m in with_meta]),
        "spec_chars": length_stats([m["spec_chars"] for m in with_meta]),
        "questions_chars": length_stats([m["questions_chars"] for m in with_meta]),
        "answers_chars": length_stats([m["answers_chars"] for m in with_meta]),
        "per_tier": per_tier,
        "all_questions": len(all_questions),
        "label_counts": label_counts,
        "format_counts": format_counts,
        "process_counts": process_counts,
        "per_process_labels": per_process_labels,
        "option_counts": option_counts,
        "sample_jobs": sample_jobs,
        "calls": {
            "total": total_calls,
            "per_sample": length_stats([j["calls"] for j in sample_jobs]),
            "mean": round(mean_calls, 1),
            "group_sizes": count_by([{"n": str(n)} for n in group_sizes], "n"),
            "multi_blanks": multi_blanks,
        },
        "prompt_chars": length_stats(prompt_lengths),
        "prompt_by_process": {k: length_stats(v) for k, v in prompt_by_process.items()},
        "prompt_total_chars": total_prompt_chars,
        "prompt_token_estimate": {
            "chars_per_token": 2.2,
            "mean_tokens": round(statistics.mean(prompt_lengths) / 2.2, 0),
            "total_tokens": round(total_prompt_chars / 2.2, 0),
        },
        "full_record": {
            "json_chars": len(full_json),
            "prior_chars": len(full_record["prior"]),
            "process_chars": process_lengths,
            "process_total_chars": sum(process_lengths.values()),
            "markdown_chars": len(markdown),
            "injection_example_chars": len(full_record["injection_example"]),
            "declaration_chars": len(full_record["declaration"]),
        },
        "review_10": review,
        "stress_10": stress,
        "pilot_result": pilot_result,
        "theory": {
            "factors": factors,
            "scalars": scalars,
            "frames": 3,
            "frames_legal": 48,
            "frames_nonstandard": 47,
            "full_space": full_space,
            "tiered": tiered,
            "tiered_sum": tiered_sum,
            "blanks_full": blanks_full,
            "blanks_tiered": blanks_tiered,
            "calls_per_definition": round(mean_calls, 1),
            "calls_full": round(full_space * mean_calls, 0),
            "calls_tiered": round(tiered_sum * mean_calls, 0),
            "conditions": 5,
        },
        "snippets": {
            "change_100": full_row["change"],
            "spec_100": json.dumps(full_row["spec"], ensure_ascii=False, indent=2),
            "declaration": full_record["declaration"],
            "prior": full_record["prior"],
            "process_tz_fd": full_record["processes"]["风轴系到体轴系"],
            "question_60": json.dumps(
                next(q for q in full_record["questions"] if q["process"] == "风轴系到体轴系" and q["id"] == "60"),
                ensure_ascii=False, indent=2,
            ),
            "example_sample": example_sample,
            "example_prompt": example_prompt,
            "example_answers": multi_answers,
            "single_sample": single_sample,
            "single_prompt": single_prompt,
            "single_answers": single_answers,
            "multi_sample": multi_sample,
            "multi_prompt": multi_prompt,
            "multi_answers": multi_answers,
            "record_example": json.dumps(
                json.loads(pilot_records.read_text(encoding="utf-8").splitlines()[0]), ensure_ascii=False
            ) if pilot_records.is_file() else "",
        },
    }
    out = ROOT / "report_build" / "stats.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": stats["rows"],
        "row_chars": stats["row_chars"],
        "declaration_chars": stats["declaration_chars"],
        "labels": stats["label_counts"],
        "calls": stats["calls"],
        "prompt_chars": stats["prompt_chars"],
        "full_record": stats["full_record"],
        "theory": stats["theory"],
        "review": {k: review[k] for k in ("samples", "questions", "declaration_chars")},
        "stress": {k: stress[k] for k in ("samples", "questions", "declaration_chars")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
