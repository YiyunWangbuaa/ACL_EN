# -*- coding: utf-8 -*-
"""Score cloze sentences with DeepSeek.

The main condition is deepseek-v4-pro with thinking enabled. deepseek-flash
and thinking disabled are controls, not the V4 result. Thinking mode ignores
temperature. The official V4 API also ignores seed, so repeated runs are the
variance control. max_tokens includes the chain of thought. The floor is
2048 and the default is 8192 when thinking is off. An unspecified thinking
run sends 65536, twice the previous 32768 ceiling. reasoning_content is stored
for audit and is never written into the next prompt.

When thinking uses up max_tokens the reply ends with finish_reason=length.
That reply is not retried and is scored as no answer. Each sentence starts
from a fresh single user message, so nothing from a previous reply is re-sent.

One call answers one sentence, except proc-e2e (one subprocess) and final-e2e
(one final question, no cloze chain). gold-trunc fills earlier blanks with gold
and stops at the period. self-trunc uses the model's own raw earlier outputs.
gold-keep keeps the gold prefix and the rest of the subprocess. self-keep uses
the model's raw prefix and keeps the rest. self-trunc and self-keep run one
sentence at a time inside each subprocess. Old letters A–F still select these
six names. The default data is each domain's full corpus. Wind tunnel reads
the batch jsonl. Flight, vision and robotics read their review manuscripts.
Strict equality is the stripped reply. The normalized column also accepts
spacing, symbol and LaTeX differences, an exact option after that cleanup,
and sympy equivalence for expressions and matrices.

Run one domain's full set. --inventory only counts, and does not call the model.

    python cloze_eval.py --domain wind_tunnel --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/wind_tunnel
    python cloze_eval.py --domain flight --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/flight
    python cloze_eval.py --domain cv --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/cv
    python cloze_eval.py --domain robotics --condition session-model --model deepseek-flash --thinking disabled --runs 1 --out results/validate/robotics

Run the formal condition::

    python cloze_eval.py --domain wind_tunnel --model deepseek-v4-pro --thinking enabled --runs 3 --out results/pro_thinking
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from body_frame_model import BodyFrameModel
from domain_layout import EVAL, EVAL_DOMAINS, TIERS
from domains.wind_tunnel.dataset_generate import _injection, _injection_end_to_end, _local_id, _mechanism_phrases, _sentence_groups
from semi_auto_eval import call_model, load_env_file, make_client, resolve_api_config

ROOT = Path(__file__).resolve().parent
SHANGHAI = timezone(timedelta(hours=8))


_MATRIX_CODE = {"matrix"}
_CHOICE_CODE = {"choice", "none"}
_END_TO_END_LABEL = "end-to-end"

# Condition names state the injection. Old letters still resolve to these names.
#   gold-trunc  gold prefix, truncated at the current sentence          (was A)
#   self-trunc  own prefix,  truncated                                  (was B)
#   gold-keep   gold prefix, later text kept                            (was C)
#   self-keep   own prefix,  later text kept                            (was D)
#   proc-e2e    one call per subprocess, every blank open               (was E)
#   final-e2e   one final question, no step-by-step cloze               (was F)
#   session-gold   one chat; definition once; earlier turns are gold
#   session-model  one chat; definition once; earlier turns stay as answered
GOLD_TRUNC = "gold-trunc"
SELF_TRUNC = "self-trunc"
GOLD_KEEP = "gold-keep"
SELF_KEEP = "self-keep"
PROC_E2E = "proc-e2e"
FINAL_E2E = "final-e2e"
SESSION_GOLD = "session-gold"
SESSION_MODEL = "session-model"
CONDITIONS = (
    GOLD_TRUNC, SELF_TRUNC, GOLD_KEEP, SELF_KEEP, PROC_E2E, FINAL_E2E,
    SESSION_GOLD, SESSION_MODEL,
)
_SEQUENTIAL = {SELF_TRUNC, SELF_KEEP}
_SESSION = {SESSION_GOLD, SESSION_MODEL}
_KEEP_SUFFIX = {GOLD_KEEP, SELF_KEEP}
_END_TO_END = {PROC_E2E, FINAL_E2E}
CONDITION_GLOSS = {
    GOLD_TRUNC: "gold prefix, truncated at the period",
    SELF_TRUNC: "own prefix, truncated at the period",
    GOLD_KEEP: "gold prefix, later text kept",
    SELF_KEEP: "own prefix, later text kept",
    PROC_E2E: "one call per subprocess, every blank open",
    FINAL_E2E: "final question only, no step-by-step cloze",
    SESSION_GOLD: "definition once, one chat, earlier turns use gold",
    SESSION_MODEL: "definition once, one chat, earlier turns use the model's answers",
}
# Old spellings still accepted so that archived commands keep working.
# Note: the old code "D" meant end to end, which is now proc-e2e.
CONDITION_ALIASES = {
    "A": GOLD_TRUNC, "GOLD-TRUNC": GOLD_TRUNC, "GOLD_TRUNC": GOLD_TRUNC,
    "B": SELF_TRUNC, "SELF-TRUNC": SELF_TRUNC, "SELF_TRUNC": SELF_TRUNC,
    "C": GOLD_KEEP, "C'": GOLD_KEEP, "C′": GOLD_KEEP, "CP": GOLD_KEEP, "CPRIME": GOLD_KEEP,
    "GOLD-KEEP": GOLD_KEEP, "GOLD_KEEP": GOLD_KEEP,
    "D": SELF_KEEP, "C''": SELF_KEEP, "C″": SELF_KEEP, "CPP": SELF_KEEP, "CDPRIME": SELF_KEEP,
    "SELF-KEEP": SELF_KEEP, "SELF_KEEP": SELF_KEEP,
    "E": PROC_E2E, "END": PROC_E2E, "E2E": PROC_E2E, "PROC-E2E": PROC_E2E, "PROC_E2E": PROC_E2E,
    "F": FINAL_E2E, "FINAL": FINAL_E2E, "FINAL-E2E": FINAL_E2E, "FINAL_E2E": FINAL_E2E,
    "SESSION-GOLD": SESSION_GOLD,
    "SESSION-MODEL": SESSION_MODEL,
}


def canonical_condition(raw: str) -> str:
    key = (raw or "").strip().upper().replace("′", "'").replace("″", "''").replace("_", "-")
    return CONDITION_ALIASES.get(key, "")


def condition_label(code: str) -> str:
    resolved = canonical_condition(code) or code
    gloss = CONDITION_GLOSS.get(resolved)
    return f"{resolved} ({gloss})" if gloss else str(code)

# DeepSeek prices per 1M tokens, peak then off-peak.
# https://api-docs.deepseek.com/quick_start/pricing
# Peak is 09:00-12:00 and 14:00-18:00 Beijing time, Monday-Friday, excluding
# the 2026 Chinese public-holiday dates below. Weekends are off-peak.
_FLASH_NAMES = {"deepseek-flash", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"}
_PRO_NAMES = {"deepseek-v4-pro"}
_RATES = {
    "deepseek-flash": {"peak": (0.006, 0.3, 1.2), "off_peak": (0.003, 0.15, 0.6)},
    "deepseek-v4-pro": {"peak": (0.044, 1.32, 3.96), "off_peak": (0.022, 0.66, 1.98)},
}
_DOCUMENTED_VERSION = {
    "deepseek-flash": "DeepSeek-V4.1-Flash-1017",
    "deepseek-v4-flash": "DeepSeek-V4.1-Flash-1017",
    "deepseek-v4-flash-vision-exp": "DeepSeek-V4.1-Flash-1017",
    "deepseek-v4-pro": "DeepSeek-V4-Pro-0813",
}
# 2026 public-holiday dates from the State Council. Used only to tell DeepSeek peak from off-peak, not as a substitute for the official bill.
_HOLIDAYS_2026 = {
    date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3),
    *{date(2026, 2, day) for day in range(15, 24)},
    date(2026, 4, 4), date(2026, 4, 5), date(2026, 4, 6),
    *{date(2026, 5, day) for day in range(1, 6)},
    date(2026, 6, 19), date(2026, 6, 20), date(2026, 6, 21),
    date(2026, 9, 25), date(2026, 9, 26), date(2026, 9, 27),
    *{date(2026, 10, day) for day in range(1, 8)},
}


def _dependency_fields(item: dict) -> dict:
    """Graph fields for one wind-flight blank. Other domains contribute nothing."""
    from domains.wind_tunnel.error_graph import dependency_fields

    return dependency_fields(item)


def load_samples(path: Path, wanted: set[str] | None, strict_graph: bool = True) -> list[dict]:
    from domains.wind_tunnel.error_graph import annotate_question, annotate_sample

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if wanted is not None and row["id"] not in wanted:
                continue
            if strict_graph:
                rows.append(annotate_sample(row))
                continue
            skipped = 0
            for question in row.get("questions") or []:
                try:
                    annotate_question(question)
                except RuntimeError:
                    skipped += 1
            row["graph_label_mismatches"] = skipped
            rows.append(row)
    if wanted is not None:
        found = {row["id"] for row in rows}
        missing = wanted - found
        if missing:
            raise SystemExit("These ids are not in the samples: " + ", ".join(sorted(missing)))
    return rows


def contexts_for(row: dict) -> list[dict]:
    model = BodyFrameModel().declare(deepcopy(row["spec"]))
    phrases = _mechanism_phrases(model)
    custom = row.get("process_text") or {}
    grouped: dict[str, dict] = {}
    processes: list[str] = []
    for item in row["questions"]:
        if item["process"] not in grouped:
            processes.append(item["process"])
        if item["process"] in custom:
            slot_id = str(item["id"])
        else:
            # Batch samples store consecutive public ids. Template slots still use the manuscript ids inside each subprocess.
            slot_id = str(item.get("local_id") or _local_id(item["process"], str(item["id"])))
        grouped.setdefault(item["process"], {})[slot_id] = item
    chains = []
    for process in processes:
        by_id = grouped[process]
        template = custom.get(process)
        sentences = []
        for sentence in _sentence_groups(process, phrases, template):
            blanks = []
            public_ids = []
            for local_id in sentence:
                item = by_id[local_id]
                public_ids.append(item["id"])
                blanks.append({
                    "blank_id": item["id"],
                    "label": item["label"],
                    "format_code": item["format_code"],
                    "answer": item["answer"],
                    "options": list(item.get("options") or []),
                    **_dependency_fields(item),
                })
            sentences.append({"blank_ids": public_ids, "blanks": blanks})
        chains.append({
            "sample_id": row["id"],
            "tier": row.get("tier", 0),
            "process": process,
            "declaration": row["declaration"],
            "phrases": phrases,
            "by_id": by_id,
            "sentences": sentences,
            "process_text": template,
        })
    return chains


def end_to_end_job(
    chain: dict,
    run: int,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
) -> dict:
    blank_ids = []
    blanks = []
    for sentence in chain["sentences"]:
        blank_ids.extend(sentence["blank_ids"])
        blanks.extend(sentence["blanks"])
    return {
        "sample_id": chain["sample_id"],
        "tier": chain["tier"],
        "process": chain["process"],
        "blank_ids": blank_ids,
        "blanks": blanks,
        "condition": PROC_E2E,
        "run": run,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": stop,
        "standard": standard,
        "prompt": (
            _manuscript_end_to_end(chain)
            if chain.get("domain") == "manuscript"
            else _cv_end_to_end(chain)
            if chain.get("domain") == "cv"
            else _flight_end_to_end(chain)
            if chain.get("domain") == "flight"
            else _injection_end_to_end(
                chain["declaration"],
                chain["process"],
                chain["by_id"],
                chain["phrases"],
                chain.get("process_text"),
            )
        ),
    }


def jobs_for(row: dict) -> list[dict]:
    """Condition A jobs. B and D cannot be built ahead of the model's replies."""
    jobs = []
    for chain in contexts_for(row):
        for sentence in chain["sentences"]:
            jobs.append(job_for(chain, sentence, GOLD_TRUNC, None, 0, 0, 0, None, {}))
    return jobs


def _cv_end_to_end(chain: dict) -> str:
    return _cv_module(chain.get("process"))._cv_end_to_end(chain)


def _cv_module(process: str | None = None):
    from domains.computer_vision import cv_basic_dataset, cv_ctrl_dataset, cv_dataset

    if process == "world to camera":
        return cv_ctrl_dataset
    if process == "vision basics":
        return cv_basic_dataset
    return cv_dataset


def contexts_for_cv(row: dict) -> list[dict]:
    """One chain for the whole computer-vision subprocess (main or control)."""
    process = (row["questions"][0].get("process") if row.get("questions") else None) or ""
    module = _cv_module(process)
    by_id = {str(item["local_id"]): item for item in row["questions"]}
    sentences = []
    for sentence in module._sentence_groups():
        blanks = []
        public_ids = []
        for local_id in sentence:
            item = by_id[local_id]
            public_ids.append(item["id"])
            blanks.append({
                "blank_id": item["id"],
                "label": item["label"],
                "format_code": item["format_code"],
                "answer": item["answer"],
                "options": list(item.get("options") or []),
                **_dependency_fields(item),
            })
        sentences.append({
            "blank_ids": public_ids,
            "local_ids": list(sentence),
            "blanks": blanks,
        })
    process_name = process if process == "pure stereo pose" else module.PROCESS
    return [{
        "sample_id": row["id"],
        "tier": row.get("tier", 0),
        "process": process_name,
        "domain": "cv",
        "group": row.get("group", "main"),
        "declaration": row["declaration"],
        "by_id": by_id,
        "sentences": sentences,
    }]


def _flight_modules() -> dict:
    from domains.aerodynamics import body_air_dataset, dataset_generate, ground_track_air_dataset
    from domains.flight_ctrl import fm_ctrl_dataset

    return {
        "ground to body": dataset_generate,
        "ground to track and airflow": ground_track_air_dataset,
        "body to airflow": body_air_dataset,
        "body relative to ground": fm_ctrl_dataset,
        "airflow to body": fm_ctrl_dataset,
    }


def contexts_for_flight(row: dict) -> list[dict]:
    """One chain per flight subprocess. Slots stay on the local manuscript id."""
    modules = _flight_modules()
    grouped: dict[str, dict] = {}
    processes: list[str] = []
    for item in row["questions"]:
        if item["process"] not in modules:
            raise KeyError(f"This flight-mechanics sample has no such subprocess: {item['process']}")
        if item["process"] not in grouped:
            processes.append(item["process"])
        grouped.setdefault(item["process"], {})[str(item["local_id"])] = item
    chains = []
    for process in processes:
        module = modules[process]
        by_id = grouped[process]
        sentences = []
        for sentence in module._sentence_groups():
            blanks = []
            public_ids = []
            for local_id in sentence:
                item = by_id[local_id]
                public_ids.append(item["id"])
                blanks.append({
                    "blank_id": item["id"],
                    "label": item["label"],
                    "format_code": item["format_code"],
                    "answer": item["answer"],
                    "options": list(item.get("options") or []),
                    **_dependency_fields(item),
                })
            sentences.append({
                "blank_ids": public_ids,
                "local_ids": list(sentence),
                "blanks": blanks,
            })
        chains.append({
            "sample_id": row["id"],
            "tier": row["tier"],
            "process": process,
            "domain": "flight",
            "group": row.get("group", "main"),
            "declaration": row["declaration"],
            "by_id": by_id,
            "sentences": sentences,
        })
    return chains


def _flight_end_to_end(chain: dict) -> str:
    module = _flight_modules()[chain["process"]]
    by_id = chain["by_id"]
    local_ids = [blank_id for group in module._sentence_groups() for blank_id in group]
    items = [by_id[blank_id] for blank_id in local_ids]
    control = chain["process"] in {"body relative to ground", "airflow to body"}
    if control:
        lead = (
            "This is an end-to-end question. Fill every open slot in this subprocess, and call the model only once. "
            "Output one JSON object. The keys are the blank ids and the values are the answers. "
            "The object must contain every id in this subprocess. Do not split it into several objects and do not answer sentence by sentence. "
            "A choice value is a letter, with no quotation marks. Formulas, matrices, vectors, expressions, and distances are strings inside English double quotes. "
            "Do not output the original sentence and do not explain."
        )
        output = "\n".join([
            "[Output]",
            "Put every answer in the same JSON object. The keys are the blank ids and the values are the answers.",
            'For example {1:A, 2:C, 11:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}. The example is not the answer to this item.',
            "A choice value is only the letter, with no quotation marks. Formulas, matrices, vectors, expressions, and distances must be strings inside English double quotes.",
            "Output only that one JSON line. Do not explain, and do not use Markdown or LaTeX.",
        ])
    else:
        lead = (
            "This is an end-to-end question. Fill every open slot in this subprocess, and call the model only once. "
            "Output one JSON object. The keys are the blank ids and the values are the answers. "
            "The object must contain every id in this subprocess. Do not split it into several objects and do not answer sentence by sentence. "
            "A choice value is a letter, with no quotation marks. A matrix value is a string inside English double quotes. "
            "Do not output the original sentence and do not explain."
        )
        output = "\n".join([
            "[Output]",
            "Put every answer in the same JSON object. The keys are the blank ids and the values are the answers.",
            "A choice value is only the letter, with no quotation marks. A matrix must be a string inside English double quotes.",
            f"Output one line {module._json_object(items)}.",
            f"Format example (not the answer to this item): {module._json_object(items, example=True)}",
        ])
    return "\n".join([
        lead,
        "",
        "[Declaration]",
        chain["declaration"].strip(),
        "",
        "[Background]",
        module.PRIOR,
        "",
        "[Subprocess]",
        chain["process"],
        "",
        "[Process]",
        module._render_process(by_id, targets=local_ids, keep_suffix=True),
        "",
        output,
    ])


# Manuscript files hold the end-to-end questions. Gold matrices come from the
# generator's local ids, which drifted from a few handwritten cloze numbers:
# ground-to-body SθψΦ is local 28, and body-to-air Sαβ is local 35.
_FINAL_SOURCES = (
    (ROOT / "doc" / "ground to body.md", "ground to body", {"1": "28"}),
    (ROOT / "doc" / "ground to track and airflow.md", "ground to track and airflow", {"1": "27", "2": "38", "3": "41"}),
    (ROOT / "doc" / "body to airflow.md", "body to airflow", {"1": "21", "2": "32", "3": "35"}),
)
_FINAL_MARK = re.compile(
    r"end-to-end question\s*:\s*(.+?)\.\s*[^{]{0,80}\{DID\s*:\s*(\d+)",
    re.S | re.I,
)


def load_final_questions() -> list[dict]:
    """Read the end-to-end questions the manuscripts added after each final matrix."""
    found = []
    for path, process, gold in _FINAL_SOURCES:
        if not path.is_file():
            raise SystemExit(f"End-to-end question file not found: {path}")
        matches = _FINAL_MARK.findall(path.read_text(encoding="utf-8"))
        dids = [did for _question, did in matches]
        if set(dids) != set(gold) or len(dids) != len(gold):
            raise SystemExit(
                f"{path.name} end-to-end items should be DID {sorted(gold)}, but the file has {dids or 'nothing'}."
            )
        for question, did in matches:
            found.append({
                "process": process,
                "did": did,
                "local_id": gold[did],
                "question": " ".join(question.split()) + ".",
            })
    return found


# λ and ω are not angles in the flight-mechanics declaration. All nine cells use the same factor so that 0, 1, and a lone trig function do not reveal the shape of the rotation matrix.
_MATRIX_EXAMPLE_CELL = "sin(λ)*cos(ω)"
_MATRIX_EXAMPLE_BODY = "[[" + "], [".join(
    ", ".join(_MATRIX_EXAMPLE_CELL for _cell in range(3)) for _row in range(3)
) + "]]"
_MATRIX_FORMAT_RULES = (
    "Write the matrix as one 3×3 line, in the form [[a, b, c], [d, e, f], [g, h, i]]. "
    "Write * between factors, and also between a number and a function. "
    "sin, cos, and tan must have parentheses, and the parentheses must use the Greek letters from the declaration, as in cos(λ). "
    "The angle symbols must match the Greek letters in the declaration. "
    "Wrap the whole matrix in English double quotes inside the JSON, and output only that line. Do not use LaTeX."
)


def _flight_final_prompt(declaration: str, question: str, public_id: str) -> str:
    if any(char in _MATRIX_EXAMPLE_BODY for char in "θψΦφμγαβ"):
        raise RuntimeError("The format example used an angle from this item.")
    asked = f'{{{public_id}:"answer"}}'
    example = f'{{{public_id}:"{_MATRIX_EXAMPLE_BODY}"}}'
    return "\n".join([
        "This call asks only for the final result and does not give a step-by-step cloze. Derive it from the declaration yourself, and leave the derivation in the chain of thought.",
        f"The final answer is one JSON object {asked}.",
        _MATRIX_FORMAT_RULES,
        "",
        "[Declaration]",
        declaration.strip(),
        "",
        "[Question]",
        question.strip(),
        "",
        "[Output]",
        _MATRIX_FORMAT_RULES,
        "The λ and ω in the example only show how the characters are arranged. They are not this item's angles, so do not write them in the answer.",
        f"Output only one line {asked}.",
        f"Format example (not the answer to this item): {example}",
    ])


def final_question_jobs(
    row: dict,
    questions: list[dict],
    run: int,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
) -> list[dict]:
    indexed = {
        (item["process"], str(item["local_id"])): item
        for item in row["questions"]
    }
    jobs = []
    for spec in questions:
        item = indexed.get((spec["process"], spec["local_id"]))
        answer = "" if item is None else str(item.get("answer") or "")
        if item is None or item.get("format_code") not in _MATRIX_CODE or not answer.startswith("[["):
            raise SystemExit(
                f"Sample {row['id']} subprocess {spec['process']} local id {spec['local_id']} is not the final matrix."
            )
        public_id = str(item["id"])
        jobs.append({
            "sample_id": row["id"],
            "tier": row["tier"],
            "process": spec["process"],
            "blank_ids": [public_id],
            "blanks": [{
                "blank_id": public_id,
                "label": _END_TO_END_LABEL,
                "format_code": "matrix",
                "answer": item["answer"],
                "options": [],
                **_dependency_fields(item),
            }],
            "condition": FINAL_E2E,
            "run": run,
            "seed": seed,
            "max_tokens": max_tokens,
            "stop": stop,
            "standard": standard,
            "prompt": _flight_final_prompt(row["declaration"], spec["question"], public_id),
        })
    return jobs


# The example letters do not appear in the wind-tunnel answers. Every blank in one item uses the same example, so the example does not reveal which term carries the minus sign.
_WIND_FORMAT_EXAMPLE = {
    "expression": "sin(λ)*cos(ω)",
    "arctan": "-arctan((sin(λ))/(cos(ω)))",
    "arcsin": "-arcsin(sin(λ))",
    "moment-arm": "-P·(Q+R)/L_RT",
}
_WIND_FORMAT_RULE = {
    "expression": (
        "Write the expression on one line. Multiplication must be written as *. "
        "sin, cos, and tan must have parentheses, and the parentheses must use this item's angle symbols. "
        "Write a minus sign directly in front of the term it belongs to. Do not use LaTeX and do not write an equals sign."
    ),
    "arctan": (
        "Write arctan on one line, and use only one arctan. The numerator and the denominator must both be positive. "
        "Put the sign of the angle outside the whole arctan: a positive angle is arctan((numerator)/(denominator)), and a negative angle is -arctan((numerator)/(denominator)). "
        "Do not put the minus sign inside the numerator or the denominator. Do not use LaTeX."
    ),
    "arcsin": (
        "Write arcsin on one line, and use only one arcsin. The numerator must be positive. "
        "Put the sign of the angle outside the whole arcsin: a positive angle is arcsin(numerator), and a negative angle is -arcsin(numerator). "
        "Do not write it as a fraction, and do not put the minus sign inside the numerator. Do not use LaTeX."
    ),
    "moment-arm": (
        "Write the moment arm on one line, in the form force-coefficient·arm/L_RT. Use a middle dot · between the force coefficient and the arm. "
        "Put the minus sign outside the whole expression. The arm itself must be positive. "
        "The force coefficient must be one of CA, CZ, and CN. The arm is built only from Xg, Yg, Zg, and Xr; add parentheses when it contains a subtraction. "
        "Keep L_RT as the denominator. Do not leave the letters C or L_c, do not write an equals sign, and do not use LaTeX."
    ),
}
_WIND_FINALS = (
    (
        "wind frame to body-axis frame",
        (
            ("1", "77", "α_t"),
            ("2", "78", "β_t"),
        ),
        "Initially the body axes are level. The incoming flow lies on the wind-axis axial direction and blows toward the nose. "
        "After the mechanism changes α_M, β_M, and γ_M, the body is in a new attitude. "
        "From the known α_M, β_M, and γ_M, derive the expressions for the theoretical airflow angles α_t and β_t in that new attitude. Ignore balance deformation and airflow compensation. "
        "Use this formula template: the template for the theoretical angle of attack α_t is [sign][inverse trig function]([numerator]/[denominator]), "
        "and the template for β_t is [sign][inverse trig function]([numerator]/[denominator]). "
        "The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle. Judge the sign separately, and write it only outside the inverse trig function. "
        "Do not put the sign inside the inverse trig function in place of the outer sign. "
        "The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle. Judge the sign separately, and write it only outside the inverse trig function. "
        "Do not put the sign inside the inverse trig function in place of the outer sign. "
        "Find the theoretical angle of attack α_t and the theoretical sideslip β_t.",
    ),
    (
        "balance frame to body-axis frame",
        (
            ("1", "27", "CN_TZ"),
            ("2", "28", "CA_TZ"),
            ("3", "29", "CZ_TZ"),
            ("4", "30", "MN_TZ"),
            ("5", "31", "MA_TZ"),
            ("6", "32", "MZ_TZ"),
        ),
        "The six components measured in the balance frame are CN_TP, MN_TP, CA_TP, MA_TP, CZ_TP, and MZ_TP. "
        "From the current declaration and these balance measurements, derive CN_TZ, CA_TZ, CZ_TZ, MN_TZ, MA_TZ, and MZ_TZ in the body-axis frame.",
    ),
    (
        "balance frame to wind frame",
        (
            ("1", "60", "Δα_e"),
            ("2", "61", "Δβ_e"),
        ),
        "Δα_e is the angle between the wind-tunnel axis and the projection of the balance axis, after balance deformation, onto the vertical plane through the wind-tunnel axis. "
        "Δβ_e is the angle between the balance axis after deformation and the vertical plane through the wind-tunnel axis. "
        "When the balance is level it coincides with the wind-tunnel axis. When a balance installation angle is present and the balance then deforms, "
        "what is the relation between Δα_e, Δβ_e, and the balance elastic angles? "
        "Simplify with sinθ=θ and cosθ=1, and use the elastic angles in radians directly as that approximation. "
        "Give Δα_e and Δβ_e.",
    ),
    (
        "moment reference point",
        (
            ("1", "15", "MCN_CA"),
            ("2", "16", "MCZ_CA"),
            ("3", "17", "MCA_CZ"),
            ("4", "18", "MCN_CZ"),
            ("5", "19", "MCA_CN"),
            ("6", "20", "MCZ_CN"),
        ),
        "After the balance frame is transformed into the body-axis frame, the body-axis origin is at the balance calibration center. Engineering practice gives a moment reference point, and the body-axis moments must be transferred to that moment reference point. "
        "The distance from the balance calibration center to the model nose is Xr. It is computed indirectly from the measured model length L, the distance L_t from the model base to the sting base, and the distance L_r from the balance calibration center to the sting base. "
        "The engineering coordinates of the moment reference point are [Xg, Yg, Zg]. The axial component is the distance from the reference point to the model nose, and the engineering positive direction points toward the rear of the model. "
        "The positive directions of the other two components follow the positive directions of the corresponding body-axis coordinates. "
        "MCA_g=MCA+[MCN_CA]+[MCZ_CA], MCZ_g=MCZ+[MCA_CZ]+[MCN_CZ], and MCN_g=MCN+[MCA_CN]+[MCZ_CN]. "
        "Write each equivalent moment coefficient only as a force coefficient times an arm, divided by the fixed reference length L_RT. "
        "Give MCN_CA, MCZ_CA, MCA_CZ, MCN_CZ, MCA_CN, and MCZ_CN.",
    ),
)


def _wind_final_prompt(declaration: str, question: str, blanks: list[dict]) -> str:
    for blank in blanks:
        example = _WIND_FORMAT_EXAMPLE[blank["format_code"]]
        if any(char in example for char in "αβγθφψμα"):
            raise RuntimeError("The wind-tunnel format example used a symbol from this item.")
    rules = []
    seen = set()
    for blank in blanks:
        code = blank["format_code"]
        if code in seen:
            continue
        seen.add(code)
        rules.append(_WIND_FORMAT_RULE[code] + f" Example (not the answer to this item): {_WIND_FORMAT_EXAMPLE[code]}.")
    asked = ", ".join(
        f'{blank["symbol"]} goes in key {blank["blank_id"]}' for blank in blanks
    )
    fields = ", ".join(
        f'{blank["blank_id"]}:"{_WIND_FORMAT_EXAMPLE[blank["format_code"]]}"'
        for blank in blanks
    )
    example_object = "{" + fields + "}"
    answer_object = "{" + ", ".join(f'{blank["blank_id"]}:"answer"' for blank in blanks) + "}"
    return "\n".join([
        "This call asks only for the final result and does not give a step-by-step cloze. Derive it from the declaration yourself, and leave the derivation in the chain of thought.",
        f"The final answer is one JSON object {answer_object}.",
        " ".join(rules),
        "The λ, ω, P, Q, and R in the example only show how the characters are arranged. They are not quantities in this item, so do not write them in the answer.",
        "",
        "[Declaration]",
        declaration.strip(),
        "",
        "[Question]",
        question.strip(),
        "",
        "[Output]",
        " ".join(rules),
        asked + ".",
        "Write one value for each blank id. Do not merge several blanks into one expression that has no blank id. Do not use LaTeX.",
        "The λ, ω, P, Q, and R in the example only show how the characters are arranged. They are not quantities in this item, so do not write them in the answer.",
        f"Output only one line {answer_object}.",
        f"Format example (not the answer to this item): {example_object}",
    ])


_CTRL_FINALS = (
    (
        "body relative to ground",
        (
            ("1", "77", "α"),
            ("2", "78", "β"),
        ),
        "Initially the body axes are level. The incoming flow lies on the ground-frame axial direction and blows toward the nose. "
        "After the mechanism changes α_M, β_M, and γ_M, the body is in a new attitude. "
        "From the known α_M, β_M, and γ_M, derive the expressions for the theoretical airflow angles α and β in that new attitude. "
        "Use this formula template: the template for the theoretical angle of attack α is [sign][inverse trig function]([numerator]/[denominator]), "
        "and the template for β is [sign][inverse trig function]([numerator]/[denominator]). "
        "The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle. Judge the sign separately, and write it only outside the inverse trig function. "
        "Do not put the sign inside the inverse trig function in place of the outer sign. "
        "The numerator and the denominator must both be positive. That positive value is only the absolute size of the angle. Judge the sign separately, and write it only outside the inverse trig function. "
        "Do not put the sign inside the inverse trig function in place of the outer sign. "
        "Find the theoretical angle of attack α and the theoretical sideslip β.",
    ),
)


def wind_final_jobs(
    row: dict,
    run: int,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
    finals: tuple = _WIND_FINALS,
) -> list[dict]:
    indexed = {
        (item["process"], str(item["local_id"])): item
        for item in row["questions"]
    }
    jobs = []
    for process, slots, question in finals:
        blanks = []
        for _did, local_id, symbol in slots:
            item = indexed.get((process, local_id))
            answer = "" if item is None else str(item.get("answer") or "")
            if item is None or item.get("format_code") not in _WIND_FORMAT_EXAMPLE or not answer:
                raise SystemExit(
                    f"Sample {row['id']} subprocess {process} local id {local_id} is not an end-to-end answer."
                )
            blanks.append({
                "blank_id": str(item["id"]),
                "label": _END_TO_END_LABEL,
                "format_code": item["format_code"],
                "answer": item["answer"],
                "options": [],
                "symbol": symbol,
            })
        jobs.append({
            "sample_id": row["id"],
            "tier": row["tier"],
            "process": process,
            "blank_ids": [blank["blank_id"] for blank in blanks],
            "blanks": blanks,
            "condition": FINAL_E2E,
            "run": run,
            "seed": seed,
            "max_tokens": max_tokens,
            "stop": stop,
            "standard": standard,
            "prompt": _wind_final_prompt(row["declaration"], question, blanks),
        })
    return jobs


# The example uses only λ and ω, so it does not match the angles, focal lengths, or optical-center coordinates in the computer-vision answers.
_CV_FORMAT_EXAMPLE = {
    "matrix": _MATRIX_EXAMPLE_BODY,
    "vector": "[sin(λ), cos(ω), 1]",
    "choice": "A",
}
_CV_FORMAT_RULE = {
    "matrix": (
        "Write the matrix as one 3×3 line, in the form [[a, b, c], [d, e, f], [g, h, i]]. "
        "Multiplication must be written as *, and division as /. sin, cos, and tan must have parentheses. "
        "Do not use LaTeX, do not write an equals sign, and do not leave the matrix as an unexpanded product. "
        "Write each entry as an integer or as an expression in the given quantities. Do not leave P_c1x, P_c1y, P_c1z, or braced subscripts."
    ),
    "vector": (
        "Write the vector as one line of three components, in the form [a, b, c]. "
        "Multiplication must be written as *, and division as /. sin, cos, and tan must have parentheses. "
        "Write an integer when the value is an integer. Do not use LaTeX, do not write an equals sign, and do not leave the vector as an unexpanded matrix product. "
        "Write one vector for each blank id. Do not merge several blanks into one expression that has no blank id."
    ),
    "choice": "For a choice, write only the option letter, with no quotation marks.",
}
_CV_PROCESS = "stereo pose"
_CV_FINALS = (
    (
        (
            ("124", "R_t'"),
        ),
        "The transform matrix from the body frame to the wind frame is R_t'. "
        "From the mechanism angles α_M, β_M, and γ_M and the current axis declaration, give the expanded form of R_t'. "
        "Give R_t'.",
    ),
    (
        (
            ("37", "R_cw1"),
            ("39", "t_cw1"),
            ("76", "R_cw2"),
            ("78", "t_cw2"),
        ),
        "Two calibrated cameras photograph the scene from different directions. "
        "Camera 1 has azimuth, pitch, and roll ψ1, θ1, and φ1. Camera 2 has ψ2, θ2, and φ2. "
        "The optical centers in the wind frame are C1=(x1,y1,z1) and C2=(x2,y2,z2). "
        "From these angles, optical-center coordinates, and the current axis declaration, derive the two camera poses. "
        "R_cw1 is the rotation from the wind frame to camera 1, and t_cw1 is the pose translation of camera 1. "
        "R_cw2 is the rotation from the wind frame to camera 2, and t_cw2 is the pose translation of camera 2. "
        "Give R_cw1, t_cw1, R_cw2, and t_cw2.",
    ),
    (
        (
            ("140", "R21"),
            ("141", "t21"),
        ),
        "The optical centers of camera 1 and camera 2 in the wind frame are C1=(x1,y1,z1) and C2=(x2,y2,z2). "
        "The attitude angles are ψ1, θ1, φ1 and ψ2, θ2, φ2. "
        "Derive the rotation R21 and the translation t21 from camera 1 to camera 2. "
        "Give R21 and t21.",
    ),
    (
        (
            ("132", "P_c1"),
            ("135", "[P_c1]_×"),
            ("134", "P_c2"),
            ("136", "[P_c2]_×"),
        ),
        "The nose coordinates in camera 1 are P_c1, written as a vector. "
        "The skew-symmetric matrix of P_c1 is [P_c1]_×, written as a matrix. "
        "The nose coordinates in camera 2 are P_c2, written as a vector. "
        "The skew-symmetric matrix of P_c2 is [P_c2]_×, written as a matrix. "
        "From the mechanism angles α_M, β_M, and γ_M, "
        "camera 1 attitude ψ1, θ1, φ1, camera 2 attitude ψ2, θ2, φ2, "
        "optical centers C1=(x1,y1,z1) and C2=(x2,y2,z2), and the current axis declaration, expand all four. "
        "Expand [P_c1]_× by itself. Do not multiply it by (K1·S1)^(-T), R_cw1, or d_w. "
        "Expand [P_c2]_× by itself. Do not multiply it by (K2·S2)^(-T), R_cw2, or d_w. "
        "Give P_c1, [P_c1]_×, P_c2, and [P_c2]_×.",
    ),
    (
        (
            ("137", "d_wV"),
            ("138", "d_wL"),
            ("139", "d_wN"),
        ),
        "The direction of the airspeed in the wind frame is the vector d_wV. "
        "The airspeed appears in camera 1 as the image line l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV, "
        "and in camera 2 as l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV. "
        "The body longitudinal axis is the unit vector e in the body frame. Its direction in the wind frame is d_wL. "
        "That axis appears in camera 1 as l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL, "
        "and in camera 2 as l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL. "
        "The body normal axis is the unit vector e in the body frame. Its direction in the wind frame is d_wN. "
        "That axis appears in camera 1 as l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN, "
        "and in camera 2 as l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN. "
        "Write d_wV, d_wL, and d_wN.",
    ),
    (
        (
            ("145", "d1"),
            ("146", "d2"),
            ("147", "X_c1"),
            ("148", "constraint equation"),
            ("149", "cross-product vector"),
            ("150", "λ1"),
            ("151", "X_w"),
            ("152", "X_b"),
        ),
        "The homogeneous coordinates of the shock point in the camera 1 and camera 2 images are [u1,v1,1] and [u2,v2,1]. "
        "The ray direction in camera 1 is d1=(K1·S1)^(-1)·[u1,v1,1]^T, "
        "and the ray direction in camera 2 is d2=(K2·S2)^(-1)·[u2,v2,1]^T. "
        "Give the expanded form of each ray direction, written as a vector. "
        "Then invert the shock point. "
        "The shock-point coordinate X_c1 in camera 1 lies along the ray d1, and λ1 is the depth. Choose X_c1 from: "
        "A. X_c1 = λ1·d1; B. X_c1 = d1 + λ1; C. X_c1 = λ1 + d1; D. X_c1 = d1 - λ1. "
        "Both cameras see the same shock point, and the transform from camera 1 to camera 2 is X_c2=R21·X_c1+t21. Choose the constraint equation from: "
        "A. λ2·d2 = R21·λ1·d1 + t21; B. λ2·d2 = R21·λ1·d1 - t21; "
        "C. λ2·d2 = R21^T·λ1·d1 + t21; D. λ1·d1 = R21·λ2·d2 + t21. "
        "Crossing both sides of the constraint with a vector removes λ2 and gives d2×(λ1·R21·d1+t21)=0. Choose that vector from: "
        "A. d2; B. d1; C. R21·d1; D. t21. "
        "Under a least-squares criterion, and when d2×R21·d1 ≠ 0, "
        "find the λ1 that minimizes the squared residual of d2×(λ1·R21·d1+t21)=0. Choose λ1 from: "
        "A. λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²; "
        "B. λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²; "
        "C. λ1 = −(d2×t21) / (d2×R21·d1); "
        "D. λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|². "
        "The shock-point coordinate X_c1 in camera 1 is known. Choose its wind-frame coordinate X_w from: "
        "A. X_w = R_cw1^T·(X_c1 − t_cw1); B. X_w = R_cw1·X_c1 + t_cw1; "
        "C. X_w = R_cw1^T·X_c1 − t_cw1; D. X_w = R_cw1·(X_c1 − t_cw1). "
        "The wind-frame coordinate X_w is known. Choose its body-frame coordinate X_b from: "
        "A. X_b = R_t·RP1·(X_w − p_nose); B. X_b = RP1^T·R_t'·(X_w − p_nose); "
        "C. X_b = R_t·RP1·(X_w + p_nose); D. X_b = R_t'·RP1·(X_w − p_nose). "
        "Give d1 and d2, and give the option letters for X_c1, the constraint equation, the cross-product vector, λ1, X_w, and X_b.",
    ),
)
_BASIC_PROCESS = "vision basics"
_BASIC_FINALS = (
    (
        (
            ("37", "R21"),
            ("43", "t21"),
        ),
        "The camera 1 frame is the world frame. Camera 2 has azimuth, pitch, and roll ψ2, θ2, and φ2, "
        "and its optical center in the camera 1 frame is C2=(x2,y2,z2). "
        "From these angles, the optical-center coordinates, and the current axis declaration, "
        "derive the rotation R21 and the translation t21 from camera 1 to camera 2. "
        "Give R21 and t21.",
    ),
    (
        (
            ("47", "P_c1"),
            ("50", "[P_c1]_×"),
            ("49", "P_c2"),
            ("51", "[P_c2]_×"),
        ),
        "A point on the line has coordinates P_c1 in camera 1, written as a vector. "
        "The skew-symmetric matrix of P_c1 is [P_c1]_×, written as a matrix. "
        "The same point has coordinates P_c2 in camera 2, written as a vector. "
        "The skew-symmetric matrix of P_c2 is [P_c2]_×, written as a matrix. "
        "The line is parallel to the positive optical axis of camera 1. The point lies in the positive normal direction of camera 1, at distance L from the optical center. "
        "The camera 1 frame is the world frame. Camera 2 has azimuth, pitch, and roll ψ2, θ2, and φ2, "
        "and its optical center in the camera 1 frame is C2=(x2,y2,z2). "
        "From the distance L, these angles, the optical-center coordinates, and the current axis declaration, expand all four: "
        "P_c1, [P_c1]_×, P_c2, and [P_c2]_×.",
    ),
    (
        (
            ("52", "d_w"),
        ),
        "The camera 1 frame is the world frame. "
        "The image line in camera 1 is l=(K1·S1)^(-T)·[P_c1]_×·d_w, "
        "and the image line in camera 2 is l=(K2·S2)^(-T)·[P_c2]_×·R21·d_w. "
        "The line is parallel to the positive optical axis of camera 1. The direction d_w is the unit vector of that positive direction in the camera 1 frame. "
        "Give the expanded form of d_w. "
        "Expand d_w by itself as a vector. Do not multiply it by (K1·S1)^(-T), [P_c1]_×, or R21, and do not give the combined line l. "
        "Give d_w.",
    ),
    (
        (
            ("56", "d1"),
            ("57", "d2"),
            ("58", "X_c1"),
            ("59", "constraint equation"),
            ("60", "cross-product vector"),
            ("61", "λ1"),
            ("62", "X_w"),
        ),
        "The homogeneous coordinates of the target point in the camera 1 and camera 2 images are [u1,v1,1] and [u2,v2,1]. "
        "The ray direction in camera 1 is d1=(K1·S1)^(-1)·[u1,v1,1]^T, "
        "and the ray direction in camera 2 is d2=(K2·S2)^(-1)·[u2,v2,1]^T. "
        "Give the expanded form of each ray direction, written as a vector. "
        "Then invert the target point. "
        "The target-point coordinate X_c1 in camera 1 lies along the ray d1, and λ1 is the depth. Choose X_c1 from: "
        "A. X_c1 = λ1·d1; B. X_c1 = d1 + λ1; C. X_c1 = λ1 + d1; D. X_c1 = d1 - λ1. "
        "Both cameras see the same target point, and the transform from camera 1 to camera 2 is X_c2=R21·X_c1+t21. Choose the constraint equation from: "
        "A. λ2·d2 = R21·λ1·d1 + t21; B. λ2·d2 = R21·λ1·d1 - t21; "
        "C. λ2·d2 = R21^T·λ1·d1 + t21; D. λ1·d1 = R21·λ2·d2 + t21. "
        "Crossing both sides of the constraint with a vector removes λ2 and gives d2×(λ1·R21·d1+t21)=0. Choose that vector from: "
        "A. d2; B. d1; C. R21·d1; D. t21. "
        "Under a least-squares criterion, and when d2×R21·d1 ≠ 0, "
        "find the λ1 that minimizes the squared residual of d2×(λ1·R21·d1+t21)=0. Choose λ1 from: "
        "A. λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²; "
        "B. λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²; "
        "C. λ1 = −(d2×t21) / (d2×R21·d1); "
        "D. λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|². "
        "The target-point coordinate X_c1 in camera 1 is known. Camera 1 is the world frame, so choose its world coordinate X_w from: "
        "A. X_w = X_c1; B. X_w = R21^T·(X_c1 − t21); "
        "C. X_w = R21·X_c1 + t21; D. X_w = X_c1 + t21. "
        "Give d1 and d2, and give the option letters for X_c1, the constraint equation, the cross-product vector, λ1, and X_w.",
    ),
)

def _cv_declaration(declaration: str) -> str:
    text = declaration.strip()
    if text.startswith("[Declaration]"):
        text = text[len("[Declaration]"):].strip()
    return text


def _cv_example_field(blank: dict) -> str:
    example = _CV_FORMAT_EXAMPLE[blank["format_code"]]
    if blank["format_code"] in _CHOICE_CODE:
        return example
    return f'"{example}"'


def _cv_final_prompt(declaration: str, question: str, blanks: list[dict]) -> str:
    for blank in blanks:
        example = _CV_FORMAT_EXAMPLE[blank["format_code"]]
        if any(char in example for char in "αβγθφψμα"):
            raise RuntimeError("The computer-vision format example used a symbol from this item.")
    rules = []
    seen = set()
    for blank in blanks:
        code = blank["format_code"]
        if code in seen:
            continue
        seen.add(code)
        rules.append(_CV_FORMAT_RULE[code] + f" Example (not the answer to this item): {_CV_FORMAT_EXAMPLE[code]}.")
    asked = ", ".join(
        f'{blank["symbol"]} goes in key {blank["blank_id"]}' for blank in blanks
    )
    fields = ", ".join(
        f'{blank["blank_id"]}:{_cv_example_field(blank)}'
        for blank in blanks
    )
    example_object = "{" + fields + "}"
    answer_object = "{" + ", ".join(
        f'{blank["blank_id"]}:letter' if blank["format_code"] in _CHOICE_CODE else f'{blank["blank_id"]}:"answer"'
        for blank in blanks
    ) + "}"
    return "\n".join([
        "This call asks only for the final result and does not give a step-by-step cloze. Derive it from the declaration yourself, and leave the derivation in the chain of thought.",
        f"The final answer is one JSON object {answer_object}.",
        " ".join(rules),
        "The λ and ω in the example only show how the characters are arranged. They are not quantities in this item, so do not write them in the answer.",
        "",
        "[Declaration]",
        _cv_declaration(declaration),
        "",
        "[Question]",
        question.strip(),
        "",
        "[Output]",
        " ".join(rules),
        asked + ".",
        "Write one value for each blank id. Do not merge several blanks into one expression that has no blank id. Do not use LaTeX.",
        "The λ and ω in the example only show how the characters are arranged. They are not quantities in this item, so do not write them in the answer.",
        f"Output only one line {answer_object}.",
        f"Format example (not the answer to this item): {example_object}",
    ])


def cv_final_jobs(
    row: dict,
    run: int,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
    finals: tuple | None = None,
    process: str | None = None,
) -> list[dict]:
    process_name = process or _CV_PROCESS
    indexed = {
        (item["process"], str(item["local_id"])): item
        for item in row["questions"]
    }
    jobs = []
    for slots, question in (finals or _CV_FINALS):
        blanks = []
        for local_id, symbol in slots:
            item = indexed.get((process_name, local_id))
            answer = "" if item is None else str(item.get("answer") or "")
            code = "" if item is None else item.get("format_code")
            symbolic = code in {"matrix", "vector"} and answer.startswith(("[", "[["))
            choice = code in _CHOICE_CODE and len(answer) == 1 and answer.isalpha() and bool(item.get("options"))
            if item is None or not (symbolic or choice):
                raise SystemExit(
                    f"Sample {row['id']} subprocess {process_name} local id {local_id} is not an end-to-end answer."
                )
            blanks.append({
                "blank_id": str(item["id"]),
                "label": _END_TO_END_LABEL,
                "format_code": item["format_code"],
                "answer": item["answer"],
                "options": list(item.get("options") or []),
                "symbol": symbol,
            })
        jobs.append({
            "sample_id": row["id"],
            "tier": row.get("tier", 0),
            "process": process_name,
            "blank_ids": [blank["blank_id"] for blank in blanks],
            "blanks": blanks,
            "condition": FINAL_E2E,
            "run": run,
            "seed": seed,
            "max_tokens": max_tokens,
            "stop": stop,
            "standard": standard,
            "prompt": _cv_final_prompt(row["declaration"], question, blanks),
        })
    return jobs


def _manuscript_end_to_end(chain: dict) -> str:
    from domains.eval_manuscripts import render_sentence

    body = "\n".join(render_sentence(sentence, fill=False) for sentence in chain["sentences"])
    return "\n".join([
        "This is an end-to-end question. Fill every open slot in this subprocess, and call the model only once.",
        "Output one JSON object. The keys are the blank ids and the values are the answers.",
        "A choice value is a letter. Matrices, vectors, and expressions are strings inside English double quotes.",
        "Do not output the original sentence and do not explain.",
        "",
        "[Process]",
        body,
        "[Output]",
        "One JSON object.",
    ])


def _manuscript_prompt(
    chain: dict,
    blank_ids: list[str],
    condition: str,
    memory: dict[str, str] | None,
) -> str:
    from domains.eval_manuscripts import process_text

    body = process_text(
        chain,
        str(blank_ids[0]),
        condition in _KEEP_SUFFIX,
        memory,
    )
    return "\n".join([
        "[Process]",
        body,
        "[Output]",
        "Output only one JSON object.",
    ])


def prompt_for(
    chain: dict,
    blank_ids: list[str],
    condition: str,
    memory: dict[str, str] | None,
    local_target: str | None = None,
) -> str:
    if chain.get("domain") == "manuscript":
        return _manuscript_prompt(chain, blank_ids, condition, memory)
    if chain.get("domain") == "flight":
        module = _flight_modules()[chain["process"]]
        return module._injection(
            chain["declaration"],
            chain["by_id"],
            local_target or blank_ids[0],
            memory=memory,
            keep_suffix=condition in _KEEP_SUFFIX,
        )
    if chain.get("domain") == "cv":
        return _cv_module(chain.get("process"))._injection(
            chain["declaration"],
            chain["by_id"],
            local_target or blank_ids[0],
            memory=memory,
            keep_suffix=condition in _KEEP_SUFFIX,
        )
    return _injection(
        chain["declaration"],
        chain["process"],
        chain["by_id"],
        blank_ids[0],
        chain["phrases"],
        memory=memory,
        keep_suffix=condition in _KEEP_SUFFIX,
        template=chain.get("process_text"),
    )


def transcript(messages: list[dict]) -> str:
    lines = []
    for item in messages:
        lines.append(f"[{item['role']}]")
        lines.append(item["content"])
        lines.append("")
    return "\n".join(lines).strip()


def gold_reply(blanks: list[dict]) -> str:
    """The gold answer in the same JSON shape the question asks for."""
    fields = []
    for blank in blanks:
        answer = str(blank.get("answer") or "")
        blank_id = blank["blank_id"]
        if blank.get("options"):
            fields.append(f"{blank_id}:{answer}")
        else:
            escaped = answer.replace("\\", "\\\\").replace('"', '\\"')
            fields.append(f'{blank_id}:"{escaped}"')
    return "{" + ", ".join(fields) + "}"


def _process_excerpt(chain: dict, sentence: dict) -> str:
    """Manuscript from the start of the subprocess through the current period."""
    text = prompt_for(
        chain,
        sentence["blank_ids"],
        GOLD_TRUNC,
        None,
        (sentence.get("local_ids") or [None])[0],
    )
    start = text.find("[Process]")
    end = text.find("\n[Output]", start)
    if start < 0 or end < 0:
        raise RuntimeError(f"{chain['process']} has no process text for the current sentence.")
    return text[start + len("[Process]"):end].strip()


def _close_slots(text: str, sentence: dict) -> str:
    """Fill this sentence's open slots so it matches the next turn's gold prefix."""
    from domains.wind_tunnel.dataset_generate import _gold_phrase, _open_slot

    for blank in sentence["blanks"]:
        item = {
            "id": str(blank["blank_id"]),
            "answer": blank.get("answer") or "",
            "options": list(blank.get("options") or []),
            "format_code": blank.get("format_code") or "",
        }
        opened = _open_slot(item)
        if opened not in text:
            continue
        text = text.replace(opened, _gold_phrase(item), 1)
    return text


def _session_question(chain: dict, sentence: dict, previous: dict | None) -> str:
    """Only the new sentence. Earlier sentences stay in the chat history."""
    current = _process_excerpt(chain, sentence).strip()
    if previous is not None:
        earlier = _close_slots(_process_excerpt(chain, previous), previous).strip()
        if current.startswith(earlier):
            current = current[len(earlier):].strip()
    lead = "Fill the current sentence. Write only the JSON for this sentence, with no explanation."
    return "\n".join([
        lead,
        "",
        "[Subprocess]",
        chain["process"],
        "",
        "[Question]",
        current,
    ])


def _session_system(chain: dict) -> str:
    prior = ""
    if chain.get("domain") == "manuscript":
        prior = chain.get("prior") or ""
    elif chain.get("domain") == "flight":
        prior = _flight_modules()[chain["process"]].PRIOR
    elif chain.get("domain") == "cv":
        prior = _cv_module(chain.get("process")).PRIOR
    else:
        from domains.wind_tunnel.dataset_generate import PRIOR
        prior = PRIOR
    parts = [
        "This session states the definitions, the background, and the output format once, in this system message.",
        "Each later question adds only the current sentence. Retrieve the engineering definitions from this message; do not wait for them to be repeated.",
        "",
        "[Declaration]",
        chain["declaration"].strip(),
    ]
    if str(prior).strip():
        parts.extend(["", "[Background]", str(prior).strip()])
    format_note = str(chain.get("format_note") or "").strip()
    if format_note:
        parts.extend(["", "[Output format]", format_note])
    parts.extend([
        "",
        "[Output]",
        "This question writes one JSON object. The key is the blank id and the value is the answer.",
        "A multiple-choice value is only a letter. A matrix, vector, expression, or formula is a string in English double quotes.",
        "Do not write the sentence text, and do not explain.",
    ])
    return "\n".join(parts)


def _guard_session(messages: list[dict], declaration: str) -> None:
    marker = declaration.strip()[:40]
    if len(marker) < 12:
        return
    for item in messages:
        if item["role"] == "user" and marker in item["content"]:
            raise RuntimeError("A later question injected the declaration again.")


def job_for(
    chain: dict,
    sentence: dict,
    condition: str,
    memory: dict[str, str] | None,
    run: int,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
) -> dict:
    return {
        "sample_id": chain["sample_id"],
        "tier": chain["tier"],
        "process": chain["process"],
        "group": chain.get("group", "main"),
        "blank_ids": sentence["blank_ids"],
        "blanks": sentence["blanks"],
        "condition": condition,
        "run": run,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": stop,
        "standard": standard,
        "prompt": prompt_for(
            chain,
            sentence["blank_ids"],
            condition,
            memory,
            (sentence.get("local_ids") or [None])[0],
        ),
    }


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _read_string(text: str, index: int) -> tuple[str | None, int]:
    quote = text[index]
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "\\":
            if index + 1 >= len(text):
                return None, index
            chars.append(text[index + 1])
            index += 2
            continue
        if char == quote:
            return "".join(chars), index + 1
        chars.append(char)
        index += 1
    return None, index


def _read_bare(text: str, index: int) -> tuple[str, int]:
    start = index
    depth = 0
    while index < len(text):
        char = text[index]
        if char in "[{(":
            depth += 1
        elif char in "]})":
            if depth == 0:
                break
            depth -= 1
        elif char == "," and depth == 0:
            break
        index += 1
    return text[start:index].strip(), index


def _read_key(text: str, index: int) -> tuple[str | None, int]:
    index = _skip_ws(text, index)
    if index >= len(text):
        return None, index
    if text[index] in "\"'":
        return _read_string(text, index)
    start = index
    while index < len(text) and text[index] != ":":
        index += 1
    return text[start:index].strip(), index


def _read_object(text: str, index: int) -> tuple[dict[str, str] | None, int]:
    if index >= len(text) or text[index] != "{":
        return None, index
    index += 1
    obj: dict[str, str] = {}
    while index < len(text):
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == "}":
            return obj, index + 1
        key, index = _read_key(text, index)
        if key is None:
            return None, index
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            return None, index
        index = _skip_ws(text, index + 1)
        if index >= len(text):
            return None, index
        if text[index] in "\"'":
            value, index = _read_string(text, index)
            if value is None:
                return None, index
        else:
            value, index = _read_bare(text, index)
        obj[key] = value.strip()
        index = _skip_ws(text, index)
        if index < len(text) and text[index] == ",":
            index += 1
            continue
        if index < len(text) and text[index] == "}":
            return obj, index + 1
        return None, index
    return None, index


def _objects(text: str) -> list[dict[str, str]]:
    found = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        obj, end = _read_object(text, index)
        if obj is None or end <= index:
            index += 1
            continue
        found.append(obj)
        index = end
    return found


def _parse_reply(response: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for obj in _objects(response):
        lowered = {key.lower(): value for key, value in obj.items()}
        if "id" in lowered and "answer" in lowered and lowered["id"].isdigit():
            parsed[lowered["id"]] = lowered["answer"]
            continue
        for key, value in obj.items():
            if key.isdigit():
                parsed[key] = value
    if parsed:
        return parsed
    for line in reply_lines(response):
        match = re.match(r"^(?:ID\s*:?\s*)?(\d+)\s*[:.\s]\s*(.*)$", line)
        if match:
            parsed[match.group(1)] = match.group(2).strip()
    return parsed


def _escaped(text: str, index: int) -> bool:
    slashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        slashes += 1
        cursor -= 1
    return slashes % 2 == 1


def _consume_group(text: str, index: int) -> int:
    depth = 0
    cursor = index
    while cursor < len(text):
        if text.startswith("\\begin{", cursor) and not _escaped(text, cursor):
            cursor = _consume_environment(text, cursor)
            continue
        char = text[cursor]
        if char in "[{(" and not _escaped(text, cursor):
            depth += 1
        elif char in "]})" and not _escaped(text, cursor):
            depth -= 1
            if depth == 0:
                return cursor + 1
        cursor += 1
    return len(text)


def _consume_environment(text: str, index: int) -> int:
    matched = re.match(r"\\begin\{([^{}]+)\}", text[index:])
    if matched is None:
        return index + 1
    name = matched.group(1)
    closing = "\\end{" + name + "}"
    found = text.find(closing, index + matched.end())
    if found < 0:
        return len(text)
    return found + len(closing)


def _consume_atom(text: str, index: int) -> int:
    depth = 0
    cursor = index
    while cursor < len(text):
        if depth == 0 and cursor > index and (
            text[cursor] in "\n[{$" or text.startswith("\\begin{", cursor) or text.startswith("\\[", cursor)
        ):
            break
        char = text[cursor]
        if char in "([{" and not _escaped(text, cursor):
            depth += 1
        elif char in ")]}" and not _escaped(text, cursor):
            depth = max(0, depth - 1)
        elif depth == 0 and char in "\n;":
            break
        cursor += 1
    return cursor


_LABEL_ONLY = re.compile(
    r"^(?:answer|result|therefore|so)?\s*[A-Za-zα-ωθφψΔ0-9_\\]{0,16}\s*[:=]\s*$"
)


def _is_answer_chunk(chunk: str) -> bool:
    text = chunk.strip().strip(";")
    if not text:
        return False
    if re.fullmatch(r"[A-Za-z]", text):
        return True
    if _LABEL_ONLY.fullmatch(text):
        return False
    if re.search(r"[\[\]\\0-9+\-*/=α-ωθφψΔ]", text):
        return True
    return re.search(r"(?:arcsin|arctan|sin|cos|tan)", text) is not None


def _unnumbered_chunks(text: str) -> list[str]:
    source = text.replace("\r\n", "\n")
    source = re.sub(r"```(?:\w+)?", "", source)
    objects = _objects(source)
    if objects and not any(key.strip().isdigit() for obj in objects for key in obj):
        values = [value.strip() for obj in objects for value in obj.values() if value.strip()]
        if values:
            return values
    pieces: list[str] = []
    index = 0
    length = len(source)
    while index < length:
        while index < length and source[index] in " \t\r\n,":
            index += 1
        if index >= length:
            break
        if source.startswith("\\begin{", index):
            end = _consume_environment(source, index)
        elif source.startswith("\\[", index):
            found = source.find("\\]", index + 2)
            end = length if found < 0 else found + 2
        elif source.startswith("\\(", index):
            found = source.find("\\)", index + 2)
            end = length if found < 0 else found + 2
        elif source[index] == "$":
            end = source.find("$", index + 1)
            end = length if end < 0 else end + 1
        elif source[index] in "[({":
            end = _consume_group(source, index)
        else:
            end = _consume_atom(source, index)
        chunk = source[index:end].strip().strip(",")
        index = max(end, index + 1)
        if _is_answer_chunk(chunk):
            pieces.append(chunk)
    return pieces


def _assigned_predictions(response: str, blanks: list[dict]) -> dict[str, str]:
    """Map a reply onto blanks. Missing ids follow the blank order."""
    text = response.replace("\r\n", "\n").strip()
    parsed = _parse_reply(text)
    blank_ids = [str(blank["blank_id"]) for blank in blanks]
    if len(blank_ids) == 1:
        return {blank_ids[0]: parsed.get(blank_ids[0], text)}
    if any(blank_id in parsed for blank_id in blank_ids):
        return {blank_id: parsed.get(blank_id, "") for blank_id in blank_ids}
    assigned = {blank_id: "" for blank_id in blank_ids}
    for blank_id, chunk in zip(blank_ids, _unnumbered_chunks(text)):
        assigned[blank_id] = chunk
    return assigned


def _check_parse() -> None:
    parsed = _parse_reply("{1:A}\n{2:B}")
    if parsed != {"1": "A", "2": "B"}:
        raise RuntimeError(parsed)
    merged = _parse_reply('{1:A, 2:C, 11:"[[0, 1], [1, 0]]"}')
    if merged.get("1") != "A" or merged.get("2") != "C" or merged.get("11") != "[[0, 1], [1, 0]]":
        raise RuntimeError(merged)
    matrix = _parse_reply('{11:"[[0, 0, 1], [0, -1, 0], [-1, 0, 0]]"}')
    if matrix.get("11") != "[[0, 0, 1], [0, -1, 0], [-1, 0, 0]]":
        raise RuntimeError(matrix)
    if _parse_reply('{63:"-γ_e"}').get("63") != "-γ_e":
        raise RuntimeError("quoted expression")
    legacy_object = _parse_reply("{ID:1,Answer:A}\n{ID:2,Answer:B}")
    if legacy_object != {"1": "A", "2": "B"}:
        raise RuntimeError(legacy_object)
    quoted = _parse_reply('{"ID": 3, "Answer": "C"}')
    if quoted.get("3") != "C":
        raise RuntimeError(quoted)
    legacy = _parse_reply("ID:5:YTZ")
    if legacy.get("5") != "YTZ":
        raise RuntimeError(legacy)
    ordered = _score_blanks(
        "[(u1-cx1)/fx1, (v1-cy1)/fy1, 1]",
        [
            {"blank_id": "145", "label": "A2", "format_code": "vector", "options": [], "answer": "[(-cx1 + u1)/fx1, (-cy1 + v1)/fy1, 1]"},
            {"blank_id": "146", "label": "A2", "format_code": "vector", "options": [], "answer": "[1, 0, 0]"},
        ],
    )
    if not ordered[0]["normalized"] or ordered[1]["prediction"] or ordered[1]["normalized"]:
        raise RuntimeError(ordered)
    paired = _score_blanks(
        "[1, 0, 0]\n[0, 1, 0]",
        [
            {"blank_id": "137", "label": "A2", "format_code": "vector", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "138", "label": "A2", "format_code": "vector", "options": [], "answer": "[0, 1, 0]"},
            {"blank_id": "139", "label": "A2", "format_code": "vector", "options": [], "answer": "[0, 0, 1]"},
        ],
    )
    if not paired[0]["exact"] or not paired[1]["exact"] or paired[2]["prediction"] or paired[2]["normalized"]:
        raise RuntimeError(paired)
    named = _score_blanks(
        '{"d1":"[1, 0, 0]","d2":"[0, 1, 0]"}',
        [
            {"blank_id": "145", "label": "A2", "format_code": "vector", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "146", "label": "A2", "format_code": "vector", "options": [], "answer": "[0, 1, 0]"},
        ],
    )
    if not named[0]["exact"] or not named[1]["exact"]:
        raise RuntimeError(named)
    numbered = _score_blanks(
        '{146:"[0, 1, 0]", 145:"[1, 0, 0]"}',
        [
            {"blank_id": "145", "label": "A2", "format_code": "vector", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "146", "label": "A2", "format_code": "vector", "options": [], "answer": "[0, 1, 0]"},
        ],
    )
    if not numbered[0]["exact"] or not numbered[1]["exact"]:
        raise RuntimeError(numbered)
    if _resolve_max_tokens(0) != DEFAULT_MAX_TOKENS or _resolve_max_tokens(MIN_MAX_TOKENS) != MIN_MAX_TOKENS:
        raise RuntimeError("max_tokens did not stay in 2048-8192.")
    if _resolve_max_tokens(0, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("Thinking on did not use 65536.")
    if _resolve_max_tokens(4096, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("Thinking on was not fixed at 65536.")
    if _resolve_max_tokens(100000, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("Thinking on was not fixed at 65536.")
    if _resolve_max_tokens(100000, announce=False) != MAX_TOKEN_CEILING:
        raise RuntimeError("With thinking off, max_tokens was not capped at 32768.")
    if _resolve_workers(MAX_WORKERS) != MAX_WORKERS:
        raise RuntimeError("The concurrency ceiling was overwritten.")


def reply_lines(prediction: str) -> list[str]:
    lines = [line.strip() for line in prediction.replace("\r\n", "\n").splitlines()]
    return [line for line in lines if line and not line.startswith("```")]


def line_hit(prediction: str, answer: str) -> bool:
    return answer in reply_lines(prediction)


def _score_blanks(response: str, blanks: list[dict], standard: dict | None = None, process: str = "") -> list[dict]:
    from cloze_score import judge

    text = response.replace("\r\n", "\n").strip()
    lines = reply_lines(response)
    parsed = _parse_reply(response)
    assigned = _assigned_predictions(text, blanks)
    scored = []
    from cloze_score import answer_letter

    for blank in blanks:
        blank_id = blank["blank_id"]
        answer = answer_letter(blank["answer"], blank.get("options") or [])
        prediction = assigned.get(blank_id, "")
        hit = prediction == answer or line_hit(text, answer) or answer in parsed.values()
        if len(blanks) > 1 and not prediction:
            hit = answer in lines or answer in parsed.values()
        verdict = judge(
            prediction,
            answer,
            blank.get("format_code") or "",
            blank.get("options"),
            None if standard is None else standard.get((process, blank_id)),
        )
        scored.append({
            "blank_id": blank_id,
            "label": blank["label"],
            "format_code": blank.get("format_code") or "",
            "options": list(blank.get("options") or []),
            "answer": answer,
            "prediction": prediction,
            "line_hit": hit,
            **_dependency_fields(blank),
            **verdict,
        })
    return scored


def _plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    data = getattr(value, "__dict__", None)
    if isinstance(data, dict):
        return _plain({key: item for key, item in data.items() if not str(key).startswith("_")})
    return str(value)


def _price_name(model: str, base_url: str) -> str | None:
    if "deepseek" not in base_url.lower():
        return None
    name = model.strip().lower()
    if name in _FLASH_NAMES:
        return "deepseek-flash"
    if name in _PRO_NAMES:
        return "deepseek-v4-pro"
    return None


def _band(moment: datetime) -> tuple[str, str]:
    local = moment.astimezone(SHANGHAI)
    calendar = "2026" if local.year == 2026 else "weekday_only"
    holiday = local.date() in _HOLIDAYS_2026 if local.year == 2026 else False
    peak_clock = (9 <= local.hour < 12) or (14 <= local.hour < 18)
    band = "peak" if local.weekday() < 5 and peak_clock and not holiday else "off_peak"
    return band, calendar


def _token_counts(usage: dict) -> tuple[int, int, int]:
    details = usage.get("prompt_tokens_details") or {}
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    if hit is None:
        hit = details.get("cached_tokens") or 0
    if miss is None:
        miss = max(int(usage.get("prompt_tokens") or 0) - int(hit), 0)
    return int(hit), int(miss), int(usage.get("completion_tokens") or 0)


def _deepseek_cost(usage: dict, moment: datetime, model: str, base_url: str) -> dict:
    priced = _price_name(model, base_url)
    if not usage:
        return {"usd": None, "note": "This call did not return usage."}
    if priced is None:
        return {"usd": None, "note": "The price table covers only flash and v4-pro on the official DeepSeek API."}
    hit, miss, output = _token_counts(usage)
    band, calendar = _band(moment)
    hit_rate, miss_rate, output_rate = _RATES[priced][band]
    usd = (hit * hit_rate + miss * miss_rate + output * output_rate) / 1_000_000
    return {
        "usd": round(usd, 8),
        "band": band,
        "holiday_calendar": calendar,
        "cache_hit_tokens": hit,
        "cache_miss_tokens": miss,
        "output_tokens": output,
        "per_million_usd": {"cache_hit": hit_rate, "cache_miss": miss_rate, "output": output_rate},
        "source": "https://api-docs.deepseek.com/quick_start/pricing",
    }


MIN_MAX_TOKENS = 2048
DEFAULT_MAX_TOKENS = 8192
# Official DeepSeek output ceiling is 384K. Thinking uses twice the old
# 32768 ceiling and does not raise the budget after a truncated reply.
MAX_TOKEN_CEILING = 32768
THINKING_MAX_TOKENS = MAX_TOKEN_CEILING * 2
MAX_MEMORY_CHARS = 400
MAX_WORKERS = 4


def _strip_reasoning(content: str, reasoning: str) -> str:
    """Drop a reasoning block that a provider inlined into the answer text."""
    if not content or not reasoning:
        return content
    block = reasoning.strip()
    if len(block) < 40 or block not in content:
        return content
    return content.replace(block, "").strip()


def _memory_text(value: str | None) -> str:
    """Keep the previous answer short enough that the next prompt cannot balloon."""
    text = (value or "").strip().replace("\r", " ")
    if len(text) <= MAX_MEMORY_CHARS:
        return text
    return text[:MAX_MEMORY_CHARS] + "…"


def _repair_phrase(blank: dict) -> str:
    """Gold answer in the same surface form used for a gold prefix."""
    from domains.wind_tunnel.dataset_generate import _gold_phrase

    return _memory_text(_gold_phrase(blank))


def _remember(memory: dict[str, str], blank: dict, repair_ids: set[str]) -> None:
    """Store the prefix the next sentence will see.

    A repaired node keeps the model's score, but later sentences receive the
    gold answer. Same-sentence blanks are answered together and do not see it.
    """
    blank_id = str(blank["blank_id"])
    if blank_id in repair_ids:
        memory[blank_id] = _repair_phrase(blank)
        blank["prefix_source"] = "gold"
    else:
        memory[blank_id] = _prefix_phrase(blank)
        blank["prefix_source"] = "model"


def _prefix_phrase(blank: dict) -> str:
    """Model prefix in the same surface form as a gold prefix.

    A choice letter becomes the option text. An unparsed reply stays as written
    and is not replaced with the gold answer.
    """
    prediction = blank.get("prediction") or ""
    options = [str(option) for option in (blank.get("options") or []) if option and str(option) != "none"]
    if options:
        from cloze_score import identified_option

        selected = identified_option(prediction, options)
        if selected:
            prediction = selected
    return _memory_text(prediction)


def _resolve_max_tokens(requested: int, thinking: str = "disabled", announce: bool = True) -> int:
    """Reasoning tokens count against max_tokens. 512 truncates the answer.

    Thinking is fixed at 65536. A truncated thinking reply is scored as wrong
    and is not retried at another budget or effort.
    """
    if thinking == "enabled":
        if announce and requested > 0 and requested != THINKING_MAX_TOKENS:
            print(
                f"With thinking on, max_tokens is fixed at {THINKING_MAX_TOKENS}. The requested {requested} was changed to that value. "
                "If the chain of thought uses up the ceiling, the item is scored wrong. The ceiling is not raised and the thinking effort is not changed.",
                flush=True,
            )
        return THINKING_MAX_TOKENS
    if requested <= 0:
        return DEFAULT_MAX_TOKENS
    if requested < MIN_MAX_TOKENS:
        if announce:
            print(
                f"max_tokens {requested} is below {MIN_MAX_TOKENS}, so it was raised to {MIN_MAX_TOKENS}. "
                "The chain of thought counts toward that ceiling. 512 would cut off the final answer, and a rerun costs more than the extra tokens.",
                flush=True,
            )
        return MIN_MAX_TOKENS
    if requested > MAX_TOKEN_CEILING:
        if announce:
            print(
                f"max_tokens {requested} is above the global ceiling {MAX_TOKEN_CEILING}, so it was cut to {MAX_TOKEN_CEILING}.",
                flush=True,
            )
        return MAX_TOKEN_CEILING
    return requested


def _resolve_workers(requested: int) -> int:
    """Long outputs at high concurrency turn into timeouts and 429 retries."""
    if requested < 1:
        raise SystemExit("--workers must be at least 1.")
    if requested > MAX_WORKERS:
        print(
            f"Concurrency {requested} increases timeouts and 429 retries on long output, so it was limited to {MAX_WORKERS}. "
            "A retry is an extra billed call and is not included in the tokens of the successful record.",
            flush=True,
        )
        return MAX_WORKERS
    return requested


class RunStop(RuntimeError):
    """Quota or a sustained rate limit. The failed sentence is not written, so the same sentence can be resumed."""

    def __init__(self, kind: str, message: str, job: dict):
        super().__init__(message)
        self.kind = kind
        self.job = job


def _error_kind(text: str) -> str:
    lowered = text.lower()
    # 402 must be a standalone status code. A 402 inside a request id is not a quota error.
    if re.search(r"(?<![0-9a-z])402(?![0-9a-z])", lowered) or any(token in lowered for token in (
        "insufficient", "quota", "arrearage", "balance", "allocationquota",
        "freetier", "billing", "prepaid",
    )):
        return "quota"
    if any(token in lowered for token in ("429", "rate limit", "ratelimit", "throttl", "too many requests")):
        return "rate_limit"
    return ""


def score_one(client, model: str, temperature: float, thinking: str, job: dict) -> dict:
    started = datetime.now().astimezone()
    last_error = ""
    prediction = ""
    reasoning = ""
    captured: dict = {}
    attempts = 0
    max_tokens = job.get("max_tokens")
    truncated = False
    unanswered = False
    for _attempt in range(8):
        attempts = _attempt + 1
        captured = {}
        try:
            prediction, reasoning = call_model(
                client,
                model,
                job["prompt"],
                temperature,
                thinking,
                captured=captured,
                max_tokens=max_tokens,
                stop=job.get("stop"),
                seed=job.get("seed"),
                messages=job.get("messages"),
            )
            last_error = ""
            if captured.get("finish_reason") != "length":
                break
            truncated = True
            prediction = ""
            unanswered = True
            last_error = f"The model did not answer: the output was cut off at {max_tokens} tokens."
            break
        except RunStop:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the API error and retry
            last_error = f"{type(exc).__name__}: {exc}"
            prediction = ""
            reasoning = ""
            kind = _error_kind(last_error)
            if kind == "quota":
                raise RunStop(kind, last_error, job) from exc
            if _attempt < 7:
                time.sleep(20 * (_attempt + 1) if kind == "rate_limit" else min(8, 2 ** _attempt))
    if _error_kind(last_error) == "rate_limit":
        raise RunStop("rate_limit", last_error, job)
    prediction = _strip_reasoning(prediction, reasoning)
    finished = datetime.now().astimezone()
    usage = _plain(captured.get("usage"))
    if not isinstance(usage, dict):
        usage = {}
    base_url = str(getattr(client, "base_url", "") or "")
    return {
        "sample_id": job["sample_id"],
        "tier": job["tier"],
        "process": job["process"],
        "group": job.get("group", "main"),
        "blank_ids": job["blank_ids"],
        "condition": job.get("condition", "A"),
        "run": job.get("run", 0),
        "blanks": _score_blanks(prediction, job["blanks"], job.get("standard"), job["process"]),
        "prompt": job["prompt"],
        "response": prediction,
        "reasoning": reasoning,
        "attempts": attempts,
        "usage": usage,
        "cost": _deepseek_cost(usage, started, model, base_url),
        "model": model,
        "model_version": captured.get("response_model") or "",
        "documented_model_version": _DOCUMENTED_VERSION.get(model.strip().lower(), ""),
        "response_id": captured.get("response_id") or "",
        "system_fingerprint": captured.get("system_fingerprint"),
        "finish_reason": captured.get("finish_reason"),
        "started_at": started.isoformat(timespec="milliseconds"),
        "finished_at": finished.isoformat(timespec="milliseconds"),
        "latency_ms": int((finished - started).total_seconds() * 1000),
        "temperature": temperature,
        "temperature_applied": thinking != "enabled",
        "thinking": thinking,
        "reasoning_effort": captured.get("reasoning_effort") or "",
        "seed": job.get("seed"),
        "session": job.get("session", ""),
        "turn": job.get("turn"),
        "max_tokens": max_tokens,
        "truncated": truncated,
        "unanswered": unanswered,
        "error": "" if unanswered else last_error,
    }


def flatten(rows: list[dict]) -> list[dict]:
    flat = []
    for item in rows:
        if "blanks" not in item:
            flat.append(item)
            continue
        for blank in item["blanks"]:
            flat.append({
                "sample_id": item["sample_id"],
                "tier": item["tier"],
                "process": item["process"],
                "run": item.get("run", 0),
                "blank_id": blank["blank_id"],
                "label": blank["label"],
                "format_code": blank["format_code"],
                "answer": blank["answer"],
                "prediction": blank["prediction"],
                "exact": blank["exact"],
                "normalized": blank.get("normalized", blank["exact"]),
                "sympy_equal": blank.get("sympy_equal", False),
                "option_match": blank.get("option_match"),
                "error_kind": blank.get("error_kind", ""),
                "line_hit": blank["line_hit"],
                "error": item.get("error", ""),
                **_dependency_fields(blank),
            })
    return flat


def _record_key(item: dict) -> tuple:
    run = item.get("run", 0)
    condition = item.get("condition", "A")
    if "blank_ids" in item:
        return (condition, item["sample_id"], item["process"], ",".join(item["blank_ids"]), run)
    return (condition, item["sample_id"], item["process"], item["blank_id"], run)


def _rate(hit: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{hit / total:.1%}"


def summarize(rows: list[dict], preface: list[str] | None = None) -> str:
    from cloze_score import bootstrap_ci

    rows = flatten(rows)

    def block(title: str, subset: list[dict]) -> list[str]:
        total = len(subset)
        exact = sum(bool(item["exact"]) for item in subset)
        normalized = sum(bool(item.get("normalized", item["exact"])) for item in subset)
        errors = sum(bool(item.get("error")) for item in subset)
        return [
            f"### {title}",
            "",
            f"- blanks {total}, strict {exact} ({_rate(exact, total)}), normalized {normalized} ({_rate(normalized, total)}), call failures {errors}",
            "",
        ]

    lines = ["# Cloze evaluation", ""]
    if preface:
        lines.extend(preface)
        lines.append("")
    lines.extend(block("All", rows))
    grouped: dict[str, list[bool]] = {}
    grouped_norm: dict[str, list[bool]] = {}
    order = []
    for item in rows:
        key = f"{item['sample_id']}#{item.get('run', 0)}"
        if key not in grouped:
            order.append(key)
            grouped[key] = []
            grouped_norm[key] = []
        grouped[key].append(bool(item["exact"]))
        grouped_norm[key].append(bool(item.get("normalized", item["exact"])))
    strict_low, strict_high = bootstrap_ci([grouped[key] for key in order])
    norm_low, norm_high = bootstrap_ci([grouped_norm[key] for key in order])
    lines.append(
        f"Sample bootstrap 95% interval: strict {strict_low:.1%}-{strict_high:.1%}, normalized {norm_low:.1%}-{norm_high:.1%}. "
        "The resampled unit is one run of one sample."
    )
    lines.append("")
    lines.append("## Error classes")
    lines.append("")
    lines.append("Only strict mismatches are counted. format means that, after the answer keyword, dropping spaces, stray characters, and LaTeX leaves the same text, that an expression or matrix is equivalent under sympy, or that a choice hits the letter, the option text, or a fragment that maps to only the correct option.")
    lines.append("")
    misses = [item for item in rows if not item["exact"]]
    counts = Counter(item.get("error_kind") or "other" for item in misses)
    for kind in ("format", "axis", "sign", "unused-definition", "other"):
        lines.append(f"- {kind} {counts.get(kind, 0)}")
    lines.append("")
    lines.append("## By definition tier")
    lines.append("")
    present_tiers = []
    for item in rows:
        if item["tier"] not in present_tiers:
            present_tiers.append(item["tier"])
    for tier in sorted(present_tiers, key=lambda value: int(value)):
        lines.extend(block(f"{tier}%", [item for item in rows if item["tier"] == tier]))
    lines.append("## By label")
    lines.append("")
    labels = ["C1", "C2", "A1", "A2"]
    if any(item["label"] == _END_TO_END_LABEL for item in rows):
        labels.append(_END_TO_END_LABEL)
    for label in labels:
        lines.extend(block(label, [item for item in rows if item["label"] == label]))
    lines.append("## By data")
    lines.append("")
    sample_ids = []
    for item in rows:
        if item["sample_id"] not in sample_ids:
            sample_ids.append(item["sample_id"])
    for sample_id in sample_ids:
        subset = [item for item in rows if item["sample_id"] == sample_id]
        tier = subset[0]["tier"]
        lines.extend(block(f"{sample_id} ({tier}%)", subset))
    shown = [item for item in misses if item.get("error_kind") != "format"][:30]
    lines.extend(["## Still different after normalization (at most 30)", ""])
    if not shown:
        lines.append("None.")
    for item in shown:
        pred = str(item["prediction"]).replace("\n", " / ")
        if len(pred) > 180:
            pred = pred[:180] + "…"
        lines.append(
            f"- {item['sample_id']} {item['process']} ID {item['blank_id']} {item['label']} {item.get('error_kind', '')}: "
            f"answer `{item['answer']}`; output `{pred}`"
        )
    lines.append("")
    return "\n".join(lines)


def standard_answers_flight() -> dict[tuple[str, str], str]:
    from copy import deepcopy

    from domains.aerodynamics.body_air_dataset import build_record as build_body_air
    from domains.aerodynamics.dataset_generate import build_record as build_ground_body
    from domains.aerodynamics.ground_track_air_dataset import build_record as build_track_air
    from flight_frame_model import STANDARD_SPEC

    sample = {
        "id": "standard",
        "title": "Unified default",
        "change": "Unified default.",
        "spec": deepcopy(STANDARD_SPEC),
    }
    found = {}
    for build in (build_ground_body, build_track_air, build_body_air):
        record = build(sample)
        for item in record["questions"]:
            found[(item["process"], item["id"])] = item["answer"]
            local_id = item.get("local_id")
            if local_id and local_id != item["id"]:
                found[(item["process"], local_id)] = item["answer"]
    return found


def _basic_vision_data(data: Path | None) -> bool:
    if data is None:
        return False
    parts = Path(data).as_posix().split("/")
    return "basic" in parts and "vision" in parts


def standard_answers_cv(data: Path | None = None) -> dict[tuple[str, str], str]:
    if _basic_vision_data(data):
        from domains.computer_vision.cv_basic_generate import build
        from domains.computer_vision.cv_generate import STANDARD_SPEC

        record = build(STANDARD_SPEC)
    else:
        from domains.computer_vision.cv_generate import STANDARD_SPEC, generate

        record = generate(STANDARD_SPEC)
    found = {}
    for item in record["questions"]:
        found[(item["process"], item["id"])] = item["answer"]
        local_id = item.get("local_id")
        if local_id and local_id != item["id"]:
            found[(item["process"], local_id)] = item["answer"]
    if data is not None and "reconstruction" in Path(data).as_posix() and "vision" in Path(data).as_posix():
        found.update({
            ("pure stereo pose", blank_id): answer
            for (process, blank_id), answer in list(found.items())
            if process == "stereo pose"
        })
    return found


def standard_answers_fm_ctrl(data: Path | None = None) -> dict[tuple[str, str], str]:
    if data is not None and "synced" in Path(data).name:
        from domains.flight_ctrl.fm_ctrl_generate import build_standard_synced
        record = build_standard_synced()
    else:
        from domains.flight_ctrl.fm_ctrl_generate import build
        record = build(25)
    found = {}
    for item in record["questions"]:
        found[(item["process"], item["id"])] = item["answer"]
        local_id = item.get("local_id")
        if local_id and local_id != item["id"]:
            found[(item["process"], local_id)] = item["answer"]
    return found


def standard_for(domain: str, data: Path | None = None) -> dict[tuple[str, str], str]:
    posix = "" if data is None else Path(data).as_posix()
    if "flight_ctrl" in posix or "reconstruction/flight" in posix or "flight/reconstruction" in posix:
        return standard_answers_fm_ctrl(data)
    if domain in {"flight", "cv", "robotics"} and _uses_manuscript(domain, data):
        from domains.eval_manuscripts import manuscript_standard

        return manuscript_standard(domain)
    if domain == "flight":
        return standard_answers_flight()
    if domain == "cv":
        return standard_answers_cv(data)
    if domain == "robotics":
        from domains.eval_manuscripts import manuscript_standard

        return manuscript_standard(domain)
    return standard_answers()


def _uses_manuscript(domain: str, data: Path | None) -> bool:
    """True when this run reads the current full review set, not a legacy jsonl."""
    if domain == "robotics":
        return True
    if data is None:
        return True
    path = Path(data)
    if path.suffix == ".jsonl":
        return False
    return True


def standard_answers() -> dict[tuple[str, str], str]:
    from domains.wind_tunnel.dataset_generate import build_record
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

    record = build_record({
        "id": "standard",
        "title": "Unified default",
        "change": "Unified default.",
        "spec": STANDARD_SPEC,
    })
    found = {}
    for item in record["questions"]:
        found[(item["process"], item["id"])] = item["answer"]
        local_id = item.get("local_id")
        if local_id and local_id != item["id"]:
            found[(item["process"], local_id)] = item["answer"]
    return found


def option_index(path: Path) -> dict[tuple[str, str, str], list[str]]:
    found = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for item in row["questions"]:
                options = list(item.get("options") or [])
                found[(row["id"], item["process"], item["id"])] = options
                local_id = item.get("local_id")
                if local_id and local_id != item["id"]:
                    found[(row["id"], item["process"], local_id)] = options
    return found


def apply_scores(rows: list[dict], standard: dict, options: dict) -> list[dict]:
    from cloze_score import judge

    for item in rows:
        if "blanks" not in item:
            verdict = judge(
                item.get("prediction", ""),
                item.get("answer", ""),
                item.get("format_code") or "",
                options.get((item.get("sample_id"), item.get("process"), item.get("blank_id"))),
                standard.get((item.get("process"), item.get("blank_id"))),
            )
            item.update(verdict)
            continue
        for blank in item["blanks"]:
            key = (item["sample_id"], item["process"], blank["blank_id"])
            if key in options:
                blank["options"] = options[key]
        if item.get("response"):
            item["blanks"] = _score_blanks(item["response"], item["blanks"], standard, item["process"])
        else:
            from cloze_score import answer_letter

            for blank in item["blanks"]:
                blank["answer"] = answer_letter(blank.get("answer", ""), blank.get("options") or [])
                blank.update(judge(
                    blank.get("prediction", ""),
                    blank["answer"],
                    blank.get("format_code") or "",
                    blank.get("options"),
                    standard.get((item["process"], blank["blank_id"])),
                ))
    return rows


def _preface(
    rows: list[dict],
    model: str,
    thinking: str,
    temperature: float,
    runs: int,
    seed: int,
    max_tokens: int,
    condition: str = "A",
) -> list[str]:
    role = "main evaluation" if model == "deepseek-v4-pro" and thinking == "enabled" else "control"
    usd = 0.0
    known = 0
    hit = miss = output = 0
    for item in rows:
        cost = item.get("cost") or {}
        if isinstance(cost, dict) and isinstance(cost.get("usd"), (int, float)):
            usd += float(cost["usd"])
            known += 1
        usage = item.get("usage") or {}
        if isinstance(usage, dict):
            hit += int(usage.get("prompt_cache_hit_tokens") or 0)
            miss += int(usage.get("prompt_cache_miss_tokens") or 0)
            output += int(usage.get("completion_tokens") or 0)
    versions = sorted({item.get("model_version") or "" for item in rows if item.get("model_version")})
    documented = sorted({item.get("documented_model_version") or "" for item in rows if item.get("documented_model_version")})
    dates = sorted({(item.get("started_at") or "")[:10] for item in rows if item.get("started_at")})
    lines = [
        f"This run is the {role}: model {model}, thinking {thinking}, condition {condition_label(condition)}.",
        "The condition name states how the text is injected."
        + " ".join(f"{name}: {gloss}." for name, gloss in CONDITION_GLOSS.items())
        + " The old letters A, B, C, D, E, and F can still be entered, in that order, for these six names.",
        (
            "The main evaluation is still deepseek-v4-pro with thinking on."
            + (
                "This deepseek-flash run also has thinking on, to see how a personalized definition affects accuracy."
                if thinking == "enabled"
                else "With thinking off, deepseek-flash stands only for the non-thinking mode and is not the full capability of V4."
            )
        ),
        "Temperature has no effect when thinking is on. The official V4 API ignores seed, so repeated runs are what provide variance. The interval below is a sample bootstrap.",
    ]
    if max_tokens:
        stop_note = "End-to-end does not stop at a blank line." if condition in _END_TO_END else "With thinking off, generation also stops at a blank line."
        lines.append(
            f"Temperature {temperature}, repeated runs {runs}, request seed {seed}, max_tokens {max_tokens}. "
            f"That ceiling includes the chain of thought and is kept when thinking is off. {stop_note}"
        )
    else:
        lines.append(f"Temperature {temperature}. This batch did not set max_tokens, stop, or seed.")
    retries = sum(max(0, int(item.get("attempts") or 1) - 1) for item in rows)
    if known:
        lines.append(
            f"Calls with recorded usage: {known}. Cache hit {hit}, cache miss {miss}, output {output}. "
            f"Estimated from the DeepSeek price list at {usd:.4f} USD, which is not the bill. "
            f"Retries: {retries}. Usage includes only the successful return. Timed-out and 429 attempts are not in these tokens, so the billed call count is higher."
        )
    else:
        lines.append(
            "These records have no usage and no cost, so the cost cannot be checked. "
            "Timeout and 429 retries are also absent from the token estimate."
        )
    if versions or documented or dates:
        lines.append(
            "Server model names: " + (", ".join(versions) or "none")
            + ". Documented versions: " + (", ".join(documented) or "none")
            + ". Call dates: " + (", ".join(dates) or "none") + "."
        )
    return lines


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_raw_samples(out_dir: Path, rows: list[dict]) -> None:
    """One file per complete sample, containing that sample's raw model replies."""
    process_order = ["wind frame to body-axis frame", "balance frame to body-axis frame", "balance frame to wind frame", "moment reference point"]
    grouped: dict[str, list[dict]] = {}
    for item in rows:
        if "blank_ids" not in item:
            continue
        grouped.setdefault(item["sample_id"], []).append(item)
    target = out_dir / "raw_samples"
    target.mkdir(parents=True, exist_ok=True)
    for sample_id, items in grouped.items():
        def _sort_blank(item: dict) -> int:
            token = str(item["blank_ids"][0])
            return int(token) if token.isdigit() else 10**9

        items.sort(key=lambda item: (
            process_order.index(item["process"]) if item["process"] in process_order else len(process_order),
            _sort_blank(item),
            item.get("run", 0),
        ))
        head = items[0]
        lines = [
            f"# {sample_id} raw output",
            "",
            f"Model {head.get('model', '')}, thinking {head.get('thinking', '')}, condition {condition_label(head.get('condition', ''))}.",
            "The model's raw replies are kept below in subprocess and sentence order, without rewriting.",
            "",
        ]
        for item in items:
            if len(item["blank_ids"]) > 8:
                lines.append(f"## {item['process']} all {len(item['blank_ids'])} blanks")
            else:
                lines.append(f"## {item['process']} ID {', '.join(item['blank_ids'])}")
            lines.append("")
            if item.get("unanswered"):
                lines.append(f"The model did not answer: the output was cut off at {item.get('max_tokens') or THINKING_MAX_TOKENS} tokens.")
                lines.append("")
            elif item.get("error"):
                lines.append(f"Call failed: {item['error']}")
                lines.append("")
            lines.append(item.get("response") or "")
            lines.append("")
        (target / f"{sample_id}.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _ordered_records(rows: list[dict]) -> list[dict]:
    found = {}
    order = []
    for item in rows:
        key = _record_key(item)
        if key not in found:
            order.append(key)
        found[key] = item
    return [found[key] for key in order]


def _execute_chain(
    chain: dict,
    condition: str,
    run: int,
    done: dict,
    client,
    model: str,
    temperature: float,
    thinking: str,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
    write,
    repair_ids: set[str] | None = None,
) -> int:
    memory: dict[str, str] = {}
    held_reasoning = ""
    rerun = False
    ran = 0
    repair_ids = repair_ids or set()
    for sentence in chain["sentences"]:
        key = (condition, chain["sample_id"], chain["process"], ",".join(sentence["blank_ids"]), run)
        saved = done.get(key)
        if saved is not None and not rerun:
            for blank in saved.get("blanks") or []:
                _remember(memory, blank, repair_ids)
            continue
        rerun = True
        memory_arg = memory if condition in _SEQUENTIAL else None
        job = job_for(
            chain, sentence, condition, memory_arg, run, seed + run, max_tokens, stop, standard,
        )
        if held_reasoning and len(held_reasoning) > 40 and held_reasoning in job["prompt"]:
            raise RuntimeError("The chain of thought was written into the next sentence. Multi-turn keeps only the answer text.")
        result = score_one(client, model, temperature, thinking, job)
        held_reasoning = result.get("reasoning") or ""
        for blank in result["blanks"]:
            _remember(memory, blank, repair_ids)
        if repair_ids:
            result["gold_repaired"] = sorted(repair_ids)
        write(result)
        done[key] = result
        ran += 1
    return ran


def _chat_snapshot(messages: list[dict]) -> list[tuple[str, str]]:
    return [(item["role"], item["content"]) for item in messages]


def _execute_session(
    chain: dict,
    condition: str,
    run: int,
    done: dict,
    client,
    model: str,
    temperature: float,
    thinking: str,
    seed: int,
    max_tokens: int,
    stop: list[str] | None,
    standard: dict,
    write,
) -> int:
    """One subprocess, one chat. The definition is only in the system message.

    Matrix probes fork a copy of the chat at their anchor. The copy is scored
    and discarded. The main chat then continues with the next manuscript turn.
    """
    from domains.wind_tunnel.session_probe import (
        PROBE_LEAD,
        _gold_map,
        answers_to_clear,
        clear_turn,
        forks_after,
        make_branch,
    )

    system = _session_system(chain)
    messages: list[dict] = [{"role": "system", "content": system}]
    gold_by_id = _gold_map(chain)
    committed: list[dict] = []
    rerun = False
    ran = 0
    previous = None

    def commit(question: str, reply: str, sentence: dict) -> None:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": reply})
        committed.append({
            "blank_ids": [str(item) for item in sentence["blank_ids"]],
            "assistant_index": len(messages) - 1,
        })

    def wipe(sentence: dict) -> None:
        clear_ids = answers_to_clear(chain, sentence["blank_ids"])
        if clear_ids:
            clear_turn(messages, committed[-1], clear_ids, gold_by_id)

    def ask_probes(sentence: dict, turn: int) -> None:
        for spec in forks_after(chain, sentence["blank_ids"]):
            probe_key = (
                condition, chain["sample_id"], chain["process"], spec["record_id"], run,
            )
            if done.get(probe_key) is not None and not rerun:
                continue
            before = _chat_snapshot(messages)
            try:
                branch = make_branch(chain, messages, committed, spec)
            except (KeyError, RuntimeError) as exc:
                reason = str(exc)
                if "has not seen the matrix name" not in reason and "the problem text does not contain" not in reason:
                    raise
                print(f"probe skip {spec['probe_id']}: {reason}", flush=True)
                continue
            if _chat_snapshot(messages) != before:
                raise RuntimeError(f"{spec['probe_id']} was written back into the main session")
            job = {
                "sample_id": chain["sample_id"],
                "tier": chain["tier"],
                "process": chain["process"],
                "group": "probe",
                "blank_ids": [spec["record_id"]],
                "blanks": [spec["blank"]],
                "condition": condition,
                "run": run,
                "seed": seed + run,
                "max_tokens": max_tokens,
                "stop": stop,
                "standard": standard,
                "messages": branch,
                "prompt": transcript(branch),
                "session": chain["process"],
                "turn": turn,
            }
            result = score_one(client, model, temperature, thinking, job)
            if _chat_snapshot(messages) != before:
                raise RuntimeError(f"The answer of {spec['probe_id']} was written back into the main session")
            if PROBE_LEAD in "\n".join(content for _, content in before):
                raise RuntimeError(f"The stem of {spec['probe_id']} entered the main session")
            result["branch"] = "probe"
            result["probe_id"] = spec["probe_id"]
            result["probe_kind"] = spec["kind"]
            result["probe_anchor"] = spec["anchor"]
            result["probe_target"] = spec["target"]
            result["session"] = chain["process"]
            result["turn"] = turn
            write(result)
            done[probe_key] = result

    for turn, sentence in enumerate(chain["sentences"]):
        key = (condition, chain["sample_id"], chain["process"], ",".join(sentence["blank_ids"]), run)
        question = _session_question(chain, sentence, previous)
        if PROBE_LEAD in question:
            raise RuntimeError(f"The main question of {chain['process']} carried a probe stem")
        history = [*messages, {"role": "user", "content": question}]
        _guard_session(history, chain["declaration"])
        saved = done.get(key)
        if saved is not None and not rerun:
            reply = gold_reply(sentence["blanks"]) if condition == SESSION_GOLD else (saved.get("response") or "")
            commit(question, reply, sentence)
            wipe(sentence)
            ask_probes(sentence, turn)
            previous = sentence
            continue
        rerun = True
        job = job_for(
            chain, sentence, condition, None, run, seed + run, max_tokens, stop, standard,
        )
        job["messages"] = history
        job["prompt"] = transcript(history)
        job["turn"] = turn
        result = score_one(client, model, temperature, thinking, job)
        reasoning = result.get("reasoning") or ""
        if reasoning and len(reasoning) > 40 and reasoning in question:
            raise RuntimeError("The chain of thought was written into the next sentence. The session keeps only the answer text.")
        reply = gold_reply(sentence["blanks"]) if condition == SESSION_GOLD else (result.get("response") or "")
        commit(question, reply, sentence)
        result["session"] = chain["process"]
        result["turn"] = turn
        write(result)
        done[key] = result
        wipe(sentence)
        ask_probes(sentence, turn)
        previous = sentence
        ran += 1
    return ran


def _write_run_checkpoint(out: Path, payload: dict, panel: Path | None = None) -> None:
    (out / "checkpoint.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    panel = panel or (ROOT / "results" / "panel_checkpoint.json")
    events = []
    if panel.is_file():
        loaded = json.loads(panel.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            events = list(loaded.get("events") or [])
    events.append(payload)
    panel.write_text(json.dumps({
        "updated": payload.get("at"),
        "latest": payload,
        "events": events[-80:],
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _resume_command(model: str, thinking: str, condition: str, out: Path, args=None) -> str:
    flag = condition
    command = (
        f"python cloze_eval.py --model {model} --thinking {thinking} "
        f"--condition {flag} --runs 1 --workers 4 --out {out.as_posix()}"
    )
    if args is not None and getattr(args, "domain", "wind_tunnel") == "flight":
        command += (
            f" --domain flight --data {Path(args.data).as_posix()}"
            f" --ids-file {Path(args.ids_file).as_posix()}"
            f" --max-tokens {args.max_tokens}"
        )
    return command


def _publish_and_stop(
    halt: RunStop,
    args,
    done_path: Path,
    standard: dict,
    model: str,
    thinking: str,
    condition: str,
    max_tokens: int,
    base_url: str,
    ids: list,
    sentence_count: int,
    sequential: bool,
    chains: list,
    jobs: list,
) -> None:
    stored = _ordered_records(_load_jsonl(done_path)) if done_path.is_file() else []
    probes = [item for item in stored if item.get("branch") == "probe"]
    if sequential:
        wanted = {
            (condition, chain["sample_id"], chain["process"], ",".join(sentence["blank_ids"]), chain["run"])
            for chain in chains
            for sentence in chain["sentences"]
        }
    else:
        wanted = {_record_key(job) for job in jobs}
    stored = [
        item for item in stored
        if item.get("branch") != "probe" and _record_key(item) in wanted and not item.get("error")
    ]
    if probes:
        _write_jsonl(args.out / "probes.jsonl", probes)
    if stored:
        stored = apply_scores(stored, standard, option_index(args.data))
        _write_jsonl(done_path, stored)
        flat = flatten(stored)
        preface = _preface(
            stored, model, thinking, args.temperature, args.runs, args.seed, max_tokens, condition,
        )
        report = summarize(stored, preface)
        note = (
            "\n\n> This round stopped on quota or rate limiting. The scores above cover only the sentences that succeeded. "
            "The failed sentence was not written, so the original command resumes from that sentence.\n"
        )
        (args.out / "report.md").write_text(report + note, encoding="utf-8")
        exact = sum(item["exact"] for item in flat)
        normalized = sum(item["normalized"] for item in flat)
        blanks = len(flat)
    else:
        exact = normalized = blanks = 0
    job = halt.job
    payload = {
        "status": "stopped",
        "kind": halt.kind,
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "thinking": thinking,
        "condition": condition,
        "base_url": base_url,
        "out": args.out.as_posix(),
        "sample_id": job.get("sample_id"),
        "process": job.get("process"),
        "blank_ids": job.get("blank_ids"),
        "run": job.get("run", 0),
        "error": str(halt),
        "completed_sentences": len(stored),
        "expected_sentences": sentence_count,
        "completed_blanks": blanks,
        "exact": exact,
        "normalized": normalized,
        "resume_command": _resume_command(model, thinking, condition, args.out, args),
    }
    panel = args.out.parent / "panel_checkpoint.json" if getattr(args, "domain", "wind_tunnel") == "flight" else None
    _write_run_checkpoint(args.out, payload, panel)
    ids_text = ", ".join(str(item) for item in (job.get("blank_ids") or []))
    print(
        f"Stopped: {condition} sample {job.get('sample_id')} process {job.get('process')} "
        f"blanks {ids_text} reason {halt.kind}. Finished {len(stored)}/{sentence_count} sentences.",
        flush=True,
    )
    print(payload["resume_command"], flush=True)
    raise SystemExit(2)


def _load_eval_rows(domain: str, path: Path, wanted: list[str] | None) -> list[dict]:
    if domain == "wind_tunnel":
        if path.suffix != ".jsonl":
            raise SystemExit("The wind-tunnel full-set entry is batch/samples.jsonl.")
        rows = load_samples(path, set(wanted) if wanted is not None else None, strict_graph=False)
        if wanted is not None:
            order = {item: index for index, item in enumerate(wanted)}
            rows.sort(key=lambda row: order[row["id"]])
        return rows
    from domains.eval_manuscripts import load_rows

    return load_rows(domain, path, wanted)


def _row_contexts(domain: str, row: dict) -> list[dict]:
    if row.get("manuscript"):
        return row["chains"]
    if domain == "cv":
        return contexts_for_cv(row)
    if domain == "flight":
        return contexts_for_flight(row)
    return contexts_for(row)


def _check_session_prefix(chain: dict) -> None:
    previous = None
    for sentence in chain["sentences"]:
        current = _process_excerpt(chain, sentence).strip()
        if not current:
            raise RuntimeError(f"{chain['sample_id']} {chain['process']} has an empty sentence.")
        if previous is not None:
            earlier = _close_slots(_process_excerpt(chain, previous), previous).strip()
            if not current.startswith(earlier):
                raise RuntimeError(
                    f"{chain['sample_id']} {chain['process']} has a session prefix that does not line up."
                )
            if not current[len(earlier):].strip():
                raise RuntimeError(
                    f"{chain['sample_id']} {chain['process']} has a next sentence with no new text."
                )
        previous = sentence


def _check_eval_rows(domain: str, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("There are no samples to score.")
    if domain == "wind_tunnel":
        allowed = (0,) + TIERS
        bad = [str(row["id"]) for row in rows if int(row.get("tier") or 0) not in allowed]
        if bad:
            raise SystemExit("The wind-tunnel full set has samples outside 0/25/50/75/100: " + ", ".join(bad[:8]))
        return
    from domains.eval_manuscripts import manuscript_standard

    standard = manuscript_standard(domain)
    missing = []
    for row in rows:
        for chain in row["chains"]:
            _check_session_prefix(chain)
            for sentence in chain["sentences"]:
                for blank_id in sentence["blank_ids"]:
                    key = (chain["process"], str(blank_id))
                    if key not in standard:
                        missing.append(f"{row['id']} {chain['process']} {blank_id}")
    if missing:
        raise SystemExit("The standard manuscript does not have these blanks: " + ", ".join(missing[:12]))


def _print_inventory(domain: str, path: Path, rows: list[dict]) -> None:
    spec = EVAL[domain]
    grouped = [(row, _row_contexts(domain, row)) for row in rows]
    chains = [chain for _row, items in grouped for chain in items]
    sentences = sum(len(chain["sentences"]) for chain in chains)
    blanks = sum(len(sentence["blank_ids"]) for chain in chains for sentence in chain["sentences"])
    tiers = Counter(int(row.get("tier") or 0) for row in rows)
    tier_text = ", ".join(f"{tier}% x {tiers[tier]}" for tier in sorted(tiers))
    print(f"{spec['zh']}  {path}", flush=True)
    print(f"samples {len(rows)}  subprocesses {len(chains)}  sentences {sentences}  blanks {blanks}", flush=True)
    print(f"tiers {tier_text}", flush=True)
    mismatched = sum(int(row.get("graph_label_mismatches") or 0) for row in rows)
    if mismatched:
        print(
            f"{mismatched} blanks have a dependency-graph label that does not match the manuscript. Scoring uses the label and answer on the manuscript.",
            flush=True,
        )
    for row, items in grouped:
        row_blanks = sum(len(sentence["blank_ids"]) for chain in items for sentence in chain["sentences"])
        print(
            f"  {row['id']}  {int(row.get('tier') or 0)}%  subprocesses {len(items)}  blanks {row_blanks}",
            flush=True,
        )


def main() -> None:
    from cloze_score import _check_scoring

    _check_scoring()
    _check_parse()
    parser = argparse.ArgumentParser(description="Score cloze blanks with DeepSeek.")
    parser.add_argument("--domain", choices=EVAL_DOMAINS, default="wind_tunnel")
    parser.add_argument("--data", type=Path, default=None, help="The default is that domain's full data.")
    parser.add_argument("--ids-file", type=Path, default=None, help="The default scores every sample.")
    parser.add_argument("--tier", type=int, choices=TIERS, default=None, help="Score only this tier. The default scores every tier.")
    parser.add_argument("--inventory", action="store_true", help="Only count samples, subprocesses, and blanks. Do not call the model.")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "full")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N sentences. 0 means all.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="", help="Override the model in the environment. The main evaluation uses deepseek-v4-pro.")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 means 8192 when thinking is off. Thinking on is fixed at 65536. If the chain of thought uses up the ceiling, the item is scored wrong and the ceiling is not raised. The ceiling includes the chain of thought.",
    )
    parser.add_argument(
        "--condition",
        default="A",
        help="gold-trunc, self-trunc, gold-keep, self-keep, proc-e2e, final-e2e, "
             "session-gold, session-model."
             "gold/self chooses the gold answer or the model's own output before the blank. trunc/keep truncates at the period or keeps the later text. "
             "proc-e2e is one call per subprocess. final-e2e asks only the final question. "
             "session-gold and session-model ask one item at a time in one chat, and the definition is written only in the system message. "
             "The old letters A, B, C, D, E, and F still work.",
    )
    parser.add_argument("--rescore", type=Path, default=None, help="Rescore the saved replies. Do not call the model.")
    parser.add_argument(
        "--gold-nodes",
        default="",
        help="Under an own-prefix condition, inject the gold answer after these item ids. The model is still scored on its original answer. Comma-separated, for example 24,47,48.",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    spec = EVAL[args.domain]
    if args.data is None:
        args.data = spec["data"]
    if not Path(args.data).exists():
        raise SystemExit(f"Full data not found: {args.data}")
    wanted = None
    if args.ids_file is not None:
        picked = json.loads(args.ids_file.read_text(encoding="utf-8"))
        ids = picked["ids"] if isinstance(picked, dict) else picked
        wanted = [str(item) for item in ids]
    rows: list[dict] = []
    if args.rescore is None:
        try:
            rows = _load_eval_rows(args.domain, args.data, wanted)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.tier is not None:
            rows = [row for row in rows if int(row.get("tier") or -1) == args.tier]
            if not rows:
                raise SystemExit(f"{spec['zh']} full data has no {args.tier}% tier.")
        try:
            _check_eval_rows(args.domain, rows)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        if args.inventory:
            _print_inventory(args.domain, args.data, rows)
            return

    load_env_file(ROOT / ".env")
    _key, base_url, env_model = resolve_api_config()
    model = args.model.strip() or env_model
    thinking = args.thinking
    repair_ids = {
        part.strip().removeprefix("ID").removeprefix("id")
        for part in args.gold_nodes.split(",")
        if part.strip()
    }
    max_tokens = _resolve_max_tokens(args.max_tokens, thinking)
    workers = _resolve_workers(args.workers)
    stop = None if thinking == "enabled" else ["\n\n"]
    if args.rescore is None:
        print(
            f"Condition: {condition_label(args.condition)} model={model} thinking={thinking} temperature={args.temperature} "
            f"temperature_applied={thinking != 'enabled'} runs={args.runs} seed={args.seed} max_tokens={max_tokens}"
            + (f" gold_nodes={','.join(sorted(repair_ids, key=int))}" if repair_ids else ""),
            flush=True,
        )
    if args.rescore is None and (model != "deepseek-v4-pro" or thinking != "enabled"):
        print("This is not the main evaluation. The main evaluation is deepseek-v4-pro with --thinking enabled.", flush=True)

    if args.rescore is not None:
        target = args.rescore
        stored = _load_jsonl(target / "records.jsonl")
        print(f"Rescoring {len(stored)} records", flush=True)
        standard = standard_for(args.domain, args.data)
        options = option_index(args.data)
        stored = apply_scores(stored, standard, options)
        legacy = bool(stored) and "blank_ids" not in stored[0]
        recorded_model = stored[0].get("model") or model
        recorded_thinking = stored[0].get("thinking", thinking)
        recorded_temperature = stored[0].get("temperature", args.temperature)
        recorded_tokens = stored[0].get("max_tokens") or 0
        recorded_condition = stored[0].get("condition", "A")
        preface = _preface(
            stored, recorded_model, recorded_thinking, recorded_temperature,
            args.runs, args.seed, recorded_tokens, recorded_condition,
        )
        report = summarize(stored, preface)
        if legacy:
            (target / "report_normalized.md").write_text(report, encoding="utf-8")
            print(f"Wrote {target / 'report_normalized.md'}")
        else:
            _write_jsonl(target / "records.jsonl", stored)
            (target / "report.md").write_text(report, encoding="utf-8")
            meta_path = target / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            flat = flatten(stored)
            meta.update({
                "exact": sum(item["exact"] for item in flat),
                "normalized": sum(item["normalized"] for item in flat),
                "blanks": len(flat),
                "rescore": "Strict and normalized scores were recomputed from the saved replies. The model was not called again.",
            })
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(report.split("## By definition tier")[0])
        return

    print(
        f"Data: {spec['zh']} {args.data} samples {len(rows)}",
        flush=True,
    )
    print("Building the unified-default answers, used to detect an unused definition.", flush=True)
    standard = standard_for(args.domain, args.data)

    condition = canonical_condition(args.condition)
    if repair_ids and condition not in _SEQUENTIAL:
        raise SystemExit("--gold-nodes is only for an own prefix (self-trunc / self-keep). A gold prefix already injects the correct answer.")
    if not condition:
        raise SystemExit(
            "The condition must be gold-trunc, self-trunc, gold-keep, self-keep, proc-e2e, final-e2e, "
            "session-gold, session-model. "
            "gold/self chooses the gold answer or the model's own output before the blank, and trunc/keep truncates at the period or keeps the later text. "
            "proc-e2e is one call per subprocess, and final-e2e asks only the final question. "
            "session-gold and session-model ask one item at a time in one chat, and the definition is written only in the system message. "
            "The old letters A, B, C, D, E, and F still work. The old spellings C' and C'' mean gold-keep and self-keep."
        )
    if condition == FINAL_E2E and any(row.get("manuscript") for row in rows):
        raise SystemExit("The full review manuscript is scored sentence by sentence inside each subprocess. Use session-model, a sentence condition, or proc-e2e.")
    if condition == FINAL_E2E and args.domain not in {"flight", "wind_tunnel", "cv"}:
        raise SystemExit("Condition final-e2e is for the older flight-mechanics, wind-tunnel, and computer-vision question lists.")
    if condition in _END_TO_END:
        stop = None
    if condition == PROC_E2E:
        print(f"{condition_label(condition)}: max_tokens={max_tokens}, and generation does not stop at a blank line.", flush=True)
    if condition == FINAL_E2E:
        print(
            f"{condition_label(condition)}: max_tokens={max_tokens}.",
            flush=True,
        )
        final_questions = load_final_questions() if args.domain == "flight" else None
    sequential = condition in _SEQUENTIAL or condition in _SESSION
    chains = []
    jobs = []
    if condition == FINAL_E2E:
        for run in range(args.runs):
            for row in rows:
                if args.domain == "flight" and any(
                    item.get("process") == "body relative to ground" for item in row["questions"]
                ):
                    jobs.extend(wind_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard, _CTRL_FINALS,
                    ))
                elif args.domain == "flight":
                    jobs.extend(final_question_jobs(
                        row, final_questions, run, args.seed + run, max_tokens, stop, standard,
                    ))
                elif args.domain == "cv" and row["questions"][0].get("process") == _BASIC_PROCESS:
                    jobs.extend(cv_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard,
                        finals=_BASIC_FINALS, process=_BASIC_PROCESS,
                    ))
                elif args.domain == "cv" and row["questions"][0].get("process") == "pure stereo pose":
                    from domains.computer_vision.cv_vision_context import rewrite
                    finals = tuple(
                        (slots, rewrite(question))
                        for slots, question in _CV_FINALS
                    )
                    jobs.extend(cv_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard,
                        finals=finals, process="pure stereo pose",
                    ))
                elif args.domain == "cv":
                    jobs.extend(cv_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard,
                    ))
                else:
                    jobs.extend(wind_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard,
                    ))
        if args.limit:
            jobs = jobs[:args.limit]
    else:
        for run in range(args.runs):
            for row in rows:
                for chain in _row_contexts(args.domain, row):
                    chain = dict(chain)
                    chain["run"] = run
                    chains.append(chain)
        if args.limit:
            kept = []
            remaining = args.limit
            for chain in chains:
                if remaining <= 0:
                    break
                sentences = chain["sentences"][:remaining]
                remaining -= len(sentences)
                trimmed = dict(chain)
                trimmed["sentences"] = sentences
                kept.append(trimmed)
            chains = kept
        if condition == PROC_E2E:
            for chain in chains:
                jobs.append(end_to_end_job(
                    chain, chain["run"], args.seed + chain["run"], max_tokens, stop, standard,
                ))
        elif not sequential:
            for chain in chains:
                for sentence in chain["sentences"]:
                    jobs.append(job_for(
                        chain, sentence, condition, None, chain["run"],
                        args.seed + chain["run"], max_tokens, stop, standard,
                    ))
    args.out.mkdir(parents=True, exist_ok=True)
    done_path = args.out / "records.jsonl"
    meta_path = args.out / "meta.json"
    if meta_path.is_file() and done_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("model") not in (None, model) or old.get("thinking") not in (None, thinking):
            raise SystemExit(
                f"Directory {args.out} already has records for model={old.get('model')} thinking={old.get('thinking')}. "
                "Choose another --out so this condition is not mixed with that one."
            )
        if old.get("condition", "A") != condition:
            raise SystemExit(
                f"Directory {args.out} already has records for condition {old.get('condition', 'A')}. "
                "Choose another --out so this condition is not mixed with that one."
            )
        if set(old.get("gold_repaired") or []) != repair_ids:
            raise SystemExit(
                f"Directory {args.out} already has records for gold_repaired={old.get('gold_repaired') or []}. "
                "Choose another --out so this node repair is not mixed with the other one."
            )
    done = {}
    legacy = False
    probe_path = args.out / "probes.jsonl"
    if probe_path.is_file():
        for item in _load_jsonl(probe_path):
            done[_record_key(item)] = item
    if done_path.is_file():
        for item in _load_jsonl(done_path):
            if "blank_ids" not in item:
                legacy = True
            done[_record_key(item)] = item
    if legacy:
        raise SystemExit("This result directory records one blank at a time. For whole-sentence scoring, choose another --out directory so the old report is not overwritten.")
    if condition == FINAL_E2E:
        sentence_count = len(jobs)
        blank_count = sum(len(job["blank_ids"]) for job in jobs)
    else:
        sentence_count = sum(len(chain["sentences"]) for chain in chains)
        blank_count = sum(len(sentence["blank_ids"]) for chain in chains for sentence in chain["sentences"])
    def _chain_pending(chain: dict) -> int:
        rerun = False
        count = 0
        for sentence in chain["sentences"]:
            key = (condition, chain["sample_id"], chain["process"], ",".join(sentence["blank_ids"]), chain["run"])
            if key in done and not rerun:
                continue
            rerun = True
            count += 1
        return count

    if sequential:
        pending = sum(_chain_pending(chain) for chain in chains)
    else:
        pending = sum(_record_key(job) not in done for job in jobs)
    progress_total = len(jobs) if condition in _END_TO_END else sentence_count
    print(
        f"sentences={sentence_count} blanks={blank_count} pending={pending} "
        f"calls={progress_total} chains={len(chains)}",
        flush=True,
    )

    halt: RunStop | None = None
    if pending:
        client = make_client(timeout=1800.0 if thinking == "enabled" else 600.0)
        lock = threading.Lock()
        progress = {"finished": progress_total - pending}
        with done_path.open("a", encoding="utf-8") as handle:
            def write(result: dict) -> None:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()

            def _note_progress() -> None:
                progress["finished"] += 1
                current = progress["finished"]
                if current == 1 or current % 20 == 0 or current == progress_total:
                    print(f"scored {current}/{progress_total}", flush=True)

            if sequential:
                def run_one(chain: dict) -> int:
                    def locked_write(result: dict) -> None:
                        with lock:
                            write(result)
                            if result.get("branch") == "probe":
                                print(
                                    f"probe {result.get('probe_id')} {result.get('probe_kind')}",
                                    flush=True,
                                )
                                return
                            _note_progress()
                    if condition in _SESSION:
                        return _execute_session(
                            chain, condition, chain["run"], done, client, model, args.temperature,
                            thinking, args.seed, max_tokens, stop, standard, locked_write,
                        )
                    return _execute_chain(
                        chain, condition, chain["run"], done, client, model, args.temperature,
                        thinking, args.seed, max_tokens, stop, standard, locked_write, repair_ids,
                    )

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(run_one, chain) for chain in chains]
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except RunStop as exc:
                            halt = exc
                            pool.shutdown(wait=True, cancel_futures=True)
                            break
            else:
                pending_jobs = [job for job in jobs if _record_key(job) not in done]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [
                        pool.submit(score_one, client, model, args.temperature, thinking, job)
                        for job in pending_jobs
                    ]
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                        except RunStop as exc:
                            halt = exc
                            pool.shutdown(wait=True, cancel_futures=True)
                            break
                        with lock:
                            write(result)
                            _note_progress()
    if halt is not None:
        _publish_and_stop(
            halt, args, done_path, standard, model, thinking, condition,
            max_tokens, base_url, ids, sentence_count, sequential, chains, jobs,
        )

    stored = _ordered_records(_load_jsonl(done_path)) if done_path.is_file() else []
    probes = [item for item in stored if item.get("branch") == "probe"]
    if sequential:
        wanted = {
            (condition, chain["sample_id"], chain["process"], ",".join(sentence["blank_ids"]), chain["run"])
            for chain in chains
            for sentence in chain["sentences"]
        }
    else:
        wanted = {_record_key(job) for job in jobs}
    stored = [item for item in stored if item.get("branch") != "probe" and _record_key(item) in wanted]
    if probes:
        _write_jsonl(args.out / "probes.jsonl", probes)
    stored = apply_scores(stored, standard, option_index(args.data))
    _write_jsonl(done_path, stored)
    flat = flatten(stored)
    preface = _preface(
        stored, model, thinking, args.temperature, args.runs, args.seed, max_tokens, condition,
    )
    if probes and condition in _SESSION:
        preface.append(
            f"{len(probes)} matrix probes were written to probes.jsonl. Only the shallow probes and the deep re-derivation probes are kept. "
            "Factor matrices and the total matrix are scored in the main session and then replaced by placeholders, so they do not enter later questions."
        )
    report = summarize(stored, preface)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    write_raw_samples(args.out, stored)
    meta = {
        "model": model,
        "base_url": base_url,
        "thinking": thinking,
        "condition": condition,
        "temperature": args.temperature,
        "temperature_applied": thinking != "enabled",
        "runs": args.runs,
        "seed": args.seed,
        "seed_note": "The official DeepSeek V4 API ignores seed.",
        "max_tokens": max_tokens,
        "samples": ids,
        "sentences": len(stored),
        "probes": len(probes),
        "blanks": len(flat),
        "exact": sum(item["exact"] for item in flat),
        "normalized": sum(item["normalized"] for item in flat),
        "role": "main" if model == "deepseek-v4-pro" and thinking == "enabled" else "control",
        "domain": args.domain,
        "gold_repaired": sorted(repair_ids),
    }
    meta["status"] = "complete"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    panel = args.out.parent / "panel_checkpoint.json" if args.domain == "flight" else None
    _write_run_checkpoint(args.out, {
        "status": "complete",
        "kind": "",
        "at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": model,
        "thinking": thinking,
        "condition": condition,
        "base_url": base_url,
        "out": args.out.as_posix(),
        "completed_sentences": len(stored),
        "expected_sentences": sentence_count,
        "completed_blanks": len(flat),
        "exact": meta["exact"],
        "normalized": meta["normalized"],
        "resume_command": "",
    }, panel)
    if args.domain == "wind_tunnel":
        from domains.wind_tunnel.error_graph import write_error_chains

        chain_path = write_error_chains(stored, args.out)
        if chain_path is not None:
            print(f"Wrote {chain_path}", flush=True)
        from domains.wind_tunnel.run_diagnosis import write_run_diagnosis

        diagnosis_path = write_run_diagnosis(stored, args.out)
        if diagnosis_path is not None:
            print(f"Wrote {diagnosis_path}", flush=True)
    print(report.split("## By definition tier")[0])


if __name__ == "__main__":
    main()
