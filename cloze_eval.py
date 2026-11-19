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
six names. Blank numbers run from 1 through 194
across the four subprocesses.
Strict equality is the stripped reply. The normalized column also accepts
spacing, symbol and LaTeX differences, an exact option after that cleanup,
and sympy equivalence for expressions and matrices.

Run the formal condition::

    python cloze_eval.py --model deepseek-v4-pro --thinking enabled --runs 3 --out results/pro_thinking

Run the flash, thinking-off control::

    python cloze_eval.py --model deepseek-flash --thinking disabled --runs 1 --out results/flash_nothink
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
from domain_layout import SETS
from domains.wind_tunnel.dataset_generate import _injection, _injection_end_to_end, _local_id, _mechanism_phrases, _sentence_groups
from semi_auto_eval import call_model, load_env_file, make_client, resolve_api_config

ROOT = Path(__file__).resolve().parent
SHANGHAI = timezone(timedelta(hours=8))

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
    GOLD_TRUNC: "标准前缀，句号截断",
    SELF_TRUNC: "自身前缀，句号截断",
    GOLD_KEEP: "标准前缀，保留后文",
    SELF_KEEP: "自身前缀，保留后文",
    PROC_E2E: "子过程一次，全部空待填",
    FINAL_E2E: "只问最终问题，不给分步填空",
    SESSION_GOLD: "单次定义，会话逐题，前文用金标",
    SESSION_MODEL: "单次定义，会话逐题，前文用模型回答",
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
    return f"{resolved}（{gloss}）" if gloss else str(code)

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
# 国务院办公厅 2026 年放假安排。只用于判断 DeepSeek 的高峰/低谷，不替代官方账单。
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


def load_samples(path: Path, wanted: set[str] | None) -> list[dict]:
    from domains.wind_tunnel.error_graph import annotate_sample

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if wanted is None or row["id"] in wanted:
                rows.append(annotate_sample(row))
    if wanted is not None:
        found = {row["id"] for row in rows}
        missing = wanted - found
        if missing:
            raise SystemExit("样本里没有这些编号：" + "、".join(sorted(missing)))
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
            # 批量样本只存连续公开编号。模板槽位仍是各子过程内的原稿编号。
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
            _cv_end_to_end(chain)
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

    if process == "世界系到相机":
        return cv_ctrl_dataset
    if process == "机器视觉基础理论":
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
    process_name = process if process == "纯机器视觉双目姿态估计" else module.PROCESS
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
        "地面系到机体系": dataset_generate,
        "地面系转航迹系和气流系": ground_track_air_dataset,
        "体轴系到气流系": body_air_dataset,
        "机体系相对地面系": fm_ctrl_dataset,
        "气流系到机体系": fm_ctrl_dataset,
    }


def contexts_for_flight(row: dict) -> list[dict]:
    """One chain per flight subprocess. Slots stay on the local manuscript id."""
    modules = _flight_modules()
    grouped: dict[str, dict] = {}
    processes: list[str] = []
    for item in row["questions"]:
        if item["process"] not in modules:
            raise KeyError(f"飞行力学样本没有这个子过程：{item['process']}")
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
    control = chain["process"] in {"机体系相对地面系", "气流系到机体系"}
    if control:
        lead = (
            "本次是端到端问答。填写这个子过程里的全部待填槽，只调用这一次。"
            "只输出一个 JSON 对象，键是空的编号，值是答案。"
            "这个对象要包含本子过程的每一个编号，不要拆成多个对象，不要按句子分开作答。"
            "选择题的值是字母，不加引号。公式、矩阵、向量、表达式和距离的值用英文双引号包成字符串。"
            "不要输出句子原文，不要解释。"
        )
        output = "\n".join([
            "【输出】",
            "所有答案放在同一个 JSON 对象中。键是空的编号，值是答案。",
            '例如 {1:A, 2:C, 11:"[[1, 0, 0], [0, 1, 0], [0, 0, 1]]"}。示例不是本题答案。',
            "选择题的值只写字母，不加引号。公式、矩阵、向量、表达式和距离必须用英文双引号包成字符串。",
            "只输出这一行 JSON，不要解释、Markdown或LaTeX。",
        ])
    else:
        lead = (
            "本次是端到端问答。填写这个子过程里的全部待填槽，只调用这一次。"
            "只输出一个 JSON 对象，键是空的编号，值是答案。"
            "这个对象要包含本子过程的每一个编号，不要拆成多个对象，不要按句子分开作答。"
            "选择题的值是字母，不加引号。矩阵的值用英文双引号包成字符串。"
            "不要输出句子原文，不要解释。"
        )
        output = "\n".join([
            "【输出】",
            "所有答案放在同一个 JSON 对象中。键是空的编号，值是答案。",
            "选择题的值只写字母，不加引号。矩阵必须用英文双引号包成字符串。",
            f"只输出一行 {module._json_object(items)}。",
            f"格式示例（不是本题答案）：{module._json_object(items, example=True)}",
        ])
    return "\n".join([
        lead,
        "",
        "【定义声明】",
        chain["declaration"].strip(),
        "",
        "【物理常识】",
        module.PRIOR,
        "",
        "【子过程】",
        chain["process"],
        "",
        "【过程】",
        module._render_process(by_id, targets=local_ids, keep_suffix=True),
        "",
        output,
    ])


# Manuscript files hold the end-to-end questions. Gold matrices come from the
# generator's local ids, which drifted from a few handwritten cloze numbers:
# ground-to-body SθψΦ is local 28, and body-to-air Sαβ is local 35.
_FINAL_SOURCES = (
    ("地面系转换机体系.md", "地面系到机体系", {"1": "28"}),
    ("地面系转航迹系和气流系.md", "地面系转航迹系和气流系", {"1": "27", "2": "38", "3": "41"}),
    ("体轴系转换气流系.md", "体轴系到气流系", {"1": "21", "2": "32", "3": "35"}),
)
_FINAL_MARK = re.compile(
    r"端到端问题\s*[:：]\s*(.+?)[。．]\s*[^{]{0,80}\{DID\s*[:：]\s*(\d+)",
    re.S,
)


def load_final_questions() -> list[dict]:
    """Read the end-to-end questions the manuscripts added after each final matrix."""
    found = []
    for filename, process, gold in _FINAL_SOURCES:
        path = ROOT / filename
        if not path.is_file():
            raise SystemExit(f"找不到端到端题面：{path}")
        matches = _FINAL_MARK.findall(path.read_text(encoding="utf-8"))
        dids = [did for _question, did in matches]
        if set(dids) != set(gold) or len(dids) != len(gold):
            raise SystemExit(
                f"{filename} 的端到端题应是 DID {sorted(gold)}，实际读到 {dids or '空'}。"
            )
        for question, did in matches:
            found.append({
                "process": process,
                "did": did,
                "local_id": gold[did],
                "question": " ".join(question.split()) + "。",
            })
    return found


# λ、ω 不在飞行力学定义里。九个元素都用同一个因式，避免 0、1 和单独的三角函数露出旋转矩阵的样子。
_MATRIX_EXAMPLE_CELL = "sin(λ)*cos(ω)"
_MATRIX_EXAMPLE_BODY = "[[" + "], [".join(
    ", ".join(_MATRIX_EXAMPLE_CELL for _cell in range(3)) for _row in range(3)
) + "]]"
_MATRIX_FORMAT_RULES = (
    "矩阵固定写成一行 3×3，形如 [[a, b, c], [d, e, f], [g, h, i]]。"
    "因式之间必须写 *，数字和函数之间也必须写 *。"
    "sin、cos、tan 必须带圆括号，括号里用定义声明中的希腊字母，写成 cos(λ) 这种样子。"
    "角度符号与定义声明里的希腊字母一致。"
    "整个矩阵用英文双引号包进 JSON，只输出这一行。不要使用 LaTeX。"
)


def _flight_final_prompt(declaration: str, question: str, public_id: str) -> str:
    if any(char in _MATRIX_EXAMPLE_BODY for char in "θψΦφμγαβ"):
        raise RuntimeError("格式示例使用了本题的角度。")
    asked = f'{{{public_id}:"答案"}}'
    example = f'{{{public_id}:"{_MATRIX_EXAMPLE_BODY}"}}'
    return "\n".join([
        "本次只问最终结果，不提供分步填空。请根据定义声明自己推导，并把推导留在思维链里。",
        f"最终答案只输出一个 JSON 对象 {asked}。",
        _MATRIX_FORMAT_RULES,
        "",
        "【定义声明】",
        declaration.strip(),
        "",
        "【问题】",
        question.strip(),
        "",
        "【输出】",
        _MATRIX_FORMAT_RULES,
        "示例里的 λ、ω 只说明字符怎么排，不是本题角度，不要写进答案。",
        f"只输出一行 {asked}。",
        f"格式示例（不是本题答案）：{example}",
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
        if item is None or item.get("format_code") != "矩阵" or not answer.startswith("[["):
            raise SystemExit(
                f"样本 {row['id']} 的 {spec['process']} 局部编号 {spec['local_id']} 不是最终矩阵。"
            )
        public_id = str(item["id"])
        jobs.append({
            "sample_id": row["id"],
            "tier": row["tier"],
            "process": spec["process"],
            "blank_ids": [public_id],
            "blanks": [{
                "blank_id": public_id,
                "label": "端到端",
                "format_code": "矩阵",
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


# 示例字母不出现在风洞答案里。同一题的每个空都填同一个示例，避免露出哪一项带负号。
_WIND_FORMAT_EXAMPLE = {
    "表达式": "sin(λ)*cos(ω)",
    "arctan": "-arctan((sin(λ))/(cos(ω)))",
    "arcsin": "-arcsin(sin(λ))",
    "力臂": "-P·(Q+R)/L_RT",
}
_WIND_FORMAT_RULE = {
    "表达式": (
        "表达式固定写成一行。乘号必须写 *。sin、cos、tan 必须带圆括号，括号里用本题的角度符号。"
        "负号直接写在对应项前面。不要使用 LaTeX，不要写等号。"
    ),
    "arctan": (
        "arctan 固定写成一行，且只能有一个 arctan。分子和分母都必须为正。"
        "角度正负写在整个 arctan 外面：正角度写成 arctan((分子)/(分母))，负角度写成 -arctan((分子)/(分母))。"
        "不要把负号放进分子或分母。不要使用 LaTeX。"
    ),
    "arcsin": (
        "arcsin 固定写成一行，且只能有一个 arcsin。分子必须为正。"
        "角度正负写在整个 arcsin 外面：正角度写成 arcsin(分子)，负角度写成 -arcsin(分子)。"
        "不要写成分式，不要把负号放进分子。不要使用 LaTeX。"
    ),
    "力臂": (
        "力臂固定写成一行，形如力系数·力臂/L_RT。力系数与力臂之间用中点 ·。"
        "负号写在整个式子最外侧，力臂内部必须为正。"
        "力系数只能是 CA、CZ、CN 之一。力臂只能由 Xg、Yg、Zg、Xr 组成，含减法时加圆括号。"
        "分母保持 L_RT。不要留下字母 C 或 L_c，不要写等号。不要使用 LaTeX。"
    ),
}
_WIND_FINALS = (
    (
        "风轴系到体轴系",
        (
            ("1", "77", "α_t"),
            ("2", "78", "β_t"),
        ),
        "初始状态，体轴保持平正，已知来流在与风轴系轴向重合，吹向机头。"
        "当机构产生α_M、β_M、γ_M变化之后，机体处于一个新的姿态位置。"
        "现在需要根据已知量α_M、β_M、γ_M，推导新姿态位置下，机体的理论气流角α_t、β_t的表达式（不考虑天平形变和气流补偿）。"
        "这里给出公式模版：理论迎角α_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])，"
        "β_t的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。"
        "要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。"
        "禁止把正负号放进反三角函数内部来代替外层符号。"
        "要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。"
        "禁止把正负号放进反三角函数内部来代替外层符号。"
        "求得理论迎角α_t和理论侧滑角β_t。",
    ),
    (
        "天平系到体轴系",
        (
            ("1", "27", "CN_TZ"),
            ("2", "28", "CA_TZ"),
            ("3", "29", "CZ_TZ"),
            ("4", "30", "MN_TZ"),
            ("5", "31", "MA_TZ"),
            ("6", "32", "MZ_TZ"),
        ),
        "已知天平系中测量到的 6 分量分别是 CN_TP、MN_TP、CA_TP、MA_TP、CZ_TP、MZ_TP。"
        "请根据当前定义和这些天平测量量，推导体轴系中的 CN_TZ、CA_TZ、CZ_TZ、MN_TZ、MA_TZ、MZ_TZ。",
    ),
    (
        "天平系到风轴系",
        (
            ("1", "60", "Δα_e"),
            ("2", "61", "Δβ_e"),
        ),
        "Δα_e 的定义是天平变形后天平轴向在过风洞轴向铅垂面的投影与风洞轴向的夹角，"
        "Δβ_e 的定义是天平变形后天平轴向与过风洞轴向的铅垂面的夹角。"
        "天平处于平正位置时与风洞轴重合。在存在天平安装角的情况下，当天平发生形变后，"
        "Δα_e 和 Δβ_e 与天平弹性角的关系是什么？"
        "采用 sinθ=θ、cosθ=1 的原则简化，计算中直接使用弹性角的弧度值来近似表示。"
        "给出 Δα_e 和 Δβ_e。",
    ),
    (
        "力矩参考点",
        (
            ("1", "15", "MCN_CA"),
            ("2", "16", "MCZ_CA"),
            ("3", "17", "MCA_CZ"),
            ("4", "18", "MCN_CZ"),
            ("5", "19", "MCA_CN"),
            ("6", "20", "MCZ_CN"),
        ),
        "天平系转到体轴系后，体轴系原点在天平校心。工程上会给出力矩参考点，需要把体轴系上的力矩等效到力矩参考点。"
        "天平校心到模型顶点的距离为 Xr，由已测得的模型总长 L、模型底部到支杆底部的距离 L_t、天平校心到支杆底部的距离 L_r 间接算出。"
        "力矩参考点的工程约定坐标为 [Xg,Yg,Zg]。沿模型轴向的分量是参考点到模型顶点的距离，工程正方向指向模型后方；"
        "其余两分量的正方向遵循体轴系中对应坐标轴的正方向。"
        "MCA_g=MCA+[MCN_CA]+[MCZ_CA]，MCZ_g=MCZ+[MCA_CZ]+[MCN_CZ]，MCN_g=MCN+[MCA_CN]+[MCZ_CN]。"
        "每一项等效力矩系数只写成力系数乘力臂再除以固定参考长度 L_RT。"
        "给出 MCN_CA、MCZ_CA、MCA_CZ、MCN_CZ、MCA_CN、MCZ_CN。",
    ),
)


def _wind_final_prompt(declaration: str, question: str, blanks: list[dict]) -> str:
    for blank in blanks:
        example = _WIND_FORMAT_EXAMPLE[blank["format_code"]]
        if any(char in example for char in "αβγθφψμα"):
            raise RuntimeError("风洞格式示例使用了本题符号。")
    rules = []
    seen = set()
    for blank in blanks:
        code = blank["format_code"]
        if code in seen:
            continue
        seen.add(code)
        rules.append(_WIND_FORMAT_RULE[code] + f"示例（不是本题答案）：{_WIND_FORMAT_EXAMPLE[code]}。")
    asked = ", ".join(
        f'{blank["symbol"]} 写入键 {blank["blank_id"]}' for blank in blanks
    )
    fields = ", ".join(
        f'{blank["blank_id"]}:"{_WIND_FORMAT_EXAMPLE[blank["format_code"]]}"'
        for blank in blanks
    )
    example_object = "{" + fields + "}"
    answer_object = "{" + ", ".join(f'{blank["blank_id"]}:"答案"' for blank in blanks) + "}"
    return "\n".join([
        "本次只问最终结果，不提供分步填空。请根据定义声明自己推导，并把推导留在思维链里。",
        f"最终答案只输出一个 JSON 对象 {answer_object}。",
        " ".join(rules),
        "示例里的 λ、ω、P、Q、R 只说明字符怎么排，不是本题的量，不要写进答案。",
        "",
        "【定义声明】",
        declaration.strip(),
        "",
        "【问题】",
        question.strip(),
        "",
        "【输出】",
        " ".join(rules),
        asked + "。",
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。不要使用 LaTeX。",
        "示例里的 λ、ω、P、Q、R 只说明字符怎么排，不是本题的量，不要写进答案。",
        f"只输出一行 {answer_object}。",
        f"格式示例（不是本题答案）：{example_object}",
    ])


_CTRL_FINALS = (
    (
        "机体系相对地面系",
        (
            ("1", "77", "α"),
            ("2", "78", "β"),
        ),
        "初始状态，体轴保持平正，已知来流在与地面系轴向重合，吹向机头。"
        "当机构产生α_M、β_M、γ_M变化之后，机体处于一个新的姿态位置。"
        "现在需要根据已知量α_M、β_M、γ_M，推导新姿态位置下，机体的理论气流角α、β的表达式。"
        "这里给出公式模版：理论迎角α的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])，"
        "β的计算公式模板为[角度符号][反三角函数符号]([分子]/[分母])。"
        "要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。"
        "禁止把正负号放进反三角函数内部来代替外层符号。"
        "要求分子和分母都必须为正。该正值只表示角度大小的绝对值；角度正负必须单独判断，并且只能写在反三角函数外面。"
        "禁止把正负号放进反三角函数内部来代替外层符号。"
        "求得理论迎角α和理论侧滑角β。",
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
                    f"样本 {row['id']} 的 {process} 局部编号 {local_id} 不是端到端答案。"
                )
            blanks.append({
                "blank_id": str(item["id"]),
                "label": "端到端",
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


# 示例只用 λ、ω，避免和机器视觉答案里的角度、焦距、光心坐标相同。
_CV_FORMAT_EXAMPLE = {
    "矩阵": _MATRIX_EXAMPLE_BODY,
    "向量": "[sin(λ), cos(ω), 1]",
    "无": "A",
}
_CV_FORMAT_RULE = {
    "矩阵": (
        "矩阵固定写成一行 3×3，形如 [[a, b, c], [d, e, f], [g, h, i]]。"
        "乘号必须写 *，除号必须写 /。sin、cos、tan 必须带圆括号。"
        "不要使用 LaTeX，不要写等号，不要写成未乘开的矩阵乘积。"
        "每个元素写成整数或已给出量的表达式，不要留下 P_c1x、P_c1y、P_c1z 或下标花括号。"
    ),
    "向量": (
        "向量固定写成一行三个分量，形如 [a, b, c]。"
        "乘号必须写 *，除号必须写 /。sin、cos、tan 必须带圆括号。"
        "能写成整数就写整数。不要使用 LaTeX，不要写等号，不要写成未乘开的矩阵乘积。"
        "每个空号单独写一个向量，不要把多个空合成一个没有空号的式子。"
    ),
    "无": "选择题只写一个选项字母，不加引号。",
}
_CV_PROCESS = "计算机视觉双目姿态估计"
_CV_FINALS = (
    (
        (
            ("124", "R_t'"),
        ),
        "从机体系向风轴系的转换矩阵为 R_t'。"
        "现在需要根据机构角 α_M、β_M、γ_M 和当前坐标轴定义，给出 R_t' 的展开式。"
        "给出 R_t'。",
    ),
    (
        (
            ("37", "R_cw1"),
            ("39", "t_cw1"),
            ("76", "R_cw2"),
            ("78", "t_cw2"),
        ),
        "两台已标定相机从不同方位拍摄。"
        "相机1的方位角、俯仰角、滚转角为 ψ1、θ1、φ1，相机2为 ψ2、θ2、φ2，"
        "光心在风轴系中的坐标分别为 C1=(x1,y1,z1)、C2=(x2,y2,z2)。"
        "现在需要根据这些角度、光心坐标和当前坐标轴定义，推导这两个相机的位姿。"
        "R_cw1 是风轴系到相机1系的旋转矩阵，t_cw1 是相机1的位姿平移向量；"
        "R_cw2 是风轴系到相机2系的旋转矩阵，t_cw2 是相机2的位姿平移向量。"
        "给出 R_cw1、t_cw1、R_cw2 和 t_cw2。",
    ),
    (
        (
            ("140", "R21"),
            ("141", "t21"),
        ),
        "相机1、相机2的光心在风轴系中的坐标分别为 C1=(x1,y1,z1)、C2=(x2,y2,z2)，"
        "姿态角为 ψ1、θ1、φ1 与 ψ2、θ2、φ2。"
        "现在需要推导相机1系到相机2系的旋转 R21 和平移 t21。"
        "给出 R21 和 t21。",
    ),
    (
        (
            ("132", "P_c1"),
            ("135", "[P_c1]_×"),
            ("134", "P_c2"),
            ("136", "[P_c2]_×"),
        ),
        "机头在相机1系中的坐标为 P_c1，以向量形式表示。"
        "P_c1 的反对称矩阵为 [P_c1]_×，以矩阵形式表示。"
        "机头在相机2系中的坐标为 P_c2，以向量形式表示。"
        "P_c2 的反对称矩阵为 [P_c2]_×，以矩阵形式表示。"
        "现在需要根据机构角 α_M、β_M、γ_M，"
        "相机1姿态角 ψ1、θ1、φ1，相机2姿态角 ψ2、θ2、φ2，"
        "光心 C1=(x1,y1,z1)、C2=(x2,y2,z2) 和当前坐标轴定义，给出四者的展开式。"
        "[P_c1]_× 单独展开，不要与 (K1·S1)^(-T)、R_cw1、d_w 相乘。"
        "[P_c2]_× 单独展开，不要与 (K2·S2)^(-T)、R_cw2、d_w 相乘。"
        "给出 P_c1、[P_c1]_×、P_c2 和 [P_c2]_×。",
    ),
    (
        (
            ("137", "d_wV"),
            ("138", "d_wL"),
            ("139", "d_wN"),
        ),
        "空速在风轴系中的方向向量d_wV。"
        "空速在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wV，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wV。"
        "机体纵轴在机体系的单位向量为 e。它在风轴系中的方向向量为 d_wL。"
        "机体纵轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wL，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wL。"
        "机体法向轴在机体系的单位向量为 e。它在风轴系中的方向向量为d_wN。"
        "机体法向轴在相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·R_cw1·d_wN，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R_cw2·d_wN。"
        "写出 d_wV、d_wL、d_wN。",
    ),
    (
        (
            ("145", "d1"),
            ("146", "d2"),
            ("147", "X_c1"),
            ("148", "约束方程"),
            ("149", "叉乘向量"),
            ("150", "λ1"),
            ("151", "X_w"),
            ("152", "X_b"),
        ),
        "相机1、相机2图像上激波点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
        "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
        "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
        "现在需要给出这两个射线方向各自的展开式,写成向量形式。"
        "然后完成激波点反解。"
        "激波点在相机1系中的坐标 X_c1 沿射线方向 d1，λ1 为深度。X_c1 从下列选项中选择："
        "A. X_c1 = λ1·d1；B. X_c1 = d1 + λ1；C. X_c1 = λ1 + d1；D. X_c1 = d1 − λ1。"
        "两个相机看到同一激波点，相机1系到相机2系的变换为 X_c2=R21·X_c1+t21。约束方程从下列选项中选择："
        "A. λ2·d2 = R21·λ1·d1 + t21；B. λ2·d2 = R21·λ1·d1 − t21；"
        "C. λ2·d2 = R21^T·λ1·d1 + t21；D. λ1·d1 = R21·λ2·d2 + t21。"
        "在约束方程两侧叉乘一个向量可消去 λ2，得 d2×(λ1·R21·d1+t21)=0。叉乘的向量从下列选项中选择："
        "A. d2；B. d1；C. R21·d1；D. t21。"
        "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
        "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1。λ1 从下列选项中选择："
        "A. λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²；"
        "B. λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²；"
        "C. λ1 = −(d2×t21) / (d2×R21·d1)；"
        "D. λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²。"
        "激波点在相机1系中的坐标 X_c1 已知，它在风轴系中的坐标 X_w 从下列选项中选择："
        "A. X_w = R_cw1^T·(X_c1 − t_cw1)；B. X_w = R_cw1·X_c1 + t_cw1；"
        "C. X_w = R_cw1^T·X_c1 − t_cw1；D. X_w = R_cw1·(X_c1 − t_cw1)。"
        "激波点在风轴系中的坐标 X_w 已知，它在机体系中的坐标 X_b 从下列选项中选择："
        "A. X_b = R_t·RP1·(X_w − p_nose)；B. X_b = RP1^T·R_t'·(X_w − p_nose)；"
        "C. X_b = R_t·RP1·(X_w + p_nose)；D. X_b = R_t'·RP1·(X_w − p_nose)。"
        "给出 d1、d2，并给出 X_c1、约束方程、叉乘向量、λ1、X_w、X_b 的选项字母。",
    ),
)
_BASIC_PROCESS = "机器视觉基础理论"
_BASIC_FINALS = (
    (
        (
            ("37", "R21"),
            ("43", "t21"),
        ),
        "相机1系就是世界系。相机2的方位角、俯仰角、滚转角为 ψ2、θ2、φ2，"
        "光心在相机1系中的坐标为 C2=(x2,y2,z2)。"
        "现在需要根据这些角度、光心坐标和当前坐标轴定义，"
        "推导相机1系到相机2系的旋转 R21 和平移 t21。"
        "给出 R21 和 t21。",
    ),
    (
        (
            ("47", "P_c1"),
            ("50", "[P_c1]_×"),
            ("49", "P_c2"),
            ("51", "[P_c2]_×"),
        ),
        "直线上的点在相机1系中的坐标为 P_c1，以向量形式表示。"
        "P_c1 的反对称矩阵为 [P_c1]_×，以矩阵形式表示。"
        "该点在相机2系中的坐标为 P_c2，以向量形式表示。"
        "P_c2 的反对称矩阵为 [P_c2]_×，以矩阵形式表示。"
        "这条直线平行于相机1光轴正方向，该点沿相机1法向正方向，到光心的距离为 L。"
        "相机1系就是世界系。相机2的方位角、俯仰角、滚转角为 ψ2、θ2、φ2，"
        "光心在相机1系中的坐标为 C2=(x2,y2,z2)。"
        "现在需要根据距离 L、这些角度、光心坐标和当前坐标轴定义，给出四者的展开式："
        " P_c1、[P_c1]_×、P_c2 和 [P_c2]_×。",
    ),
    (
        (
            ("52", "d_w"),
        ),
        "相机1系就是世界系。"
        "相机1图像直线为 l=(K1·S1)^(-T)·[P_c1]_×·d_w，"
        "相机2图像直线为 l=(K2·S2)^(-T)·[P_c2]_×·R21·d_w。"
        "直线平行于相机1光轴正方向，方向向量 d_w 是该正方向在相机1系中的单位向量。"
        "现在需要给出 d_w 的展开式。"
        "d_w 单独展开成向量，不要与 (K1·S1)^(-T)、[P_c1]_×、R21 相乘，不要给出合并后的直线 l。"
        "给出 d_w。",
    ),
    (
        (
            ("56", "d1"),
            ("57", "d2"),
            ("58", "X_c1"),
            ("59", "约束方程"),
            ("60", "叉乘向量"),
            ("61", "λ1"),
            ("62", "X_w"),
        ),
        "相机1、相机2图像上目标点的齐次坐标分别为 [u1,v1,1]、[u2,v2,1]。"
        "相机1系射线方向 d1=(K1·S1)^(-1)·[u1,v1,1]^T，"
        "相机2系射线方向 d2=(K2·S2)^(-1)·[u2,v2,1]^T。"
        "现在需要给出这两个射线方向各自的展开式,写成向量形式。"
        "然后完成目标点反解。"
        "目标点在相机1系中的坐标 X_c1 沿射线方向 d1，λ1 为深度。X_c1 从下列选项中选择："
        "A. X_c1 = λ1·d1；B. X_c1 = d1 + λ1；C. X_c1 = λ1 + d1；D. X_c1 = d1 − λ1。"
        "两个相机看到同一目标点，相机1系到相机2系的变换为 X_c2=R21·X_c1+t21。约束方程从下列选项中选择："
        "A. λ2·d2 = R21·λ1·d1 + t21；B. λ2·d2 = R21·λ1·d1 − t21；"
        "C. λ2·d2 = R21^T·λ1·d1 + t21；D. λ1·d1 = R21·λ2·d2 + t21。"
        "在约束方程两侧叉乘一个向量可消去 λ2，得 d2×(λ1·R21·d1+t21)=0。叉乘的向量从下列选项中选择："
        "A. d2；B. d1；C. R21·d1；D. t21。"
        "在采用最小二乘准则、且 d2×R21·d1 ≠ 0 的条件下，"
        "由 d2×(λ1·R21·d1+t21)=0 求使残差平方最小的 λ1。λ1 从下列选项中选择："
        "A. λ1 = −(d2×t21)·(d2×R21·d1) / |d2×R21·d1|²；"
        "B. λ1 = (d2×t21)·(d2×R21·d1) / |d2×R21·d1|²；"
        "C. λ1 = −(d2×t21) / (d2×R21·d1)；"
        "D. λ1 = (d1×t21)·(d1×R21·d2) / |d1×R21·d2|²。"
        "目标点在相机1系中的坐标 X_c1 已知。相机1系就是世界系，所以它在世界系中的坐标 X_w 从下列选项中选择："
        "A. X_w = X_c1；B. X_w = R21^T·(X_c1 − t21)；"
        "C. X_w = R21·X_c1 + t21；D. X_w = X_c1 + t21。"
        "给出 d1、d2，并给出 X_c1、约束方程、叉乘向量、λ1、X_w 的选项字母。",
    ),
)


def _cv_declaration(declaration: str) -> str:
    text = declaration.strip()
    if text.startswith("【定义声明】"):
        text = text[len("【定义声明】"):].strip()
    return text


def _cv_example_field(blank: dict) -> str:
    example = _CV_FORMAT_EXAMPLE[blank["format_code"]]
    if blank["format_code"] == "无":
        return example
    return f'"{example}"'


def _cv_final_prompt(declaration: str, question: str, blanks: list[dict]) -> str:
    for blank in blanks:
        example = _CV_FORMAT_EXAMPLE[blank["format_code"]]
        if any(char in example for char in "αβγθφψμα"):
            raise RuntimeError("机器视觉格式示例使用了本题符号。")
    rules = []
    seen = set()
    for blank in blanks:
        code = blank["format_code"]
        if code in seen:
            continue
        seen.add(code)
        rules.append(_CV_FORMAT_RULE[code] + f"示例（不是本题答案）：{_CV_FORMAT_EXAMPLE[code]}。")
    asked = ", ".join(
        f'{blank["symbol"]} 写入键 {blank["blank_id"]}' for blank in blanks
    )
    fields = ", ".join(
        f'{blank["blank_id"]}:{_cv_example_field(blank)}'
        for blank in blanks
    )
    example_object = "{" + fields + "}"
    answer_object = "{" + ", ".join(
        f'{blank["blank_id"]}:答案' if blank["format_code"] == "无" else f'{blank["blank_id"]}:"答案"'
        for blank in blanks
    ) + "}"
    return "\n".join([
        "本次只问最终结果，不提供分步填空。请根据定义声明自己推导，并把推导留在思维链里。",
        f"最终答案只输出一个 JSON 对象 {answer_object}。",
        " ".join(rules),
        "示例里的 λ、ω 只说明字符怎么排，不是本题的量，不要写进答案。",
        "",
        "【定义声明】",
        _cv_declaration(declaration),
        "",
        "【问题】",
        question.strip(),
        "",
        "【输出】",
        " ".join(rules),
        asked + "。",
        "每个空号各写一个值，不要把多个空合成一个没有空号的式子。不要使用 LaTeX。",
        "示例里的 λ、ω 只说明字符怎么排，不是本题的量，不要写进答案。",
        f"只输出一行 {answer_object}。",
        f"格式示例（不是本题答案）：{example_object}",
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
            symbolic = code in {"矩阵", "向量"} and answer.startswith(("[", "[["))
            choice = code == "无" and len(answer) == 1 and answer.isalpha() and bool(item.get("options"))
            if item is None or not (symbolic or choice):
                raise SystemExit(
                    f"样本 {row['id']} 的 {process_name} 局部编号 {local_id} 不是端到端答案。"
                )
            blanks.append({
                "blank_id": str(item["id"]),
                "label": "端到端",
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


def prompt_for(
    chain: dict,
    blank_ids: list[str],
    condition: str,
    memory: dict[str, str] | None,
    local_target: str | None = None,
) -> str:
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
        lines.append(f"【{item['role']}】")
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
    start = text.find("【过程】")
    end = text.find("\n【输出】", start)
    if start < 0 or end < 0:
        raise RuntimeError(f"{chain['process']} 的当前句没有过程正文。")
    return text[start + len("【过程】"):end].strip()


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
    lead = "填写当前这一句。只输出这一句的 JSON，不要解释。"
    return "\n".join([
        lead,
        "",
        "【子过程】",
        chain["process"],
        "",
        "【问题】",
        current,
    ])


def _session_system(chain: dict) -> str:
    prior = ""
    if chain.get("domain") == "flight":
        prior = _flight_modules()[chain["process"]].PRIOR
    elif chain.get("domain") == "cv":
        prior = _cv_module(chain.get("process")).PRIOR
    else:
        from domains.wind_tunnel.dataset_generate import PRIOR
        prior = PRIOR
    parts = [
        "本会话只在这条系统消息里给出一次定义、物理常识和输出格式。",
        "后面的每一问只追加当前句子。请从本消息检索工程定义，不要等待定义被重复。",
        "",
        "【定义声明】",
        chain["declaration"].strip(),
    ]
    if str(prior).strip():
        parts.extend(["", "【物理常识】", str(prior).strip()])
    parts.extend([
        "",
        "【输出】",
        "当前这一问只输出一个 JSON 对象。键是空的编号，值是答案。",
        "选择题的值只写字母。矩阵、向量、表达式和公式用英文双引号包成字符串。",
        "不要输出句子原文，不要解释。",
    ])
    return "\n".join(parts)


def _guard_session(messages: list[dict], declaration: str) -> None:
    marker = declaration.strip()[:40]
    if len(marker) < 12:
        return
    for item in messages:
        if item["role"] == "user" and marker in item["content"]:
            raise RuntimeError("后续问题重复注入了定义声明。")


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
        elif char in ",，" and depth == 0:
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
    while index < len(text) and text[index] not in ":：":
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
        if index >= len(text) or text[index] not in ":：":
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
        if index < len(text) and text[index] in ",，":
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
        match = re.match(r"^(?:ID\s*[:：]?\s*)?(\d+)\s*[:：、.\s]\s*(.*)$", line)
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
        elif depth == 0 and char in "\n;；":
            break
        cursor += 1
    return cursor


_LABEL_ONLY = re.compile(
    r"^(?:答案|结果|因此|所以)?\s*[A-Za-zα-ωθφψΔ0-9_\\]{0,16}\s*[:：=]\s*$"
)


def _is_answer_chunk(chunk: str) -> bool:
    text = chunk.strip().strip("。；;")
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
        while index < length and source[index] in " \t\r\n,，、":
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
        chunk = source[index:end].strip().strip("，,、")
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
    if _parse_reply("{1：A}").get("1") != "A":
        raise RuntimeError("fullwidth colon")
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
            {"blank_id": "145", "label": "A2", "format_code": "向量", "options": [], "answer": "[(-cx1 + u1)/fx1, (-cy1 + v1)/fy1, 1]"},
            {"blank_id": "146", "label": "A2", "format_code": "向量", "options": [], "answer": "[1, 0, 0]"},
        ],
    )
    if not ordered[0]["normalized"] or ordered[1]["prediction"] or ordered[1]["normalized"]:
        raise RuntimeError(ordered)
    paired = _score_blanks(
        "[1, 0, 0]\n[0, 1, 0]",
        [
            {"blank_id": "137", "label": "A2", "format_code": "向量", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "138", "label": "A2", "format_code": "向量", "options": [], "answer": "[0, 1, 0]"},
            {"blank_id": "139", "label": "A2", "format_code": "向量", "options": [], "answer": "[0, 0, 1]"},
        ],
    )
    if not paired[0]["exact"] or not paired[1]["exact"] or paired[2]["prediction"] or paired[2]["normalized"]:
        raise RuntimeError(paired)
    named = _score_blanks(
        '{"d1":"[1, 0, 0]","d2":"[0, 1, 0]"}',
        [
            {"blank_id": "145", "label": "A2", "format_code": "向量", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "146", "label": "A2", "format_code": "向量", "options": [], "answer": "[0, 1, 0]"},
        ],
    )
    if not named[0]["exact"] or not named[1]["exact"]:
        raise RuntimeError(named)
    numbered = _score_blanks(
        '{146:"[0, 1, 0]", 145:"[1, 0, 0]"}',
        [
            {"blank_id": "145", "label": "A2", "format_code": "向量", "options": [], "answer": "[1, 0, 0]"},
            {"blank_id": "146", "label": "A2", "format_code": "向量", "options": [], "answer": "[0, 1, 0]"},
        ],
    )
    if not numbered[0]["exact"] or not numbered[1]["exact"]:
        raise RuntimeError(numbered)
    if _resolve_max_tokens(0) != DEFAULT_MAX_TOKENS or _resolve_max_tokens(MIN_MAX_TOKENS) != MIN_MAX_TOKENS:
        raise RuntimeError("max_tokens 没有保持在 2048–8192。")
    if _resolve_max_tokens(0, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("思考开启时未使用 65536。")
    if _resolve_max_tokens(4096, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("思考开启时没有固定为 65536。")
    if _resolve_max_tokens(100000, "enabled", announce=False) != THINKING_MAX_TOKENS:
        raise RuntimeError("思考开启时没有固定为 65536。")
    if _resolve_max_tokens(100000, announce=False) != MAX_TOKEN_CEILING:
        raise RuntimeError("思考关闭时 max_tokens 没有截在 32768。")
    if _resolve_workers(MAX_WORKERS) != MAX_WORKERS:
        raise RuntimeError("并发上限被改写了。")


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
        return {"usd": None, "note": "这次调用没有返回用量。"}
    if priced is None:
        return {"usd": None, "note": "计价表只覆盖 DeepSeek 官方接口上的 flash 和 v4-pro。"}
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
    options = [str(option) for option in (blank.get("options") or []) if option and str(option) != "无"]
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
                f"思考开启时 max_tokens 固定为 {THINKING_MAX_TOKENS}，请求的 {requested} 已改为该值。"
                "思维链用完即记错，不再提高上限或改思考强度。",
                flush=True,
            )
        return THINKING_MAX_TOKENS
    if requested <= 0:
        return DEFAULT_MAX_TOKENS
    if requested < MIN_MAX_TOKENS:
        if announce:
            print(
                f"max_tokens {requested} 低于 {MIN_MAX_TOKENS}，已提高到 {MIN_MAX_TOKENS}。"
                "思维链计入该上限，512 会截断最终答案，重跑比多出来的 token 更贵。",
                flush=True,
            )
        return MIN_MAX_TOKENS
    if requested > MAX_TOKEN_CEILING:
        if announce:
            print(
                f"max_tokens {requested} 超过全局上限 {MAX_TOKEN_CEILING}，已截为 {MAX_TOKEN_CEILING}。",
                flush=True,
            )
        return MAX_TOKEN_CEILING
    return requested


def _resolve_workers(requested: int) -> int:
    """Long outputs at high concurrency turn into timeouts and 429 retries."""
    if requested < 1:
        raise SystemExit("--workers 至少为 1。")
    if requested > MAX_WORKERS:
        print(
            f"并发 {requested} 在长输出下会增加超时和 429 重试，已限制为 {MAX_WORKERS}。"
            "重试在账单上是额外调用，不在成功记录的 token 里。",
            flush=True,
        )
        return MAX_WORKERS
    return requested


class RunStop(RuntimeError):
    """额度或持续限流。失败的这一句不写入记录，方便从同一句继续。"""

    def __init__(self, kind: str, message: str, job: dict):
        super().__init__(message)
        self.kind = kind
        self.job = job


def _error_kind(text: str) -> str:
    lowered = text.lower()
    # 402 必须是独立状态码。请求号里的 402 不能当成额度错误。
    if re.search(r"(?<![0-9a-z])402(?![0-9a-z])", lowered) or any(token in lowered for token in (
        "insufficient", "quota", "arrearage", "balance", "allocationquota",
        "freetier", "billing", "prepaid", "余额", "欠费", "额度",
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
            last_error = f"模型没有答复：输出在 {max_tokens} token 处被截断。"
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
            f"- 空数 {total}，严格 {exact}（{_rate(exact, total)}），归一化 {normalized}（{_rate(normalized, total)}），调用失败 {errors}",
            "",
        ]

    lines = ["# 填空评测", ""]
    if preface:
        lines.extend(preface)
        lines.append("")
    lines.extend(block("全部", rows))
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
        f"样本自助法 95% 区间：严格 {strict_low:.1%}–{strict_high:.1%}，归一化 {norm_low:.1%}–{norm_high:.1%}。"
        "重抽样单位是一条样本的一次运行。"
    )
    lines.append("")
    lines.append("## 错误分类")
    lines.append("")
    lines.append("只统计严格不一致。格式表示关键词之后的答案去掉空格、杂字符和 LaTeX 后一致，表达式与矩阵经 sympy 等价，或选择题的字母、选项原文、能唯一对应正确选项的片段命中。")
    lines.append("")
    misses = [item for item in rows if not item["exact"]]
    counts = Counter(item.get("error_kind") or "其他" for item in misses)
    for kind in ("格式", "轴", "符号", "定义未用", "其他"):
        lines.append(f"- {kind} {counts.get(kind, 0)}")
    lines.append("")
    lines.append("## 按定义变更档")
    lines.append("")
    present_tiers = []
    for item in rows:
        if item["tier"] not in present_tiers:
            present_tiers.append(item["tier"])
    for tier in sorted(present_tiers, key=lambda value: int(value)):
        lines.extend(block(f"{tier}%", [item for item in rows if item["tier"] == tier]))
    lines.append("## 按标签")
    lines.append("")
    labels = ["C1", "C2", "A1", "A2"]
    if any(item["label"] == "端到端" for item in rows):
        labels.append("端到端")
    for label in labels:
        lines.extend(block(label, [item for item in rows if item["label"] == label]))
    lines.append("## 按数据")
    lines.append("")
    sample_ids = []
    for item in rows:
        if item["sample_id"] not in sample_ids:
            sample_ids.append(item["sample_id"])
    for sample_id in sample_ids:
        subset = [item for item in rows if item["sample_id"] == sample_id]
        tier = subset[0]["tier"]
        lines.extend(block(f"{sample_id}（{tier}%）", subset))
    shown = [item for item in misses if item.get("error_kind") != "格式"][:30]
    lines.extend(["## 归一化后仍不一致（最多30条）", ""])
    if not shown:
        lines.append("没有。")
    for item in shown:
        pred = str(item["prediction"]).replace("\n", " / ")
        if len(pred) > 180:
            pred = pred[:180] + "…"
        lines.append(
            f"- {item['sample_id']} {item['process']} ID {item['blank_id']} {item['label']} {item.get('error_kind', '')}："
            f"答案 `{item['answer']}`；输出 `{pred}`"
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
        "title": "统一默认",
        "change": "统一默认。",
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
            ("纯机器视觉双目姿态估计", blank_id): answer
            for (process, blank_id), answer in list(found.items())
            if process == "计算机视觉双目姿态估计"
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
    if domain == "flight":
        return standard_answers_flight()
    if domain == "cv":
        return standard_answers_cv(data)
    return standard_answers()


def standard_answers() -> dict[tuple[str, str], str]:
    from domains.wind_tunnel.dataset_generate import build_record
    from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

    record = build_record({
        "id": "standard",
        "title": "统一默认",
        "change": "统一默认。",
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
    role = "主评测" if model == "deepseek-v4-pro" and thinking == "enabled" else "对照"
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
        f"本次是{role}：模型 {model}，思考 {thinking}，条件 {condition_label(condition)}。",
        "条件名写明注入方式。"
        + " ".join(f"{name}：{gloss}。" for name, gloss in CONDITION_GLOSS.items())
        + " 旧代号 A、B、C、D、E、F 仍可输入，依次对应这六个名字。",
        (
            "主评测口径仍是 deepseek-v4-pro 且思考开启。"
            + (
                "本次 deepseek-flash 也开启了思考，用来看个性化定义对正确率的影响。"
                if thinking == "enabled"
                else "deepseek-flash 在思考关闭时只代表非思考模式，不能当作 V4 的完整能力。"
            )
        ),
        "思考开启时温度不生效。官方 V4 接口忽略 seed，重复运行才提供方差，下面的区间是样本自助法。",
    ]
    if max_tokens:
        stop_note = "端到端不在空行处停止。" if condition in _END_TO_END else "思考关闭时另在空行处停止。"
        lines.append(
            f"温度参数 {temperature}，重复运行 {runs}，请求种子 {seed}，max_tokens {max_tokens}。"
            f"该上限含思维链，思考关闭时也保留这段长度。{stop_note}"
        )
    else:
        lines.append(f"温度参数 {temperature}。这批调用没有设置 max_tokens、stop 或 seed。")
    retries = sum(max(0, int(item.get("attempts") or 1) - 1) for item in rows)
    if known:
        lines.append(
            f"已记录用量的调用 {known} 次。缓存命中 {hit}，缓存未命中 {miss}，输出 {output}。"
            f"按 DeepSeek 价目估算 {usd:.4f} 美元，不是账单。"
            f"其中重试 {retries} 次。用量只含成功返回的那一次，超时和 429 的失败尝试不在这些 token 里，账单调用数会更高。"
        )
    else:
        lines.append(
            "这批记录没有用量和费用，无法核对成本。"
            "超时和 429 的重试同样不会出现在 token 估算里。"
        )
    if versions or documented or dates:
        lines.append(
            "服务端模型名：" + ("、".join(versions) or "无")
            + "。文档版本：" + ("、".join(documented) or "无")
            + "。调用日期：" + ("、".join(dates) or "无") + "。"
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
    process_order = ["风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点"]
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
            f"# {sample_id} 原始输出",
            "",
            f"模型 {head.get('model', '')}，思考 {head.get('thinking', '')}，条件 {condition_label(head.get('condition', ''))}。",
            "下面按子过程和句序保留模型的原始回复，没有改写。",
            "",
        ]
        for item in items:
            if len(item["blank_ids"]) > 8:
                lines.append(f"## {item['process']} 全部 {len(item['blank_ids'])} 个空")
            else:
                lines.append(f"## {item['process']} ID {'、'.join(item['blank_ids'])}")
            lines.append("")
            if item.get("unanswered"):
                lines.append(f"模型没有答复：输出在 {item.get('max_tokens') or THINKING_MAX_TOKENS} token 处被截断。")
                lines.append("")
            elif item.get("error"):
                lines.append(f"调用失败：{item['error']}")
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
            raise RuntimeError("思维链被写回了下一句的输入。多轮只保留答案文本。")
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
            branch = make_branch(chain, messages, committed, spec)
            if _chat_snapshot(messages) != before:
                raise RuntimeError(f"{spec['probe_id']} 写回了主会话")
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
                raise RuntimeError(f"{spec['probe_id']} 的答案写回了主会话")
            if PROBE_LEAD in "\n".join(content for _, content in before):
                raise RuntimeError(f"{spec['probe_id']} 的题干进入了主会话")
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
            raise RuntimeError(f"{chain['process']} 的主问题带了探针题干")
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
            raise RuntimeError("思维链被写回了下一句的输入。会话只保留答案文本。")
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
            "\n\n> 这一轮因额度或限流中断，上面的分数只覆盖已经成功的句子。"
            "失败的那一句没有写入记录，用原命令再跑会从该句继续。\n"
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
    ids_text = "、".join(str(item) for item in (job.get("blank_ids") or []))
    print(
        f"断点：{condition} 样本 {job.get('sample_id')} 过程 {job.get('process')} "
        f"空 {ids_text} 原因 {halt.kind}。已完成 {len(stored)}/{sentence_count} 句。",
        flush=True,
    )
    print(payload["resume_command"], flush=True)
    raise SystemExit(2)


def main() -> None:
    from cloze_score import _check_scoring

    _check_scoring()
    _check_parse()
    parser = argparse.ArgumentParser(description="Score cloze blanks with DeepSeek.")
    parser.add_argument("--domain", choices=("wind_tunnel", "flight", "cv"), default="wind_tunnel")
    parser.add_argument("--data", type=Path, default=SETS["cross.wind_flight"]["batch"] / "samples.jsonl")
    parser.add_argument("--ids-file", type=Path, default=SETS["cross.wind_flight"]["batch"] / "pilot_10.json")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "pilot_100")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="Only score the first N sentences. 0 means all.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--model", default="", help="覆盖环境变量中的模型。主评测用 deepseek-v4-pro。")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="0 在思考关闭时是 8192。思考开启时固定为 65536，思维链用完即记错，不再提高上限。该上限含思维链。",
    )
    parser.add_argument(
        "--condition",
        default="A",
        help="gold-trunc、self-trunc、gold-keep、self-keep、proc-e2e、final-e2e、"
             "session-gold、session-model。"
             "gold/self 是空前用标准答案还是自身输出，trunc/keep 是句号截断还是保留后文。"
             "proc-e2e 每个子过程一次，final-e2e 只问最终问题。"
             "session-gold、session-model 在一个会话里逐题提问，定义只写在系统消息。"
             "旧代号 A、B、C、D、E、F 仍可用。",
    )
    parser.add_argument("--rescore", type=Path, default=None, help="只按已保存的回复重算分数，不调用模型。")
    parser.add_argument(
        "--gold-nodes",
        default="",
        help="自身前缀条件下，这些题号答完后向后文注入标准答案。模型该题得分仍按原回答计算。逗号分隔，如 24,47,48。",
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs 至少为 1。")
    wind_batch = SETS["cross.wind_flight"]["batch"]
    if args.domain == "flight":
        flight_root = SETS["basic.flight"]["pilot"]
        if args.data == wind_batch / "samples.jsonl":
            args.data = flight_root / "samples.jsonl"
        if args.ids_file == wind_batch / "pilot_10.json":
            args.ids_file = flight_root / "pilot_10.json"
    elif args.domain == "cv":
        vision = SETS["cross.wind_vision"]
        basic_vision = SETS["basic.vision"]
        if args.data == wind_batch / "samples.jsonl":
            args.data = vision["samples"]
        if args.ids_file == wind_batch / "pilot_10.json":
            args.ids_file = vision["pilot"]
        if _basic_vision_data(args.data) and args.ids_file in {
            wind_batch / "pilot_10.json", vision["pilot"],
        }:
            args.ids_file = basic_vision["pilot"]

    load_env_file(ROOT / ".env")
    _key, base_url, env_model = resolve_api_config()
    model = args.model.strip() or env_model
    thinking = args.thinking
    repair_ids = {
        part.strip().removeprefix("ID").removeprefix("id")
        for part in args.gold_nodes.replace("，", ",").split(",")
        if part.strip()
    }
    max_tokens = _resolve_max_tokens(args.max_tokens, thinking)
    workers = _resolve_workers(args.workers)
    stop = None if thinking == "enabled" else ["\n\n"]
    if args.rescore is None:
        print(
            f"条件：{condition_label(args.condition)} model={model} thinking={thinking} temperature={args.temperature} "
            f"temperature_applied={thinking != 'enabled'} runs={args.runs} seed={args.seed} max_tokens={max_tokens}"
            + (f" gold_nodes={','.join(sorted(repair_ids, key=int))}" if repair_ids else ""),
            flush=True,
        )
    if args.rescore is None and (model != "deepseek-v4-pro" or thinking != "enabled"):
        print("这不是主评测。主评测是 deepseek-v4-pro 且 --thinking enabled。", flush=True)

    if args.rescore is not None:
        target = args.rescore
        stored = _load_jsonl(target / "records.jsonl")
        print(f"重算 {len(stored)} 条记录", flush=True)
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
                "rescore": "按保存的回复重算严格分和归一化分，没有重新调用模型。",
            })
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(report.split("## 按定义变更档")[0])
        return

    picked = json.loads(args.ids_file.read_text(encoding="utf-8"))
    ids = picked["ids"] if isinstance(picked, dict) else picked
    rows = load_samples(args.data, set(ids))
    order = {sample_id: index for index, sample_id in enumerate(ids)}
    rows.sort(key=lambda row: order[row["id"]])
    print("生成统一默认答案，用于判断定义未用。", flush=True)
    standard = standard_for(args.domain, args.data)
    build_contexts = (
        contexts_for_cv if args.domain == "cv"
        else contexts_for_flight if args.domain == "flight"
        else contexts_for
    )

    condition = canonical_condition(args.condition)
    if repair_ids and condition not in _SEQUENTIAL:
        raise SystemExit("--gold-nodes 只用于自身前缀（self-trunc / self-keep）。标准答案前缀本来就注入正确答案。")
    if not condition:
        raise SystemExit(
            "条件只能是 gold-trunc、self-trunc、gold-keep、self-keep、proc-e2e、final-e2e、"
            "session-gold、session-model。"
            "gold/self 表示空前用标准答案还是模型自己的输出，trunc/keep 表示句号截断还是保留后文。"
            "proc-e2e 是每个子过程一次，final-e2e 只问最终问题。"
            "session-gold 和 session-model 在一个会话里逐题提问，定义只写在系统消息。"
            "旧代号 A、B、C、D、E、F 仍可用。旧拼写 C'、C'' 分别等价于 gold-keep、self-keep。"
        )
    if condition == FINAL_E2E and args.domain not in {"flight", "wind_tunnel", "cv"}:
        raise SystemExit("条件 final-e2e 用于飞行力学、风洞和机器视觉。")
    if condition in _END_TO_END:
        stop = None
    if condition == PROC_E2E:
        print(f"{condition_label(condition)}：max_tokens={max_tokens}，不在空行处停止。", flush=True)
    if condition == FINAL_E2E:
        print(
            f"{condition_label(condition)}：max_tokens={max_tokens}。",
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
                    item.get("process") == "机体系相对地面系" for item in row["questions"]
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
                elif args.domain == "cv" and row["questions"][0].get("process") == "纯机器视觉双目姿态估计":
                    from domains.computer_vision.cv_vision_context import rewrite
                    finals = tuple(
                        (slots, rewrite(question).replace("来流方向", "参考方向"))
                        for slots, question in _CV_FINALS
                    )
                    jobs.extend(cv_final_jobs(
                        row, run, args.seed + run, max_tokens, stop, standard,
                        finals=finals, process="纯机器视觉双目姿态估计",
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
                for chain in build_contexts(row):
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
                f"目录 {args.out} 已有 model={old.get('model')} thinking={old.get('thinking')} 的记录。"
                "换一个 --out，避免和这组条件混在一起。"
            )
        if old.get("condition", "A") != condition:
            raise SystemExit(
                f"目录 {args.out} 已有条件 {old.get('condition', 'A')} 的记录。"
                "换一个 --out，避免和这组条件混在一起。"
            )
        if set(old.get("gold_repaired") or []) != repair_ids:
            raise SystemExit(
                f"目录 {args.out} 已有 gold_repaired={old.get('gold_repaired') or []} 的记录。"
                "换一个 --out，避免和另一组节点修复混在一起。"
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
        raise SystemExit("这个结果目录是按单个空记录的。整句评测请换一个 --out 目录，避免覆盖原来的报告。")
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
            f"矩阵探针 {len(probes)} 条写在 probes.jsonl。只保留浅层和重推导深层。"
            "分矩阵和总矩阵在主会话里记分后换成占位符，不进入后续提问。"
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
        "seed_note": "DeepSeek 官方 V4 接口忽略 seed。",
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
    print(report.split("## 按定义变更档")[0])


if __name__ == "__main__":
    main()
