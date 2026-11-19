# -*- coding: utf-8 -*-
"""四层工程契约指标。

每条反常定义按窗口计分，不把整卷准确率混在一起。四层是：

1. 规则识别：直接询问目标定义的空。无标记和有标记稿已删去这些空。
2. 规则执行：该窗口里第一处使用目标定义、且前面还没有别的目标结果的空。
3. 后续推理：符号、转向、分量等继续沿用该定义的空。
4. 数值推导：矩阵和公式。

同一窗口里更早的执行或推理空，在 gold-trunc 下会把正确答案写进前缀。
那些后继空仍单独出准确率，但不进入“局部执行准确率”。

先问规则稿不拆文件。注入方式决定它属于哪一臂：

- rule-gold：先问规则稿，条件 gold-trunc。规则空的正确答案写入后继空的前缀。
- rule-model：先问规则稿，条件 self-trunc。前缀里是模型自己此前的全部回答。

完整前缀传播损失是同一稿、同一空上 gold-trunc 减 self-trunc。
self-trunc 的前缀是模型此前的全部回答，不是最近一个回答。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import canonical_condition  # noqa: E402
from cloze_score import bootstrap_ci, judge  # noqa: E402

PAIR_PATH = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
GOLD = "gold-trunc"
SELF = "self-trunc"
FAMILIES = ("unmarked", "marked", "rule")
RESULT_LAYERS = {"execution", "reasoning", "derivation"}
LAYER_TITLE = {
    "recognition": "规则识别",
    "execution": "规则执行",
    "reasoning": "后续推理",
    "derivation": "数值推导",
}

# 步骤按题面顺序书写。识别题放在它真正出现的位置。
# 局部执行只取“前面还没有目标结果”的执行空。
WINDOWS = {
    "alpha": (
        ("理论迎角", "风轴系到体轴系", (
            ("21", "recognition"),
            ("22", "execution"),
            ("68", "reasoning"),
            ("80", "derivation"),
        )),
        ("弹性迎角", "天平系到风轴系", (
            ("155", "recognition"),
            ("158", "execution"),
            ("164", "reasoning"),
            ("174", "derivation"),
            ("176", "derivation"),
            ("179", "reasoning"),
        )),
    ),
    "roll": (
        ("机构滚转", "风轴系到体轴系", (
            ("35", "recognition"),
            ("36", "execution"),
            ("39", "reasoning"),
            ("42", "execution"),
            ("43", "derivation"),
            ("52", "derivation"),
            ("55", "derivation"),
            ("60", "derivation"),
            ("61", "derivation"),
            ("80", "derivation"),
            ("81", "derivation"),
        )),
        ("滚转对齐", "天平系到体轴系", (
            ("84", "recognition"),
            ("85", "execution"),
            ("87", "reasoning"),
            ("88", "derivation"),
            ("103", "derivation"),
            ("104", "derivation"),
            ("111", "derivation"),
            ("113", "derivation"),
            ("114", "derivation"),
            ("116", "derivation"),
        )),
    ),
    "order": (
        ("机构角顺序", "风轴系到体轴系", (
            ("16", "recognition"),
            ("17", "recognition"),
            ("48", "execution"),
            ("49", "reasoning"),
            ("53", "derivation"),
            ("54", "derivation"),
            ("55", "derivation"),
            ("60", "derivation"),
            ("61", "derivation"),
            ("80", "derivation"),
            ("81", "derivation"),
        )),
    ),
    "mount": (
        ("安装·风轴到体轴", "风轴系到体轴系", (
            ("11", "recognition"),
            ("12", "execution"),
            ("13", "derivation"),
            ("19", "recognition"),
            ("27", "recognition"),
        )),
        ("安装·天平到体轴", "天平系到体轴系", (
            ("96", "recognition"),
            ("97", "execution"),
            ("101", "derivation"),
            ("104", "derivation"),
            ("111", "derivation"),
            ("113", "derivation"),
            ("114", "derivation"),
            ("116", "derivation"),
        )),
        ("安装·弹性角", "天平系到风轴系", (
            ("151", "recognition"),
            ("152", "execution"),
            ("153", "reasoning"),
            ("158", "reasoning"),
            ("159", "reasoning"),
            ("164", "reasoning"),
            ("169", "reasoning"),
            ("174", "derivation"),
            ("175", "derivation"),
            ("176", "derivation"),
            ("177", "derivation"),
        )),
    ),
    "body": (
        ("体轴手性·风轴到体轴", "风轴系到体轴系", (
            ("7", "recognition"),
            ("8", "execution"),
            ("25", "reasoning"),
            ("34", "derivation"),
            ("43", "derivation"),
            ("52", "derivation"),
            ("53", "derivation"),
            ("55", "derivation"),
            ("61", "derivation"),
            ("67", "reasoning"),
            ("76", "reasoning"),
        )),
        ("体轴手性·天平到体轴", "天平系到体轴系", (
            ("94", "recognition"),
            ("95", "execution"),
            ("102", "derivation"),
            ("104", "derivation"),
            ("113", "derivation"),
            ("116", "derivation"),
        )),
        ("力矩参考点", "力矩参考点", (
            ("189", "execution"),
            ("190", "reasoning"),
            ("191", "reasoning"),
            ("192", "reasoning"),
            ("193", "reasoning"),
            ("194", "reasoning"),
            ("195", "derivation"),
            ("196", "derivation"),
            ("197", "derivation"),
            ("198", "derivation"),
            ("199", "derivation"),
            ("200", "derivation"),
        )),
    ),
}


def _matches(prediction: str, answer: str, format_code: str, options: list[str]) -> bool:
    if prediction == "" or answer == "":
        return False
    if str(prediction).strip() == str(answer).strip():
        return True
    return bool(judge(str(prediction), str(answer), format_code, options).get("normalized"))


def _differs(abnormal: str, standard: str, format_code: str, options: list[str]) -> bool:
    return not _matches(abnormal, standard, format_code, options)


def classify_outcome(prediction: str, abnormal: str, standard: str, format_code: str, options: list[str]) -> str:
    """反常答案、标准答案诱饵、其他错误。答案相同的空不能用来计算诱饵率。"""
    if _matches(prediction, abnormal, format_code, options):
        return "abnormal"
    if _differs(abnormal, standard, format_code, options) and _matches(prediction, standard, format_code, options):
        return "standard"
    return "other"


def _split_sample(sample_id: str) -> tuple[str, str, str] | None:
    for family in FAMILIES:
        suffix = f"-{family}"
        if sample_id.endswith(suffix):
            stem = sample_id[: -len(suffix)]
            factor = stem.split("-", 1)[0]
            if factor not in WINDOWS:
                return None
            return factor, family, stem
    return None


def _arm(family: str, condition: str) -> str:
    if condition == GOLD and family == "unmarked":
        return "unmarked"
    if condition == GOLD and family == "marked":
        return "marked"
    if condition == GOLD and family == "rule":
        return "rule-gold"
    if condition == SELF and family == "rule":
        return "rule-model"
    return ""


@lru_cache(maxsize=1)
def load_catalog(path: Path = PAIR_PATH) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["id"]: row for row in rows}
    factors = []
    for factor, windows in WINDOWS.items():
        rule = by_id[f"{factor}-rule"]
        unmarked = by_id[f"{factor}-unmarked"]
        marked = by_id[f"{factor}-marked"]
        standard = by_id[rule["paired_standard_id"]]
        rule_q = {item["id"]: item for item in rule["questions"]}
        standard_q = {item["id"]: item for item in standard["questions"]}
        unmarked_q = {item["id"]: item for item in unmarked["questions"]}
        built_windows = []
        assigned = []
        for name, process, spec in windows:
            steps = []
            earlier: list[dict] = []
            for blank_id, layer in spec:
                item = rule_q[blank_id]
                base = standard_q[blank_id]
                if item["process"] != process:
                    raise RuntimeError(f"{factor} ID{blank_id} 不在{process}，实际在{item['process']}。")
                options = list(item.get("options") or [])
                if options != list(base.get("options") or []):
                    raise RuntimeError(f"{factor} ID{blank_id} 的选项与标准稿不同。")
                if blank_id in unmarked_q and unmarked_q[blank_id]["answer"] != item["answer"]:
                    raise RuntimeError(f"{factor} 无标记稿 ID{blank_id} 的答案与先问规则稿不同。")
                if blank_id in {q["id"] for q in marked["questions"]}:
                    marked_answer = next(q["answer"] for q in marked["questions"] if q["id"] == blank_id)
                    if marked_answer != item["answer"]:
                        raise RuntimeError(f"{factor} 有标记稿 ID{blank_id} 的答案与先问规则稿不同。")
                format_code = item.get("format_code") or ""
                abnormal = str(item["answer"])
                standard_answer = str(base["answer"])
                if not _differs(abnormal, standard_answer, format_code, options):
                    raise RuntimeError(f"{factor} ID{blank_id} 的反常答案与标准答案相同，不能放入契约链条。")
                earlier_results = tuple(step["blank_id"] for step in earlier if step["layer"] in RESULT_LAYERS)
                earlier_rules = tuple(step["blank_id"] for step in earlier if step["layer"] == "recognition")
                clean = layer in {"recognition", "execution"} and not earlier_results
                step = {
                    "blank_id": blank_id,
                    "process": process,
                    "layer": layer,
                    "prompt": item.get("prompt") or "",
                    "label": item.get("label") or "",
                    "format_code": format_code,
                    "options": options,
                    "abnormal": abnormal,
                    "standard": standard_answer,
                    "clean": clean,
                    "earlier_results": earlier_results,
                    "earlier_rules": earlier_rules,
                }
                steps.append(step)
                earlier.append(step)
                assigned.append(blank_id)
            built_windows.append({"name": name, "process": process, "steps": steps})
        differing = []
        for blank_id, item in rule_q.items():
            base = standard_q.get(blank_id)
            if base is None:
                continue
            if _differs(str(item["answer"]), str(base["answer"]), item.get("format_code") or "", list(item.get("options") or [])):
                differing.append(blank_id)
        if sorted(differing, key=int) != sorted(assigned, key=int):
            missing = sorted(set(differing) - set(assigned), key=int)
            extra = sorted(set(assigned) - set(differing), key=int)
            raise RuntimeError(f"{factor} 链条与答案差异不一致，未列入 {missing}，多列入 {extra}。")
        excluded = set(unmarked["excluded_ids"])
        same_rule = []
        for blank_id in sorted(excluded, key=int):
            if blank_id in assigned:
                continue
            item = rule_q[blank_id]
            base = standard_q[blank_id]
            if _differs(str(item["answer"]), str(base["answer"]), item.get("format_code") or "", list(item.get("options") or [])):
                raise RuntimeError(f"{factor} 删去的 ID{blank_id} 答案已改变，却没有进入链条。")
            same_rule.append(blank_id)
        if excluded != set(assigned).intersection(excluded) | set(same_rule):
            raise RuntimeError(f"{factor} 的规则询问没有全部有着落。")
        background = []
        for item in rule["questions"]:
            if item.get("label") != "C1" or item["id"] in excluded:
                continue
            background.append({
                "blank_id": item["id"],
                "process": item["process"],
                "prompt": item.get("prompt") or "",
                "format_code": item.get("format_code") or "",
                "options": list(item.get("options") or []),
                "answer": str(item["answer"]),
            })
        factors.append({
            "factor": factor,
            "title": str(rule.get("title") or factor).split("·", 1)[0],
            "standard_id": rule["paired_standard_id"],
            "windows": built_windows,
            "same_rule": same_rule,
            "background": background,
            "present": {
                "unmarked": {item["id"] for item in unmarked["questions"]},
                "marked": {item["id"] for item in marked["questions"]},
                "rule": set(rule_q),
            },
        })
    return {"factors": factors}


def flatten_records(rows: list[dict]) -> list[dict]:
    flat = []
    for row in rows:
        condition = canonical_condition(str(row.get("condition") or ""))
        if not condition:
            raise RuntimeError(f"无法识别评测条件 {row.get('condition')!r}。")
        shared = {
            "sample_id": str(row.get("sample_id") or ""),
            "condition": condition,
            "run": int(row.get("run") or 0),
            "model": str(row.get("model") or "未标注"),
        }
        blanks = row.get("blanks")
        if isinstance(blanks, list):
            for blank in blanks:
                flat.append({
                    **shared,
                    "blank_id": str(blank.get("blank_id") or blank.get("id") or ""),
                    "prediction": str(blank.get("prediction") or ""),
                })
            continue
        if row.get("blank_id"):
            flat.append({**shared, "blank_id": str(row["blank_id"]), "prediction": str(row.get("prediction") or "")})
    seen = set()
    for item in flat:
        key = (item["sample_id"], item["condition"], item["run"], item["model"], item["blank_id"])
        if key in seen:
            raise RuntimeError(f"重复记录 {key}。")
        seen.add(key)
    return flat


def _index_factors(catalog: dict) -> dict[str, dict]:
    return {factor["factor"]: factor for factor in catalog["factors"]}


def score_predictions(catalog: dict, rows: list[dict]) -> list[dict]:
    """给每条模型回答贴上层级、三分类和是否命中标准答案。"""
    factors = _index_factors(catalog)
    scored = []
    for item in flatten_records(rows):
        parsed = _split_sample(item["sample_id"])
        if parsed is None or item["condition"] not in {GOLD, SELF}:
            continue
        factor_name, family, stem = parsed
        factor = factors[factor_name]
        if item["blank_id"] not in factor["present"][family]:
            continue
        found = None
        window_name = ""
        for window in factor["windows"]:
            for step in window["steps"]:
                if step["blank_id"] == item["blank_id"]:
                    found = step
                    window_name = window["name"]
                    break
        background = next((entry for entry in factor["background"] if entry["blank_id"] == item["blank_id"]), None)
        if found is None and background is None:
            continue
        if found is not None:
            outcome = classify_outcome(
                item["prediction"], found["abnormal"], found["standard"], found["format_code"], found["options"],
            )
            scored.append({
                **item,
                "factor": factor_name,
                "stem": stem,
                "family": family,
                "arm": _arm(family, item["condition"]),
                "window": window_name,
                "layer": found["layer"],
                "clean": found["clean"],
                "role": "target",
                "outcome": outcome,
                "answer_correct": outcome == "abnormal",
                "default_prior_match": outcome == "standard",
                "lure_eligible": True,
            })
            continue
        correct = _matches(item["prediction"], background["answer"], background["format_code"], background["options"])
        scored.append({
            **item,
            "factor": factor_name,
            "stem": stem,
            "family": family,
            "arm": _arm(family, item["condition"]),
            "window": "",
            "layer": "background",
            "clean": False,
            "role": "background",
            "outcome": "abnormal" if correct else "other",
            "answer_correct": correct,
            "default_prior_match": False,
            "lure_eligible": False,
        })
    return scored


def _groups(records: list[dict], predicate) -> list[list[bool]]:
    buckets: dict[tuple, list[bool]] = defaultdict(list)
    for item in records:
        if not predicate(item):
            continue
        key = (item["model"], item["stem"], item["run"])
        buckets[key].append(bool(item["answer_correct"]))
    return [buckets[key] for key in sorted(buckets)]


def _rate(groups: list[list[bool]]) -> dict:
    total = sum(len(group) for group in groups)
    hit = sum(sum(group) for group in groups)
    if not total:
        return {"hit": 0, "total": 0, "point": None, "low": None, "high": None, "units": 0}
    point = hit / total
    units = len(groups)
    if units < 2:
        low = high = point
    else:
        low, high = bootstrap_ci(groups)
    return {"hit": hit, "total": total, "point": point, "low": low, "high": high, "units": units}


def _paired_delta(left: list[dict], right: list[dict], predicate) -> dict | None:
    def pack(records):
        buckets: dict[tuple, list[bool]] = defaultdict(list)
        for item in records:
            if predicate(item):
                buckets[(item["stem"], item["run"])].append(bool(item["answer_correct"]))
        return buckets

    left_map = pack(left)
    right_map = pack(right)
    keys = sorted(set(left_map) & set(right_map))
    if not keys:
        return None

    def value(chosen, source) -> float:
        groups = [source[key] for key in chosen]
        total = sum(len(group) for group in groups)
        hit = sum(sum(group) for group in groups)
        return hit / total if total else 0.0

    point = value(keys, left_map) - value(keys, right_map)
    if len(keys) < 2:
        low = high = point
    else:
        draw = []
        rng_seed = 20261102
        import random
        rng = random.Random(rng_seed)
        for _ in range(2000):
            chosen = [keys[rng.randrange(len(keys))] for _ in keys]
            draw.append(value(chosen, left_map) - value(chosen, right_map))
        draw.sort()
        low = draw[int(0.025 * (len(draw) - 1))]
        high = draw[int(0.975 * (len(draw) - 1))]
    discordant_left = 0
    discordant_right = 0
    for key in keys:
        left_ok = all(left_map[key])
        right_ok = all(right_map[key])
        if left_ok and not right_ok:
            discordant_left += 1
        elif right_ok and not left_ok:
            discordant_right += 1
    return {
        "point": point,
        "low": low,
        "high": high,
        "units": len(keys),
        "only_left": discordant_left,
        "only_right": discordant_right,
        "p": _mcnemar(discordant_left, discordant_right),
    }


def _mcnemar(only_left: int, only_right: int) -> float | None:
    n = only_left + only_right
    if n == 0:
        return None
    k = min(only_left, only_right)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def _window_records(scored: list[dict], model: str, factor: str, window: str) -> list[dict]:
    return [
        item for item in scored
        if item["model"] == model and item["factor"] == factor and item["window"] == window and item["role"] == "target"
    ]


def _select(records: list[dict], family: str, condition: str, blank_id: str | None = None, layer: str | None = None, clean: bool | None = None) -> list[dict]:
    chosen = []
    for item in records:
        if item["family"] != family or item["condition"] != condition:
            continue
        if blank_id is not None and item["blank_id"] != blank_id:
            continue
        if layer is not None and item["layer"] != layer:
            continue
        if clean is not None and item["clean"] != clean:
            continue
        chosen.append(item)
    return chosen


def _fmt_rate(rate: dict) -> str:
    if not rate or rate["point"] is None:
        return "未测"
    text = f"{rate['point']:.1%}（{rate['hit']}/{rate['total']}）"
    if rate["units"] >= 2 and rate["low"] is not None:
        text += f"，95% 区间 {rate['low']:.1%}–{rate['high']:.1%}"
    elif rate["units"] == 1:
        text += "，仅 1 个配对单元"
    return text


def _fmt_delta(delta: dict | None) -> str:
    if delta is None:
        return "未测"
    text = f"{delta['point']:+.1%}"
    if delta["units"] >= 2:
        text += f"，95% 区间 {delta['low']:+.1%}–{delta['high']:+.1%}"
    else:
        text += "，仅 1 个配对单元"
    if delta["p"] is not None:
        text += f"；McNemar 仅左对 {delta['only_left']}、仅右对 {delta['only_right']}，精确双侧 p={delta['p']:.3f}"
    elif delta["only_left"] == 0 and delta["only_right"] == 0:
        text += "；两侧没有分歧对"
    return text


def _trials(records: list[dict], family: str, condition: str, steps: list[dict]) -> list[dict]:
    buckets: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for item in records:
        if item["family"] != family or item["condition"] != condition:
            continue
        buckets[(item["stem"], item["run"])][item["blank_id"]] = item
    trials = []
    for key in sorted(buckets):
        got = buckets[key]
        if not got:
            continue
        trials.append({"stem": key[0], "run": key[1], "answers": got})
    return trials


def _chain_flags(trial: dict, steps: list[dict], present: set[str]) -> dict:
    answers = trial["answers"]
    asked = [step for step in steps if step["blank_id"] in present and step["blank_id"] in answers]

    def layer_value(layer: str, clean_only: bool):
        chosen = [step for step in asked if step["layer"] == layer and (step["clean"] or not clean_only)]
        if not chosen:
            return None
        return all(answers[step["blank_id"]]["outcome"] == "abnormal" for step in chosen)

    recognition = layer_value("recognition", True)
    execution = layer_value("execution", True)
    reasoning = layer_value("reasoning", False)
    derivation = layer_value("derivation", False)
    clean_exec = [step for step in asked if step["layer"] == "execution" and step["clean"]]
    default_prior = None
    if clean_exec:
        default_prior = any(answers[step["blank_id"]]["outcome"] == "standard" for step in clean_exec)
    consistency = None
    if recognition is True:
        later = [step for step in asked if step["layer"] != "recognition"]
        consistency = all(answers[step["blank_id"]]["outcome"] == "abnormal" for step in later) if later else True
    execution_hold = None
    if execution is True:
        later = [
            step for step in asked
            if not (step["clean"] and step["layer"] in {"recognition", "execution"})
        ]
        execution_hold = all(answers[step["blank_id"]]["outcome"] == "abnormal" for step in later) if later else True
    return {
        "rule_recognition": recognition,
        "rule_execution": execution,
        "spatial_reasoning": reasoning,
        "numeric_derivation": derivation,
        "default_prior_match": default_prior,
        "process_consistency": consistency,
        "execution_hold": execution_hold,
        "answers": answers,
    }


def _conditional(trials: list[dict], steps: list[dict], present: set[str], given, target) -> dict:
    flags = [_chain_flags(trial, steps, present) for trial in trials]
    usable = [item for item in flags if given(item)]
    if not usable:
        return {"hit": 0, "total": 0, "point": None}
    hit = sum(bool(target(item)) for item in usable)
    return {"hit": hit, "total": len(usable), "point": hit / len(usable)}


def _joint(trials: list[dict], steps: list[dict], present: set[str], derivation_id: str) -> dict:
    index = {step["blank_id"]: position for position, step in enumerate(steps)}
    if derivation_id not in index:
        return {}
    position = index[derivation_id]
    priors = [step["blank_id"] for step in steps[:position] if step["blank_id"] in present]
    cells = {"过程对，答案对": 0, "过程对，答案错": 0, "过程错，答案对": 0, "过程错，答案错": 0}
    used = 0
    for trial in trials:
        answers = trial["answers"]
        if derivation_id not in answers or any(blank_id not in answers for blank_id in priors):
            continue
        used += 1
        process_ok = all(answers[blank_id]["outcome"] == "abnormal" for blank_id in priors)
        answer_ok = answers[derivation_id]["outcome"] == "abnormal"
        if process_ok and answer_ok:
            cells["过程对，答案对"] += 1
        elif process_ok and not answer_ok:
            cells["过程对，答案错"] += 1
        elif not process_ok and answer_ok:
            cells["过程错，答案对"] += 1
        else:
            cells["过程错，答案错"] += 1
    hidden = None
    correct_answers = cells["过程对，答案对"] + cells["过程错，答案对"]
    if correct_answers:
        hidden = cells["过程错，答案对"] / correct_answers
    return {"cells": cells, "used": used, "hidden": hidden, "priors": priors}


def _three_way(records: list[dict]) -> dict[str, int]:
    counts = {"abnormal": 0, "standard": 0, "other": 0}
    for item in records:
        counts[item["outcome"]] += 1
    return counts


def _conclusions(window_steps: list[dict], records: list[dict], present: dict[str, set[str]]) -> list[str]:
    clean_exec = [step["blank_id"] for step in window_steps if step["layer"] == "execution" and step["clean"]]
    reasoning_ids = [step["blank_id"] for step in window_steps if step["layer"] == "reasoning"]
    derivation_ids = [step["blank_id"] for step in window_steps if step["layer"] == "derivation"]

    def pooled(family, condition, ids):
        chosen = [
            item for item in records
            if item["family"] == family and item["condition"] == condition and item["blank_id"] in ids
        ]
        return _rate(_groups(chosen, lambda item: True))

    unmarked_exec = pooled("unmarked", GOLD, set(clean_exec))
    marked_exec = pooled("marked", GOLD, set(clean_exec))
    rule_exec = pooled("rule", GOLD, set(clean_exec))
    delta_marker = _paired_delta(
        _select(records, "marked", GOLD),
        _select(records, "unmarked", GOLD),
        lambda item: item["blank_id"] in clean_exec,
    )
    delta_rule = _paired_delta(
        _select(records, "rule", GOLD),
        _select(records, "unmarked", GOLD),
        lambda item: item["blank_id"] in clean_exec,
    )

    def lure(family, condition):
        chosen = _select(records, family, condition)
        chosen = [item for item in chosen if item["blank_id"] in clean_exec]
        if not chosen:
            return None
        return sum(item["outcome"] == "standard" for item in chosen) / len(chosen)

    lines = []
    lure_u = lure("unmarked", GOLD)
    lure_m = lure("marked", GOLD)
    if (
        delta_marker is not None and delta_marker["point"] > 0
        and lure_u is not None and lure_m is not None and lure_u > lure_m
        and unmarked_exec["point"] is not None and marked_exec["point"] is not None
    ):
        note = "方向符合结论 A：有标记的局部执行高于无标记，且无标记更容易输出标准答案。"
        if delta_marker["units"] < 2 or not (delta_marker["low"] > 0):
            note += " 配对单元不足或差值区间含 0，这仍是描述，不是已证实的差异。"
        lines.append(note)
    if delta_rule is not None and delta_rule["point"] > 0 and rule_exec["point"] is not None:
        note = "方向符合结论 B：规则以正确答案外化之后，局部执行高于无标记。"
        note += " gold-trunc 上更后面的符号和公式空已经带有前面的正确结果，那些空的回升只说明模型能顺着给定结果继续写。"
        if delta_rule["units"] < 2 or not (delta_rule["low"] > 0):
            note += " 差值区间尚未离开 0。"
        lines.append(note)
    model_trials = _trials(records, "rule", SELF, window_steps)
    if model_trials and clean_exec:
        flags = [_chain_flags(trial, window_steps, present["rule"]) for trial in model_trials]
        recognized = [item for item in flags if item["rule_recognition"] is True]
        slipped = [
            item for item in recognized
            if item["rule_execution"] is False or item["spatial_reasoning"] is False
        ]
        if recognized and slipped and (delta_rule is None or delta_rule["point"] <= 0):
            lines.append(
                "方向符合结论 C：先问规则的模型轨迹里，规则识别对了，但局部执行或后续推理仍错；"
                "正确外化也没有把局部执行抬到无标记之上。这是规则没有稳定用下去。"
            )
        held = [
            item for item in flags
            if item["rule_recognition"] is not False
            and item["rule_execution"] is True
            and item["spatial_reasoning"] is not False
        ]
        if held and derivation_ids:
            failed = [item for item in held if item["numeric_derivation"] is False]
            if len(failed) * 2 > len(held):
                lines.append(
                    "方向符合结论 D：规则识别、局部执行和后续符号都对的试次里，公式或矩阵仍有一半以上错误。"
                    "这些错误归到推导，不归到工程契约没有读到。"
                )
    if reasoning_ids or derivation_ids:
        later_ids = set(reasoning_ids + derivation_ids)
        later = _paired_delta(
            _select(records, "unmarked", GOLD),
            _select(records, "unmarked", SELF),
            lambda item: item["blank_id"] in later_ids,
        )
        early = _paired_delta(
            _select(records, "unmarked", GOLD),
            _select(records, "unmarked", SELF),
            lambda item: item["blank_id"] in clean_exec,
        )
        if later is not None and later["point"] > 0 and (early is None or later["point"] > early["point"]):
            lines.append(
                "方向符合结论 E：无标记稿上，后续步骤的完整模型前缀传播损失大于局部执行。"
                "这里的损失是 gold-trunc 减 self-trunc，self-trunc 使用模型此前的全部回答。"
            )
    return lines


def render_catalog(catalog: dict) -> list[str]:
    lines = ["# 契约指标目录", ""]
    lines.append("局部执行准确率只统计干净执行空。干净指该空之前没有更早的执行、推理或推导结果。")
    lines.append("识别准确率只统计干净识别空。出现在执行结果之后的识别空是后置复述，单独列出。")
    lines.append("答案与标准稿相同的规则空不进入诱饵率，也不进入识别准确率。")
    lines.append("")
    for factor in catalog["factors"]:
        lines.append(f"## {factor['title']}")
        lines.append("")
        lines.append(f"标准稿 `{factor['standard_id']}`。背景规则题 {len(factor['background'])} 个。")
        if factor["same_rule"]:
            lines.append("删去但答案未变的规则空：" + "、".join(factor["same_rule"]) + "。")
        lines.append("")
        for window in factor["windows"]:
            lines.append(f"### {window['name']}")
            lines.append("")
            lines.append("| 空 | 层 | 计入主指标 | A 条件前缀中已有的目标结果 | 题意 |")
            lines.append("| --- | --- | --- | --- | --- |")
            for step in window["steps"]:
                use = "是" if step["clean"] else "否"
                prior = "、".join(step["earlier_results"]) or "无"
                prompt = step["prompt"].replace("|", "｜")
                if len(prompt) > 36:
                    prompt = prompt[:36] + "…"
                lines.append(
                    f"| ID{step['blank_id']} | {LAYER_TITLE[step['layer']]} | {use} | {prior} | {prompt} |"
                )
            lines.append("")
    return lines


def render_report(catalog: dict, scored: list[dict]) -> str:
    lines = render_catalog(catalog)
    if not scored:
        lines.append("还没有评测记录。目录用来固定空的层级；分数在记录进入后按模型分窗计算。")
        lines.append("")
        return "\n".join(lines)
    models = sorted({item["model"] for item in scored})
    factors = _index_factors(catalog)
    for model in models:
        lines.append(f"# 模型 {model}")
        lines.append("")
        for factor in catalog["factors"]:
            lines.append(f"## {factor['title']}")
            lines.append("")
            background = [
                item for item in scored
                if item["model"] == model and item["factor"] == factor["factor"] and item["role"] == "background"
            ]
            for family, condition, arm in (
                ("rule", GOLD, "rule-gold"),
                ("rule", SELF, "rule-model"),
            ):
                subset = [item for item in background if item["family"] == family and item["condition"] == condition]
                target = [
                    item for item in scored
                    if item["model"] == model and item["factor"] == factor["factor"]
                    and item["role"] == "target" and item["family"] == family and item["condition"] == condition
                    and item["layer"] == "recognition" and item["clean"]
                ]
                if subset or target:
                    lines.append(
                        f"- {arm}：目标规则识别 {_fmt_rate(_rate(_groups(target, lambda item: True)))}；"
                        f"背景规则 {_fmt_rate(_rate(_groups(subset, lambda item: True)))}。"
                    )
            lines.append("")
            for window in factor["windows"]:
                records = _window_records(scored, model, factor["factor"], window["name"])
                lines.extend(_render_window(window, records, factor["present"]))
    lines.append("## 统计口径")
    lines.append("")
    lines.append("准确率和差值按（场景, 运行）重抽样。当前每种定义只有一个场景时，一个运行就是一个配对单元，区间会等于点估计。")
    lines.append("McNemar 把同一单元上两侧是否全对作为配对结果，精确双侧概率按二项分布计算。")
    lines.append("场景数和模型数都增加之后，可以用 logit(p)=β0+β1·有标记+β2·先问规则+β3·模型+题目随机效应+模型随机效应。当前先报告配对差值。")
    lines.append("过程是否正确，只看本窗口链条上已经问到的空。两侧答案相同的分量空不能区分契约，不放进这个过程标志。")
    lines.append("")
    return "\n".join(lines)


def _render_window(window: dict, records: list[dict], present: dict[str, set[str]]) -> list[str]:
    steps = window["steps"]
    lines = [f"### {window['name']}", ""]
    header = "| 条件 | " + " | ".join(f"ID{step['blank_id']}" for step in steps) + " |"
    align = "| --- | " + " | ".join("---:" for _ in steps) + " |"
    lines.extend([header, align])
    arms = (
        ("unmarked", GOLD, "unmarked", "Unmarked"),
        ("marked", GOLD, "marked", "Marked"),
        ("rule", GOLD, "rule-gold", "Rule-Gold"),
        ("rule", SELF, "rule-model", "Rule-Model"),
    )
    for family, condition, _arm_name, label in arms:
        cells = []
        for step in steps:
            if step["blank_id"] not in present[family]:
                cells.append("不适用")
                continue
            rate = _rate(_groups(_select(records, family, condition, step["blank_id"]), lambda item: True))
            cells.append(_fmt_rate(rate))
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("干净执行空的三分类：")
    lines.append("")
    lines.append("| 条件 | 反常答案 | 标准答案诱饵 | 其他错误 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for family, condition, _arm_name, label in arms:
        chosen = [
            item for item in _select(records, family, condition)
            if item["clean"] and item["layer"] == "execution"
        ]
        if not chosen:
            lines.append(f"| {label} | 未测 | 未测 | 未测 |")
            continue
        counts = _three_way(chosen)
        total = sum(counts.values())
        lines.append(
            "| {label} | {a} | {s} | {o} |".format(
                label=label,
                a=f"{counts['abnormal']}/{total}",
                s=f"{counts['standard']}/{total}",
                o=f"{counts['other']}/{total}",
            )
        )
    lines.append("")
    clean_ids = [step["blank_id"] for step in steps if step["clean"] and step["layer"] == "execution"]
    lines.append(
        "- 标记收益（Marked−Unmarked，干净执行）："
        + _fmt_delta(_paired_delta(_select(records, "marked", GOLD), _select(records, "unmarked", GOLD), lambda item: item["blank_id"] in clean_ids))
    )
    lines.append(
        "- 规则外化收益（Rule-Gold−Unmarked，干净执行）："
        + _fmt_delta(_paired_delta(_select(records, "rule", GOLD), _select(records, "unmarked", GOLD), lambda item: item["blank_id"] in clean_ids))
    )
    rule_gold_exec = _rate(_groups(_select(records, "rule", GOLD), lambda item: item["blank_id"] in clean_ids))
    if rule_gold_exec["point"] is None:
        lines.append("- 外化后的剩余执行错误：未测")
    else:
        lines.append(f"- 外化后的剩余执行错误：{1 - rule_gold_exec['point']:.1%}")
    lines.append("")
    lines.append("| 目标空 | Acc(gold-trunc) | Acc(self-trunc) | 完整前缀传播损失 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for step in steps:
        if step["blank_id"] not in present["unmarked"]:
            continue
        left = _select(records, "unmarked", GOLD, step["blank_id"])
        right = _select(records, "unmarked", SELF, step["blank_id"])
        lines.append(
            "| ID{blank} | {a} | {b} | {loss} |".format(
                blank=step["blank_id"],
                a=_fmt_rate(_rate(_groups(left, lambda item: True))),
                b=_fmt_rate(_rate(_groups(right, lambda item: True))),
                loss=_fmt_delta(_paired_delta(left, right, lambda item: True)),
            )
        )
    lines.append("")
    model_trials = _trials(records, "rule", SELF, steps)
    if model_trials:
        present_rule = present["rule"]
        given_rule = lambda item: item["rule_recognition"] is True
        lines.append("Rule-Model 轨迹上的条件概率：")
        lines.append("")
        exec_given = _conditional(model_trials, steps, present_rule, given_rule, lambda item: item["rule_execution"] is True)
        lines.append(f"- P(局部执行正确 | 规则识别正确) = {_fmt_fraction(exec_given)}")
        for step in steps:
            if step["layer"] != "reasoning":
                continue
            cond = _conditional(
                model_trials, steps, present_rule, given_rule,
                lambda item, blank=step["blank_id"]: item["answers"].get(blank, {}).get("outcome") == "abnormal",
            )
            lines.append(f"- P(ID{step['blank_id']}正确 | 规则识别正确) = {_fmt_fraction(cond)}")
        if any(step["layer"] == "reasoning" for step in steps) and any(step["layer"] == "derivation" for step in steps):
            held = lambda item: item["rule_recognition"] is True and item["rule_execution"] is True and item["spatial_reasoning"] is True
            derived = _conditional(model_trials, steps, present_rule, held, lambda item: item["numeric_derivation"] is True)
            lines.append(f"- P(推导正确 | 识别、执行、后续推理都正确) = {_fmt_fraction(derived)}")
        cons = _conditional(model_trials, steps, present_rule, given_rule, lambda item: item["process_consistency"] is True)
        lines.append(f"- 规则保持一致性 = {_fmt_fraction(cons)}")
        lines.append("")
    unmarked_trials = _trials(records, "unmarked", SELF, steps)
    if unmarked_trials:
        hold = _conditional(
            unmarked_trials, steps, present["unmarked"],
            lambda item: item["rule_execution"] is True,
            lambda item: item["execution_hold"] is True,
        )
        lines.append(f"无标记自身轨迹上，P(后续推理和推导都正确 | 局部执行正确) = {_fmt_fraction(hold)}")
        lines.append("")
    for step in steps:
        if step["layer"] != "derivation":
            continue
        joint = _joint(_trials(records, "rule", SELF, steps), steps, present["rule"], step["blank_id"])
        if not joint or not joint["used"]:
            continue
        lines.append(f"Rule-Model 上 ID{step['blank_id']} 的过程—答案联合分布（过程看 ID{'、'.join(joint['priors']) or '无'}）：")
        lines.append("")
        for name, count in joint["cells"].items():
            lines.append(f"- {name}：{count}/{joint['used']}")
        if joint["hidden"] is not None:
            lines.append(f"- P(过程错 | 答案对) = {joint['hidden']:.1%}")
        lines.append("")
    for line in _conclusions(steps, records, present):
        lines.append(line)
        lines.append("")
    return lines


def _fmt_fraction(rate: dict) -> str:
    if rate["point"] is None:
        return "未测"
    return f"{rate['point']:.1%}（{rate['hit']}/{rate['total']}）"


def _prediction_rows(sample_id: str, condition: str, run: int, answers: dict[str, str], model: str = "自检") -> list[dict]:
    return [
        {
            "sample_id": sample_id,
            "condition": condition,
            "run": run,
            "model": model,
            "blank_id": blank_id,
            "prediction": prediction,
        }
        for blank_id, prediction in answers.items()
    ]


def _check() -> None:
    catalog = load_catalog()
    factors = _index_factors(catalog)
    alpha = factors["alpha"]
    first = alpha["windows"][0]
    by_id = {step["blank_id"]: step for step in first["steps"]}
    if not by_id["22"]["clean"] or by_id["68"]["clean"] or by_id["80"]["clean"]:
        raise RuntimeError("理论迎角窗口的干净执行空应只有 ID22。")
    if by_id["68"]["earlier_results"] != ("22",) or by_id["80"]["earlier_results"] != ("22", "68"):
        raise RuntimeError("ID68、ID80 的前缀依赖与题面顺序不一致。")
    if "21" in alpha["present"]["unmarked"] or "21" not in alpha["present"]["rule"]:
        raise RuntimeError("ID21 应只出现在先问规则稿。")
    elastic = {step["blank_id"]: step for step in alpha["windows"][1]["steps"]}
    if not elastic["158"]["clean"] or elastic["164"]["earlier_results"] != ("158",):
        raise RuntimeError("弹性窗口的干净执行空应是 ID152。")
    roll = {step["blank_id"]: step for step in factors["roll"]["windows"][0]["steps"]}
    if not roll["36"]["clean"] or roll["42"]["clean"] or roll["42"]["layer"] != "execution" or "36" not in roll["42"]["earlier_results"]:
        raise RuntimeError("滚转的干净执行空应是 ID36，ID42 是执行题但前缀已经含有 ID36。")
    order = factors["order"]
    if order["same_rule"] != ["5", "6", "7", "11", "18", "19", "21", "27", "35", "84", "92", "93", "94", "96", "151", "155"]:
        raise RuntimeError(f"顺序配对中答案未变的规则空不符：{order['same_rule']}。")
    body = factors["body"]
    if body["same_rule"] != ["5", "6", "11", "16", "17", "18", "19", "21", "27", "35", "84", "92", "93", "96", "151", "155"]:
        raise RuntimeError(f"体轴配对中答案未变的规则空不符：{body['same_rule']}。")
    moment = body["windows"][2]
    if moment["process"] != "力矩参考点" or not moment["steps"][0]["clean"]:
        raise RuntimeError("力矩参考点窗口应从 ID183 的干净执行开始。")

    gold = {step["blank_id"]: step["abnormal"] for step in first["steps"]}
    standard = {step["blank_id"]: step["standard"] for step in first["steps"]}
    gold_rows = _prediction_rows("alpha-rule", GOLD, 0, gold)
    standard_rows = _prediction_rows("alpha-rule", GOLD, 0, standard)
    gold_scored = score_predictions(catalog, gold_rows)
    if any(item["outcome"] != "abnormal" for item in gold_scored if item["window"] == "理论迎角"):
        raise RuntimeError("反常标准答案没有被判为反常答案。")
    standard_scored = score_predictions(catalog, standard_rows)
    if any(item["outcome"] != "standard" for item in standard_scored if item["window"] == "理论迎角"):
        raise RuntimeError("标准稿答案没有被判为标准答案诱饵。")
    other = score_predictions(catalog, _prediction_rows("alpha-rule", GOLD, 0, {key: "???" for key in gold}))
    if any(item["outcome"] != "other" for item in other if item["window"] == "理论迎角"):
        raise RuntimeError("无关输出没有被判为其他错误。")

    rows = []
    rows += _prediction_rows("alpha-unmarked", GOLD, 0, {"22": standard["22"], "68": gold["68"], "80": gold["80"]})
    rows += _prediction_rows("alpha-marked", GOLD, 0, {"22": gold["22"], "68": gold["68"], "80": gold["80"]})
    rows += _prediction_rows("alpha-rule", GOLD, 0, gold)
    rows += _prediction_rows("alpha-unmarked", SELF, 0, {"22": gold["22"], "68": standard["68"], "80": standard["80"]})
    rows += _prediction_rows("alpha-rule", SELF, 0, {"21": standard["21"], "22": standard["22"], "68": standard["68"], "80": standard["80"]})
    rows += _prediction_rows("alpha-rule", SELF, 1, gold)
    background = alpha["background"][0]
    rows += _prediction_rows("alpha-rule", GOLD, 0, {background["blank_id"]: "???"})
    scored = score_predictions(catalog, rows)
    window_rows = [item for item in scored if item["window"] == "理论迎角"]
    marker = _paired_delta(
        _select(window_rows, "marked", GOLD),
        _select(window_rows, "unmarked", GOLD),
        lambda item: item["blank_id"] == "22",
    )
    if marker is None or abs(marker["point"] - 1) > 1e-9:
        raise RuntimeError(f"标记收益应为 100%，实际 {marker}。")
    external = _paired_delta(
        _select(window_rows, "rule", GOLD),
        _select(window_rows, "unmarked", GOLD),
        lambda item: item["blank_id"] == "22",
    )
    if external is None or abs(external["point"] - 1) > 1e-9:
        raise RuntimeError(f"规则外化收益应为 100%，实际 {external}。")
    loss = _paired_delta(
        _select(window_rows, "unmarked", GOLD),
        _select(window_rows, "unmarked", SELF),
        lambda item: item["blank_id"] == "80",
    )
    if loss is None or abs(loss["point"] - 1) > 1e-9:
        raise RuntimeError(f"ID77 的完整前缀传播损失应为 100%，实际 {loss}。")
    early_loss = _paired_delta(
        _select(window_rows, "unmarked", GOLD),
        _select(window_rows, "unmarked", SELF),
        lambda item: item["blank_id"] == "22",
    )
    if early_loss is None or abs(early_loss["point"] + 1) > 1e-9:
        raise RuntimeError("ID22 在 gold-trunc 上错、self-trunc 上对，传播损失应为 -100%。")
    target_rate = _rate(_groups(
        [item for item in scored if item["layer"] == "recognition" and item["clean"] and item["condition"] == GOLD],
        lambda item: True,
    ))
    if target_rate["point"] != 1:
        raise RuntimeError("目标规则识别被背景错题带低了。")
    background_rate = _rate(_groups(
        [item for item in scored if item["role"] == "background"],
        lambda item: True,
    ))
    if background_rate["point"] != 0:
        raise RuntimeError("背景规则错题没有被单独计上。")
    trials = _trials(window_rows, "rule", SELF, first["steps"])
    cond = _conditional(
        trials, first["steps"], alpha["present"]["rule"],
        lambda item: item["rule_recognition"] is True,
        lambda item: item["rule_execution"] is True,
    )
    if cond["hit"] != 1 or cond["total"] != 1:
        raise RuntimeError(f"识别正确后的执行条件概率应为 1/1，实际 {cond}。")
    joint = _joint(trials, first["steps"], alpha["present"]["rule"], "80")
    if joint["cells"]["过程对，答案对"] != 1 or joint["cells"]["过程错，答案错"] != 1:
        raise RuntimeError(f"过程—答案联合分布不符：{joint['cells']}。")
    conclusions = _conclusions(first["steps"], window_rows, alpha["present"])
    if not any("结论 A" in line for line in conclusions):
        raise RuntimeError(f"没有得出结论 A：{conclusions}")
    if not any("结论 B" in line for line in conclusions):
        raise RuntimeError(f"没有得出结论 B：{conclusions}")
    if not any("结论 E" in line for line in conclusions):
        raise RuntimeError(f"没有得出结论 E：{conclusions}")
    text = render_report(catalog, [])
    if "ID22" not in text or "干净执行空" not in text:
        raise RuntimeError("空目录没有写出层级。")
    print("指标自检通过")


def main() -> None:
    parser = argparse.ArgumentParser(description="计算单因素工程契约的四层指标")
    parser.add_argument("records", nargs="?", help="cloze_eval 的 records.jsonl；不给则只检查目录并做自检")
    parser.add_argument("--out", help="把报告写到这个文件；默认打印")
    args = parser.parse_args()
    _check()
    catalog = load_catalog()
    if not args.records:
        report = render_report(catalog, [])
    else:
        rows = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines() if line.strip()]
        report = render_report(catalog, score_predictions(catalog, rows))
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
    else:
        print(report)


if __name__ == "__main__":
    main()
