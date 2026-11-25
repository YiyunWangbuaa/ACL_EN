# -*- coding: utf-8 -*-
"""Session fork for the ID196 kernel: sign, force, coordinate, denominator.

The control is one session-model chat of 力矩参考点, stopped at ID196.
The fork copies that chat, deletes the ID196 assistant reply, and replaces
each wrong kernel reply (ID190, ID202, ID208, ID213) with its gold JSON.
Slots the model already matched stay as written. --gold-nodes is not used.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import (  # noqa: E402
    RunStop,
    SESSION_MODEL,
    _guard_session,
    _resolve_max_tokens,
    _session_question,
    _session_system,
    contexts_for,
    gold_reply,
    load_samples,
    score_one,
    standard_for,
)
from domains.wind_tunnel.session_probe import answers_to_clear, clear_turn, forks_after  # noqa: E402
from semi_auto_eval import load_env_file, make_client  # noqa: E402

DATA = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
PROCESS = "力矩参考点"
TARGET_ID = "196"
KERNEL = ("190", "202", "208", "213")
FORK_CONDITION = "session-fork"
BODY_GOLD = {
    "190": "{190:A}",
    "202": "{202:C}",
    "208": "{208:A}",
    "213": "{213:D}",
}


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentences_through(chain: dict, stop_id: str) -> list[dict]:
    kept = []
    for sentence in chain["sentences"]:
        kept.append(sentence)
        if stop_id in {str(item) for item in sentence["blank_ids"]}:
            return kept
    raise RuntimeError(f"{chain['process']} 没有 ID{stop_id}")


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


def _picked(result: dict, blank_id: str) -> dict:
    for blank in result.get("blanks") or []:
        if str(blank["blank_id"]) == str(blank_id):
            return _slim(blank)
    raise KeyError(blank_id)


def _turn_record(result: dict, question: str, committed: str) -> dict:
    return {
        "blank_ids": [str(item) for item in result["blank_ids"]],
        "question": question,
        "committed": committed,
        "response": result.get("response") or "",
        "error": result.get("error") or "",
        "unanswered": bool(result.get("unanswered")),
        "finish_reason": result.get("finish_reason") or "",
        "latency_ms": result.get("latency_ms") or 0,
        "blanks": [_slim(blank) for blank in result.get("blanks") or []],
    }


def _failed(record: dict) -> bool:
    return bool(record.get("error") or record.get("unanswered"))


def _job(chain: dict, sentence: dict, history: list[dict], run: int, seed: int, max_tokens: int, standard: dict, condition: str) -> dict:
    return {
        "sample_id": chain["sample_id"],
        "tier": chain["tier"],
        "process": chain["process"],
        "group": "main",
        "blank_ids": [str(item) for item in sentence["blank_ids"]],
        "blanks": sentence["blanks"],
        "condition": condition,
        "run": run,
        "seed": seed,
        "max_tokens": max_tokens,
        "stop": None,
        "standard": standard,
        "messages": history,
        "prompt": "",
        "session": chain["process"],
    }


def _commit(messages: list[dict], question: str, reply: str, sentence: dict, chain: dict) -> str:
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": reply})
    clear_ids = answers_to_clear(chain, sentence["blank_ids"])
    if clear_ids:
        gold = {
            str(blank["blank_id"]): str(blank.get("answer") or "")
            for item in chain["sentences"]
            for blank in item["blanks"]
        }
        clear_turn(
            messages,
            {"blank_ids": [str(item) for item in sentence["blank_ids"]], "assistant_index": len(messages) - 1},
            clear_ids,
            gold,
        )
    return messages[-1]["content"]


def _replay(messages: list[dict], question: str, committed: str) -> None:
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": committed})


def _blank_of(record: dict, blank_id: str) -> dict:
    for blank in record.get("blanks") or []:
        if str(blank.get("blank_id")) == str(blank_id):
            return blank
    raise KeyError(blank_id)


def kernel_replacements(
    indexes: dict[str, int],
    blanks: dict[str, dict],
    gold: dict[str, str],
) -> list[tuple[int, str]]:
    """Wrong kernel slots only. A normalized match keeps the model's own text."""
    found = []
    for blank_id in KERNEL:
        if blanks[blank_id].get("normalized"):
            continue
        found.append((indexes[blank_id], gold[blank_id]))
    return found


def build_fork(messages: list[dict], replacements: list[tuple[int, str]], question: str) -> list[dict]:
    """Copy the chat through the turn before ID196, then swap the wrong kernel replies."""
    if not replacements:
        raise RuntimeError("没有需要换回金标的核槽")
    if messages[-1]["role"] != "assistant":
        raise RuntimeError("分叉时会话末尾应是 ID196 之前的助手回复")
    fork = [{"role": item["role"], "content": item["content"]} for item in messages]
    for index, gold_text in replacements:
        if fork[index]["role"] != "assistant":
            raise RuntimeError("核槽的位置不是助手回复")
        if fork[index]["content"] == gold_text:
            raise RuntimeError("这个核槽的原文已经是金标")
        fork[index]["content"] = gold_text
    fork.append({"role": "user", "content": question})
    return fork


def prepare(sample: str) -> tuple[dict, list[dict], dict[str, str]]:
    rows = load_samples(DATA, {sample})
    chain = next(item for item in contexts_for(rows[0]) if item["process"] == PROCESS)
    sentences = _sentences_through(chain, TARGET_ID)
    target = sentences[-1]
    if target["blank_ids"] != [TARGET_ID]:
        raise RuntimeError(f"ID{TARGET_ID} 不是单独一句")
    gold = {}
    for sentence in sentences:
        ids = [str(item) for item in sentence["blank_ids"]]
        if len(ids) == 1 and ids[0] in KERNEL:
            if ids[0] in gold:
                raise RuntimeError(f"ID{ids[0]} 出现了两次")
            gold[ids[0]] = gold_reply(sentence["blanks"])
    if set(gold) != set(KERNEL):
        raise RuntimeError(f"核槽不齐：{sorted(set(KERNEL) - set(gold))}")
    for sentence in sentences:
        if forks_after(chain, sentence["blank_ids"]):
            raise RuntimeError(f"{PROCESS} 会插入探针，主会话不再只是原稿回合")
        if answers_to_clear(chain, sentence["blank_ids"]):
            raise RuntimeError(f"ID{sentence['blank_ids']} 的助手回复会被 clear_turn 清掉")
    return chain, sentences, gold


def check_shape(sample: str) -> None:
    chain, sentences, gold = prepare(sample)
    if sample == "body-rule" and gold != BODY_GOLD:
        raise RuntimeError(f"body-rule 的核金标应为 {BODY_GOLD}，实际是 {gold}")
    order = [str(item) for sentence in sentences for item in sentence["blank_ids"]]
    for earlier, later in (("194", "201"), ("213", "195"), ("195", "196")):
        if order.index(earlier) > order.index(later):
            raise RuntimeError(f"ID{earlier} 没有排在 ID{later} 前面")
    messages: list[dict] = [{"role": "system", "content": _session_system(chain)}]
    previous = None
    indexes: dict[str, int] = {}
    replies = {blank_id: "{" + blank_id + ":Z}" for blank_id in order}
    replies["190"] = "{190:B}"
    replies["202"] = "{202:CZ}"
    replies["208"] = gold["208"]
    replies["213"] = "{213:A}"
    for sentence in sentences[:-1]:
        question = _session_question(chain, sentence, previous)
        _guard_session([*messages, {"role": "user", "content": question}], chain["declaration"])
        ids = [str(item) for item in sentence["blank_ids"]]
        reply = replies[ids[0]] if len(ids) == 1 and ids[0] in replies else (
            "{" + ", ".join(f"{blank_id}:Z" for blank_id in ids) + "}"
        )
        committed = _commit(messages, question, reply, sentence, chain)
        if committed != reply:
            raise RuntimeError(f"ID{ids} 的助手回复被主会话改写")
        if len(ids) == 1 and ids[0] in KERNEL:
            indexes[ids[0]] = len(messages) - 1
        previous = sentence
    if set(indexes) != set(KERNEL):
        raise RuntimeError("四个核槽没有各自进入会话")
    question = _session_question(chain, sentences[-1], previous)
    for blank_id in KERNEL:
        if f"{{{blank_id}:" in question:
            raise RuntimeError(f"ID196 的用户消息带进了 ID{blank_id}")
    if "正" in question or "负" in question:
        raise RuntimeError("ID196 的用户消息带进了符号")
    blanks = {
        "190": {"normalized": False},
        "202": {"normalized": True},
        "208": {"normalized": True},
        "213": {"normalized": False},
    }
    replacements = kernel_replacements(indexes, blanks, gold)
    if [gold_text for _index, gold_text in replacements] != [gold["190"], gold["213"]]:
        raise RuntimeError(f"应只替换 ID190 和 ID213，实际是 {replacements}")
    fork = build_fork(messages, replacements, question)
    if fork[indexes["190"]]["content"] != gold["190"] or fork[indexes["213"]]["content"] != gold["213"]:
        raise RuntimeError("分叉没有把答错的核槽换成金标")
    if fork[indexes["202"]]["content"] != "{202:CZ}" or fork[indexes["208"]]["content"] != gold["208"]:
        raise RuntimeError("分叉改写了已经答对的核槽")
    if messages[indexes["190"]]["content"] != "{190:B}":
        raise RuntimeError("分叉写回了对照会话")
    if fork[-1] != {"role": "user", "content": question}:
        raise RuntimeError("分叉没有停在 ID196 的用户消息")
    if any(item["role"] == "assistant" and item["content"].startswith("{196:") for item in fork):
        raise RuntimeError("分叉里不该有 ID196 的旧回答")
    for marker in ("{191:Z}", "{201:Z}"):
        index = next(i for i, item in enumerate(messages) if item["role"] == "assistant" and item["content"] == marker)
        if fork[index]["content"] != marker:
            raise RuntimeError(f"分叉改动了核以外的助手回复 {marker}")
    if kernel_replacements(indexes, {blank_id: {"normalized": True} for blank_id in KERNEL}, gold):
        raise RuntimeError("四个核槽都答对时仍要求替换")
    try:
        build_fork(messages, [], question)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("空替换仍生成了分叉")
    print(
        f"形状检查通过：{sample} {PROCESS}，核金标 {gold}，ID196 用户消息 {len(question)} 字。",
        flush=True,
    )


def _call(client, model, temperature, thinking, job):
    return score_one(client, model, temperature, thinking, job)


def _ask(client, model, temperature, thinking, chain, sentence, messages, question, run, seed, max_tokens, standard, condition) -> dict:
    history = [*messages, {"role": "user", "content": question}]
    _guard_session(history, chain["declaration"])
    result = _call(
        client, model, temperature, thinking,
        _job(chain, sentence, history, run, seed, max_tokens, standard, condition),
    )
    return result


def _invalidate_after(run_dir: Path, index: int) -> None:
    for path in run_dir.glob("turn_*.json"):
        number = int(path.stem.split("_")[1])
        if number >= index:
            path.unlink()
    for name in ("control.json", "fork.json"):
        path = run_dir / name
        if path.is_file():
            path.unlink()


def _kernel_snapshot(run_dir: Path) -> dict[str, dict]:
    found = {}
    if not run_dir.is_dir():
        return found
    for path in sorted(run_dir.glob("turn_*.json")):
        record = _load(path)
        ids = [str(item) for item in record.get("blank_ids") or []]
        if len(ids) != 1 or ids[0] not in KERNEL or _failed(record):
            continue
        found[ids[0]] = _blank_of(record, ids[0])
    return found


def run_trial(chain, sentences, gold, client, model, temperature, thinking, run, seed, max_tokens, standard, out: Path, control_only: bool) -> dict:
    run_dir = out / f"run_{run:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    before = sentences[:-1]
    target = sentences[-1]
    messages: list[dict] = [{"role": "system", "content": _session_system(chain)}]
    previous = None
    indexes: dict[str, int] = {}
    for index, sentence in enumerate(before):
        path = run_dir / f"turn_{index:02d}.json"
        question = _session_question(chain, sentence, previous)
        ids = [str(item) for item in sentence["blank_ids"]]
        if path.is_file():
            record = _load(path)
            if record.get("question") != question:
                raise RuntimeError(f"第 {run + 1} 趟 ID{ids} 的题面与已保存的不一致")
            if not _failed(record):
                _replay(messages, question, record["committed"])
                if len(ids) == 1 and ids[0] in KERNEL:
                    indexes[ids[0]] = len(messages) - 1
                previous = sentence
                continue
        _invalidate_after(run_dir, index)
        print(f"第 {run + 1} 趟 提问 ID{','.join(ids)}", flush=True)
        result = _ask(
            client, model, temperature, thinking, chain, sentence, messages, question,
            run, seed, max_tokens, standard, SESSION_MODEL,
        )
        reply = result.get("response") or ""
        committed = _commit(messages, question, reply, sentence, chain) if not (result.get("error") or result.get("unanswered")) else reply
        record = _turn_record(result, question, committed)
        _save(path, record)
        if _failed(record):
            print(f"第 {run + 1} 趟 ID{ids[0]} 调用失败：{record['error']}", flush=True)
            return {"run": run, "status": "failed", "where": ids}
        if committed != reply:
            raise RuntimeError(f"ID{ids} 的助手回复被主会话改写")
        if len(ids) == 1 and ids[0] in KERNEL:
            indexes[ids[0]] = len(messages) - 1
        previous = sentence
    if set(indexes) != set(KERNEL):
        raise RuntimeError("四个核槽没有各自进入会话")
    question = _session_question(chain, target, previous)
    kernel = _kernel_snapshot(run_dir)
    wrong_ids = [blank_id for blank_id in KERNEL if not kernel[blank_id].get("normalized")]
    control_path = run_dir / "control.json"
    if control_path.is_file() and not _failed(_load(control_path)):
        control = _load(control_path)
    else:
        print(f"第 {run + 1} 趟 对照提问 ID196（错槽 {','.join(wrong_ids) or '无'}）", flush=True)
        result = _ask(
            client, model, temperature, thinking, chain, target, messages, question,
            run, seed, max_tokens, standard, SESSION_MODEL,
        )
        control = _turn_record(result, question, result.get("response") or "")
        control["blank"] = _picked(result, TARGET_ID) if result.get("blanks") else {}
        control["wrong_ids"] = wrong_ids
        _save(control_path, control)
    if _failed(control):
        print(f"第 {run + 1} 趟 ID196 对照调用失败：{control.get('error')}", flush=True)
        return {"run": run, "status": "failed", "where": [TARGET_ID], "kernel": kernel, "wrong_ids": wrong_ids}
    control_blank = control.get("blank") or _picked_saved(control)
    if not wrong_ids:
        status = "residue" if _wrong(control_blank) else "clean"
        print(f"第 {run + 1} 趟 四个核槽都是金标，状态 {status}", flush=True)
        return {"run": run, "status": status, "kernel": kernel, "wrong_ids": [], "control": control_blank}
    if control_only:
        print(f"第 {run + 1} 趟 对照已记下，错槽 {','.join(wrong_ids)}，暂不分叉", flush=True)
        return {"run": run, "status": "control", "kernel": kernel, "wrong_ids": wrong_ids, "control": control_blank}
    replacements = kernel_replacements(indexes, kernel, gold)
    injected = {blank_id: gold[blank_id] for blank_id in wrong_ids}
    fork_path = run_dir / "fork.json"
    if fork_path.is_file() and not _failed(_load(fork_path)):
        fork = _load(fork_path)
    else:
        print(f"第 {run + 1} 趟 分叉：{injected}，重问 ID196", flush=True)
        forked = build_fork(messages, replacements, question)
        for index, gold_text in replacements:
            if forked[index]["content"] != gold_text or messages[index]["content"] == gold_text:
                raise RuntimeError("分叉替换不符合对照/干预的分工")
        _guard_session(forked, chain["declaration"])
        result = _call(
            client, model, temperature, thinking,
            _job(chain, target, forked, run, seed, max_tokens, standard, FORK_CONDITION),
        )
        fork = _turn_record(result, question, result.get("response") or "")
        fork["injected"] = injected
        fork["blank"] = _picked(result, TARGET_ID) if result.get("blanks") else {}
        _save(fork_path, fork)
    if _failed(fork):
        print(f"第 {run + 1} 趟 ID196 分叉调用失败：{fork.get('error')}", flush=True)
        return {"run": run, "status": "failed", "where": [TARGET_ID], "kernel": kernel, "wrong_ids": wrong_ids, "control": control_blank}
    fork_blank = fork.get("blank") or _picked_saved(fork)
    print(
        f"第 {run + 1} 趟 保留：对照 {control_blank.get('prediction')!r}，分叉 {fork_blank.get('prediction')!r}",
        flush=True,
    )
    return {
        "run": run,
        "status": "kept",
        "kernel": kernel,
        "wrong_ids": wrong_ids,
        "control": control_blank,
        "fork": fork_blank,
    }


def _picked_saved(record: dict) -> dict:
    blanks = record.get("blanks") or []
    for blank in blanks:
        if str(blank.get("blank_id")) == TARGET_ID:
            return blank
    return record.get("blank") or {}


def _wrong(blank: dict) -> bool:
    return not bool(blank.get("normalized"))


def fisher_exact(wrong_control: int, right_control: int, wrong_fork: int, right_fork: int) -> float | None:
    """Two-sided Fisher exact test on the two conditions, treated as independent."""
    a, b, c, d = wrong_control, right_control, wrong_fork, right_fork
    n = a + b + c + d
    if n == 0 or a + b == 0 or c + d == 0:
        return None
    row1 = a + b
    col1 = a + c

    def probability(x: int) -> float:
        fork_wrong = col1 - x
        row2 = n - row1
        if x < 0 or fork_wrong < 0 or x > row1 or fork_wrong > row2:
            return 0.0
        return math.comb(col1, x) * math.comb(n - col1, row1 - x) / math.comb(n, row1)

    observed = probability(a)
    low = max(0, row1 - (n - col1))
    high = min(row1, col1)
    total = sum(probability(x) for x in range(low, high + 1) if probability(x) <= observed + 1e-15)
    return min(1.0, total)


def two_proportion_z(wrong_1: int, n_1: int, wrong_2: int, n_2: int) -> tuple[float, float] | None:
    if n_1 == 0 or n_2 == 0:
        return None
    p1 = wrong_1 / n_1
    p2 = wrong_2 / n_2
    pooled = (wrong_1 + wrong_2) / (n_1 + n_2)
    if pooled in (0.0, 1.0):
        return None
    z = (p1 - p2) / math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_1 + 1.0 / n_2))
    p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return z, p_value


def mcnemar_exact(rescued: int, harmed: int) -> float | None:
    """Two-sided exact test on the discordant pairs."""
    n = rescued + harmed
    if n == 0:
        return None
    observed = math.comb(n, rescued) / (2 ** n)
    total = sum(math.comb(n, k) / (2 ** n) for k in range(n + 1) if math.comb(n, k) / (2 ** n) <= observed + 1e-15)
    return min(1.0, total)


def collect(out: Path, runs: int) -> list[dict]:
    rows = []
    for run in range(runs):
        run_dir = out / f"run_{run:02d}"
        if not run_dir.is_dir() or not any(run_dir.glob("turn_*.json")):
            continue
        kernel = _kernel_snapshot(run_dir)
        wrong_ids = [blank_id for blank_id in KERNEL if blank_id in kernel and not kernel[blank_id].get("normalized")]
        control_path = run_dir / "control.json"
        fork_path = run_dir / "fork.json"
        if set(kernel) != set(KERNEL) or not control_path.is_file():
            rows.append({"run": run, "status": "failed", "kernel": kernel, "wrong_ids": wrong_ids})
            continue
        control = _load(control_path)
        if _failed(control):
            rows.append({"run": run, "status": "failed", "kernel": kernel, "wrong_ids": wrong_ids})
            continue
        control_blank = control.get("blank") or _picked_saved(control)
        base = {"run": run, "kernel": kernel, "wrong_ids": wrong_ids, "control": control_blank}
        if not wrong_ids:
            rows.append({**base, "status": "residue" if _wrong(control_blank) else "clean"})
            continue
        if not fork_path.is_file():
            rows.append({**base, "status": "control"})
            continue
        fork = _load(fork_path)
        if _failed(fork):
            rows.append({**base, "status": "failed"})
            continue
        rows.append({**base, "status": "kept", "fork": fork.get("blank") or _picked_saved(fork)})
    return rows


def render(sample: str, model: str, thinking: str, effort: str, runs: int, rows: list[dict]) -> str:
    kept = [row for row in rows if row["status"] == "kept"]
    waiting = [row for row in rows if row["status"] == "control"]
    residue = [row for row in rows if row["status"] == "residue"]
    clean = [row for row in rows if row["status"] == "clean"]
    failed = [row for row in rows if row["status"] == "failed"]
    scored = [row for row in rows if row.get("kernel") and set(row["kernel"]) == set(KERNEL)]
    control_wrong = sum(_wrong(row["control"]) for row in kept)
    fork_wrong = sum(_wrong(row["fork"]) for row in kept)
    n = len(kept)
    rescued = sum(_wrong(row["control"]) and not _wrong(row["fork"]) for row in kept)
    harmed = sum((not _wrong(row["control"])) and _wrong(row["fork"]) for row in kept)
    both_wrong = sum(_wrong(row["control"]) and _wrong(row["fork"]) for row in kept)
    both_right = n - rescued - harmed - both_wrong
    same_text = sum(
        (row["control"].get("prediction") or "") == (row["fork"].get("prediction") or "")
        for row in kept
        if _wrong(row["control"]) and _wrong(row["fork"])
    )
    fisher = fisher_exact(control_wrong, n - control_wrong, fork_wrong, n - fork_wrong) if n else None
    z_test = two_proportion_z(control_wrong, n, fork_wrong, n) if n else None
    paired = mcnemar_exact(rescued, harmed) if n else None
    p_prop = control_wrong / n if n else None
    p_fix = fork_wrong / n if n else None

    def rate(wrong: int) -> str:
        if n == 0:
            return "—"
        return f"{wrong}/{n} = {wrong / n:.0%}"

    lines = [
        "# 会话分叉：ID196 的四层核",
        "",
        f"样本 {sample}，子过程{PROCESS}。模型 {model}，思考 {thinking}，强度 {effort}，温度 0。",
        "对照是 session-model：定义只在系统消息，ID181 到 ID196 逐句作答，助手回复保持模型原文。",
        "分叉复制这条会话，删掉 ID196 的助手回复，把 ID190、ID202、ID208、ID213 里答错的槽换成金标 JSON，再重问 ID196。",
        "已经与金标归一化相等的核槽保持模型原文。其余槽不改、不重问。",
        "四个核槽都是金标的趟不分叉：ID196 仍错记为公式层残留，ID196 也对记为干净。",
        "思维链不写回下一轮。这一步测的是可见会话，不是同一次生成内部的隐藏状态。",
        "",
        f"计划 {runs} 趟，完成记录 {len(rows)} 趟：分叉 {n}，待分叉 {len(waiting)}，公式层残留 {len(residue)}，干净 {len(clean)}，调用失败 {len(failed)}。",
        "",
        "| 槽 | 对照答错 |",
        "| --- | ---: |",
    ]
    for blank_id in (*KERNEL, TARGET_ID):
        if blank_id == TARGET_ID:
            wrong = sum(_wrong(row["control"]) for row in scored if row.get("control"))
            base = sum(1 for row in scored if row.get("control"))
        else:
            wrong = sum(not row["kernel"][blank_id].get("normalized") for row in scored)
            base = len(scored)
        shown = "—" if base == 0 else f"{wrong}/{base} = {wrong / base:.0%}"
        lines.append(f"| ID{blank_id} | {shown} |")
    gold_kernel = len(residue) + len(clean)
    if gold_kernel == 0:
        residual = "—"
    else:
        residual = f"{len(residue)}/{gold_kernel} = {len(residue) / gold_kernel:.0%}"
    lines.extend([
        "",
        f"四槽全对时 ID196 仍错：{residual}。分母是四个核槽都与金标归一化相等的趟。",
        "",
        "上表用全部走完四个核槽的对照。下面两个误差率只用至少有一个核槽答错、并且已经重问过 ID196 的趟。",
        "",
        "| 量 | 值 |",
        "| --- | --- |",
        f"| 进入分叉 | {n}/{len(rows) or runs} |",
        f"| p_对照 | {rate(control_wrong)} |",
        f"| p_修复 | {rate(fork_wrong)} |",
        f"| 效应 p_对照 − p_修复 | {'—' if p_prop is None else f'{(p_prop - p_fix):.0%}'} |",
        f"| Fisher 精确检验 | {'—' if fisher is None else f'{fisher:.4f}'} |",
        f"| 两比例 z | {'—' if z_test is None else f'z={z_test[0]:.2f}，p={z_test[1]:.4f}'} |",
        f"| 配对精确检验 | {'—' if paired is None else f'{paired:.4f}'} |",
        "",
        "Fisher 和 z 把同一趟的对照、分叉当成两个独立样本。配对精确检验只数两边不一致的趟。",
        "",
        "| 对照错、分叉对 | 对照对、分叉错 | 两边都错 | 两边都对 |",
        "| ---: | ---: | ---: | ---: |",
        f"| {rescued} | {harmed} | {both_wrong} | {both_right} |",
        "",
        f"两边都错的 {both_wrong} 趟里，分叉文本与对照相同的有 {same_text} 趟，其余 {both_wrong - same_text} 趟换了式子，归一化后仍不等于金标。",
        f"公式层残留 {len(residue)} 趟：四个核槽都是金标，ID196 仍错，不进入上面两个误差率。",
        "",
        "| 趟 | 状态 | 错的核槽 | 对照 ID196 | 分叉 ID196 |",
        "| ---: | --- | --- | --- | --- |",
    ])
    labels = {
        "kept": "已分叉",
        "control": "待分叉",
        "residue": "公式层残留",
        "clean": "干净",
        "failed": "调用失败",
    }
    for row in rows:
        control = (row.get("control") or {}).get("prediction") or ""
        fork = (row.get("fork") or {}).get("prediction") or ""
        if row["status"] == "failed":
            control = control or "调用失败"
            fork = fork or "未做"
        elif row["status"] != "kept":
            fork = "—"
        wrong = "、".join(f"ID{blank_id}" for blank_id in row.get("wrong_ids") or []) or "—"
        lines.append(f"| {row['run'] + 1} | {labels.get(row['status'], row['status'])} | {wrong} | {control} | {fork} |")
    lines.append("")
    if waiting and n == 0:
        lines.append("对照已记下，分叉还没跑。")
    elif n == 0:
        lines.append("没有核槽答错的趟，修复率没有分母。")
    elif p_fix is not None and p_prop is not None and p_fix + 0.5 <= p_prop and fisher is not None and fisher < 0.05:
        lines.append("完整核换回金标后，ID196 的误差率下降，Fisher p<0.05。")
    elif p_fix is not None and p_prop is not None and abs(p_prop - p_fix) <= 0.1:
        lines.append("完整核换回金标后，ID196 的误差率基本没变。")
    else:
        lines.append("两个误差率有差别，这一批的检验还没有把它钉成下降或没变。")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="body-rule")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=ROOT / "analysis" / "会话分叉_ID196核.md")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs 至少为 1。")
    out = args.out or (ROOT / "results" / "fork_id196_kernel" / args.sample)
    check_shape(args.sample)
    if args.check:
        if fisher_exact(10, 0, 0, 10) >= 0.001:
            raise RuntimeError("Fisher 没有把全部分叉改对判成小概率")
        if fisher_exact(10, 0, 10, 0) != 1:
            raise RuntimeError("Fisher 没有把两边同样全错判成 1")
        if mcnemar_exact(10, 0) >= 0.01:
            raise RuntimeError("配对检验没有把 10 次只在分叉改对判成小概率")
        from domains.wind_tunnel.injector_catalog import e3_rows
        variants = {
            row.question[0]: row.variant
            for row in e3_rows("body-rule")
            if row.question
        }
        found = {key: variants.get(key) for key in ("190", "202", "213")}
        if found != {"190": "sign", "202": "choice", "213": "choice"}:
            raise RuntimeError(f"力矩测点的行标记不对：{found}")
        print("检验函数通过。", flush=True)
        return
    if not args.report_only:
        load_env_file(ROOT / ".env")
        os.environ["DEEPSEEK_REASONING_EFFORT"] = args.effort
        chain, sentences, gold = prepare(args.sample)
        standard = standard_for("wind_tunnel", DATA)
        max_tokens = _resolve_max_tokens(0, args.thinking, announce=False)
        client = make_client(timeout=1800.0 if args.thinking == "enabled" else 600.0)
        meta = {
            "sample": args.sample,
            "model": args.model,
            "thinking": args.thinking,
            "effort": args.effort,
            "temperature": args.temperature,
            "seed": args.seed,
            "runs": args.runs,
            "injected": gold,
            "kernel": list(KERNEL),
            "condition": SESSION_MODEL,
            "fork": FORK_CONDITION,
        }
        meta_path = out / "meta.json"
        if meta_path.is_file():
            saved = _load(meta_path)
            comparable = {key: saved.get(key) for key in ("sample", "model", "thinking", "effort", "injected", "kernel")}
            current = {key: meta[key] for key in comparable}
            if comparable != current:
                raise SystemExit(f"已有结果的条件与这次不一致：{comparable}")
        _save(meta_path, meta)
        print(
            f"样本 {args.sample} 模型 {args.model} 思考 {args.thinking} 强度 {args.effort} "
            f"核 {gold} 趟数 {args.runs} {'只跑对照' if args.control_only else '对照后分叉'}",
            flush=True,
        )
        try:
            for run in range(args.runs):
                run_trial(
                    chain, sentences, gold, client, args.model, args.temperature, args.thinking,
                    run, args.seed + run, max_tokens, standard, out, args.control_only,
                )
        except RunStop as exc:
            print(f"调用中断：{exc}", flush=True)
    rows = collect(out, args.runs)
    text = render(args.sample, args.model, args.thinking, args.effort, args.runs, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
