# -*- coding: utf-8 -*-
"""One robotics review, thinking off, gold prefix, stop at the current period.

The kinematics review holds two samples. Each subprocess is injected alone:
earlier blanks in that subprocess are filled with the option text or the open
answer, and later sentences are omitted. Probe text, layer notes and generator
notes stay out. Answers are not written forward; each call is one fresh prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cloze_eval import (
    ROOT,
    _dependency_fields,
    _preface,
    _record_key,
    _resolve_max_tokens,
    _resolve_workers,
    score_one,
    summarize,
    transcript,
)
from cloze_score import resolve_choice
from semi_auto_eval import load_env_file, make_client, resolve_api_config

REVIEW = ROOT / "data" / "robotics" / "review" / "kinematics" / "sample_01.md"
FLOW = ROOT / "data" / "flight" / "review" / "flow" / "sample_01.md"
SLOT_RE = re.compile(
    r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}"
)
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _options(shown: str) -> list[str]:
    if shown.strip() == "无":
        return []
    found = []
    for part in shown.split("；"):
        _letter, sep, text = part.partition("：")
        if not sep:
            raise RuntimeError(f"选项缺少分隔：{part}")
        found.append(text)
    return found


def _gold(item: dict) -> str:
    if item["options"]:
        _letter, text = resolve_choice(item["answer"], item["options"])
        return text
    return item["answer"]


def _open(item: dict) -> str:
    if item["options"]:
        shown = "；".join(
            f"{LETTERS[index]}：{text}" for index, text in enumerate(item["options"])
        )
    else:
        shown = "无"
    return f"{{ID:{item['id']}, 选项:{shown}, 输出格式:{item['format_code']}}}"


def _section(text: str, heading: str, stop: str) -> str:
    start = text.index(heading) + len(heading)
    end = text.find(stop, start)
    if end < 0:
        end = len(text)
    return text[start:end].strip()


def _sentences(body: str) -> list[dict]:
    pieces = []
    cursor = 0
    for match in re.finditer("。", body):
        chunk = body[cursor:match.end()]
        cursor = match.end()
        slots = list(SLOT_RE.finditer(chunk))
        if len(slots) > 1:
            raise RuntimeError(f"一句里有多个空：{chunk[:80]}")
        item = None
        if slots:
            slot = slots[0]
            options = _options(slot.group(2))
            item = {
                "id": slot.group(1),
                "options": options,
                "format_code": slot.group(3),
                "label": slot.group(4),
                "answer": slot.group(5),
                "span": (slot.start(), slot.end()),
            }
        pieces.append({"text": chunk, "item": item})
    tail = body[cursor:].strip()
    if tail:
        raise RuntimeError(f"子过程末尾没有句号：{tail[:80]}")
    return pieces


def _fill(sentence: dict, mode: str) -> str:
    item = sentence["item"]
    if item is None:
        return sentence["text"].strip()
    start, end = item["span"]
    replacement = _open(item) if mode == "open" else _gold(item)
    return (sentence["text"][:start] + replacement + sentence["text"][end:]).strip()


def load_samples(path: Path) -> tuple[str, list[dict]]:
    raw = path.read_text(encoding="utf-8")
    spec = _section(raw, "## 输出格式规范", "\n# ")
    forward = _section(raw, "# 正运动学样本", "\n# 逆运动学样本")
    inverse = _section(raw, "# 逆运动学样本", "\n# 层间对应")
    samples = [
        _sample("kinematics-forward", "正运动学", forward),
        _sample("kinematics-inverse", "逆运动学", inverse),
    ]
    return spec, samples


def _sample(sample_id: str, title: str, text: str) -> dict:
    declaration = _section(text, "## 定义声明", "\n## 物理常识")
    prior = _section(text, "## 物理常识", "\n## 完整过程")
    process = text.split("## 完整过程", 1)[1]
    chains, seen = _chains(title, process, declaration, prior)
    expected = [str(number) for number in range(1, len(seen) + 1)]
    if seen != expected:
        raise RuntimeError(f"{title} 的题号是 {seen[:8]}…{seen[-4:]}，应当是 1 到 {len(seen)}。")
    return {
        "sample_id": sample_id,
        "title": title,
        "declaration": declaration,
        "prior": prior,
        "chains": chains,
    }


def _chains(title: str, process: str, declaration: str, prior: str) -> tuple[list[dict], list[str]]:
    parts = re.split(r"\n### ", process)
    chains = []
    seen = []
    for part in parts[1:]:
        name, _, body = part.partition("\n")
        sentences = _sentences(body.strip() + ("\n" if not body.endswith("\n") else ""))
        ids = [item["item"]["id"] for item in sentences if item["item"] is not None]
        if not ids:
            raise RuntimeError(f"{title} / {name.strip()} 没有空。")
        seen.extend(ids)
        chains.append({
            "process": name.strip(),
            "sentences": sentences,
            "ids": ids,
            "declaration": declaration,
            "prior": prior,
            "session": title,
        })
    return chains, seen


def load_flow(path: Path) -> tuple[str, list[dict]]:
    raw = path.read_text(encoding="utf-8")
    spec = _section(raw, "## 输出格式规范", "\n## 定义声明")
    declaration = _section(raw, "## 定义声明", "\n## 物理常识")
    prior = _section(raw, "## 物理常识", "\n## 完整过程")
    rest = raw.split("## 完整过程", 1)[1]
    coordinate, _, aero = rest.partition("\n## 气动力与参考点定义")
    aero = aero.split("\n# ", 1)[0]
    chains, seen = _chains("坐标转换", coordinate, declaration, prior)
    heading, _, aero_body = aero.partition("\n### ")
    if not aero_body:
        raise RuntimeError("气动力定义后面没有子过程。")
    aero_chains, aero_ids = _chains("气动力", "\n### " + aero_body, heading.strip(), "")
    chains.extend(aero_chains)
    seen.extend(aero_ids)
    expected = [str(number) for number in range(1, len(seen) + 1)]
    if seen != expected:
        raise RuntimeError(f"飞行力学题号是 {seen[:8]}…{seen[-4:]}，应当是 1 到 {len(seen)}。")
    if len(chains) != 6:
        raise RuntimeError(f"飞行力学应有 6 个子过程，实际 {len(chains)}。")
    for chain in chains[:4]:
        chain["session"] = "坐标转换"
    chains[4]["session"] = "气动力"
    chains[5]["session"] = "参考点"
    return spec, [{
        "sample_id": "flow-standard",
        "title": "统一默认",
        "declaration": declaration,
        "prior": prior,
        "chains": chains,
        "group": "flight",
    }]


def _prompt(spec: str, sample: dict, chain: dict, index: int) -> str:
    sentences = chain["sentences"]
    current = sentences[index]
    item = current["item"]
    if item is None:
        raise RuntimeError("当前句没有空。")
    earlier = "\n".join(_fill(sentence, "gold") for sentence in sentences[:index] if sentence["text"].strip())
    asked = _fill(current, "open")
    body = "\n".join(part for part in (earlier, asked) if part)
    if item["options"]:
        example = f'{{"{item["id"]}":"A"}}'
        value_rule = "选择题的值只写选项字母。"
    else:
        example = f'{{"{item["id"]}":"答案"}}'
        value_rule = "矩阵、向量、公式、关节角、距离、范围、任务姿态角和表达式都写成 JSON 字符串。"
    return "\n".join([
        "只填写当前这一句。本次是分步问答，只输出一个 JSON 对象。",
        "键是空的编号字符串，值是答案。不要解释，不要填写其他空。",
        "",
        "【输出格式规范】",
        spec.strip(),
        "",
        "【定义声明】",
        chain.get("declaration", sample["declaration"]).strip(),
        "",
        *([
            "【物理常识】",
            chain.get("prior", sample["prior"]).strip(),
            "",
        ] if chain.get("prior", sample["prior"]).strip() else []),
        "【子过程】",
        chain["process"],
        "",
        "【过程】",
        body,
        "",
        "【输出】",
        f"只输出一行 {example}。",
        value_rule,
        "不要输出句子原文，不要使用 LaTeX。",
    ])


def _turn_question(chain: dict, index: int) -> str:
    sentences = chain["sentences"]
    start = 0
    for earlier in range(index - 1, -1, -1):
        if sentences[earlier]["item"] is not None:
            start = earlier + 1
            break
    chunks = [sentence["text"].strip() for sentence in sentences[start:index]]
    chunks.append(_fill(sentences[index], "open"))
    return "\n".join(chunk for chunk in chunks if chunk)


def _gold_turn(item: dict) -> str:
    if item["options"]:
        return "{" + f'{item["id"]}:{item["answer"]}' + "}"
    escaped = item["answer"].replace("\\", "\\\\").replace('"', '\\"')
    return "{" + f'{item["id"]}:"{escaped}"' + "}"


def _system_message(spec: str, chain: dict) -> str:
    parts = [
        "本会话只在这条系统消息里给出一次定义、物理常识和输出格式。",
        "后面的每一问只追加当前句子。请从本消息检索工程定义，不要等待定义被重复。",
        "",
        "【定义声明】",
        chain["declaration"].strip(),
    ]
    if chain["prior"].strip():
        parts.extend(["", "【物理常识】", chain["prior"].strip()])
    parts.extend([
        "",
        "【输出格式规范】",
        spec.strip(),
        "",
        "当前这一问只输出一个 JSON 对象。键是空的编号字符串，值是答案。",
        "选择题的值只写字母。其他答案用英文双引号包成字符串。不要解释。",
    ])
    return "\n".join(parts)


def _session_groups(samples: list[dict]) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    order: list[tuple] = []
    for sample in samples:
        for chain in sample["chains"]:
            key = (sample["sample_id"], chain["session"])
            if key not in grouped:
                grouped[key] = {
                    "sample": sample,
                    "session": chain["session"],
                    "chains": [],
                }
                order.append(key)
            grouped[key]["chains"].append(chain)
    return [grouped[key] for key in order]


def _session_turns(spec: str, group: dict, condition: str) -> list[dict]:
    chain = group["chains"][0]
    system = _system_message(spec, chain)
    marker = chain["declaration"].strip()[:40]
    messages = [{"role": "system", "content": system}]
    turns = []
    for link in group["chains"]:
        if link["declaration"] != chain["declaration"]:
            raise RuntimeError(f"会话 {group['session']} 里混入了另一份定义。")
        targets = [index for index, sentence in enumerate(link["sentences"]) if sentence["item"] is not None]
        for index in targets:
            item = link["sentences"][index]["item"]
            question = "\n".join([
                "填写当前这一句。",
                "",
                "【子过程】",
                link["process"],
                "",
                "【问题】",
                _turn_question(link, index),
            ])
            if marker and marker in question:
                raise RuntimeError(f"{link['process']} ID {item['id']} 的问题重复了定义声明。")
            history = [*messages, {"role": "user", "content": question}]
            blank = {
                "blank_id": item["id"],
                "label": item["label"],
                "format_code": item["format_code"],
                "answer": item["answer"],
                "options": list(item["options"]),
                "process": link["process"],
            }
            turns.append({
                "turn": len(turns),
                "sample_id": group["sample"]["sample_id"],
                "tier": 0,
                "process": link["process"],
                "group": group["sample"].get("group", "robotics"),
                "blank_ids": [item["id"]],
                "blanks": [{**blank, **_dependency_fields(blank)}],
                "condition": condition,
                "run": 0,
                "seed": 20261102,
                "max_tokens": 0,
                "stop": None,
                "standard": {(link["process"], item["id"]): item["answer"]},
                "question": question,
                "messages": history if condition == "session-gold" else None,
                "prompt": transcript(history) if condition == "session-gold" else "",
                "session": group["session"],
                "gold_reply": _gold_turn(item),
                "system": system,
            })
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": _gold_turn(item)})
    return turns


def jobs_for(spec: str, sample: dict, max_tokens: int, stop: list[str], seed: int) -> list[dict]:
    found = []
    for chain in sample["chains"]:
        targets = [index for index, sentence in enumerate(chain["sentences"]) if sentence["item"] is not None]
        for index in targets:
            item = chain["sentences"][index]["item"]
            blank = {
                "blank_id": item["id"],
                "label": item["label"],
                "format_code": item["format_code"],
                "answer": item["answer"],
                "options": list(item["options"]),
                "process": chain["process"],
            }
            found.append({
                "sample_id": sample["sample_id"],
                "tier": 0,
                "process": chain["process"],
                "group": sample.get("group", "robotics"),
                "blank_ids": [item["id"]],
                "blanks": [{**blank, **_dependency_fields(blank)}],
                "condition": "gold-trunc",
                "run": 0,
                "seed": seed,
                "max_tokens": max_tokens,
                "stop": stop,
                "standard": {(chain["process"], item["id"]): item["answer"]},
                "prompt": _prompt(spec, sample, chain, index),
            })
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=("robotics", "flow"), default="robotics")
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--part", choices=("forward", "inverse", "both"), default="both")
    parser.add_argument("--combined", action="store_true")
    parser.add_argument(
        "--condition",
        choices=("gold-trunc", "session-gold", "session-model"),
        default="gold-trunc",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.domain == "flow":
        review = args.review or FLOW
        out = args.out or (ROOT / "results" / "nothink_recon" / "flight_flow")
        spec, samples = load_flow(review)
    else:
        review = args.review or REVIEW
        out = args.out or (ROOT / "results" / "nothink_recon" / "robotics")
        spec, samples = load_samples(review)
        if args.part == "forward":
            samples = samples[:1]
        elif args.part == "inverse":
            samples = samples[1:]
        if args.combined:
            for sample in samples:
                sample["sample_id"] = "kinematics"
    args.review = review
    args.out = out
    load_env_file(ROOT / ".env")
    _key, _base, model = resolve_api_config()
    thinking = "disabled"
    max_tokens = _resolve_max_tokens(0, thinking)
    stop = ["\n\n"]
    workers = _resolve_workers(args.workers)
    jobs = []
    sessions = []
    if args.condition == "gold-trunc":
        for sample in samples:
            jobs.extend(jobs_for(spec, sample, max_tokens, stop, 20261102))
            blanks = sum(len(chain["ids"]) for chain in sample["chains"])
            print(
                f"{sample['sample_id']} 子过程 {len(sample['chains'])} 空 {blanks}",
                flush=True,
            )
    else:
        for group in _session_groups(samples):
            turns = _session_turns(spec, group, args.condition)
            for turn in turns:
                turn["max_tokens"] = max_tokens
                turn["stop"] = stop
            sessions.append(turns)
            jobs.extend(turns)
            print(
                f"会话 {group['session']} 题 {len(turns)}",
                flush=True,
            )
        second = sessions[0][1]["messages"] if args.condition == "session-gold" else None
        if second is not None:
            marker = sessions[0][0]["system"].split("【定义声明】", 1)[1].strip()[:40]
            users = [item["content"] for item in second if item["role"] == "user"]
            if any(marker in content for content in users):
                raise RuntimeError("第二问的用户消息里出现了定义声明。")
    print(
        f"条件：{args.condition} model={model} thinking={thinking} sentences={len(jobs)} max_tokens={max_tokens}",
        flush=True,
    )
    if args.dry:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    done_path = args.out / "records.jsonl"
    done = {}
    if done_path.is_file():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                done[_record_key(item)] = item
    pending = [job for job in jobs if _record_key(job) not in done]
    print(f"pending={len(pending)}", flush=True)
    if pending and args.condition == "session-model":
        client = make_client(timeout=600.0)
        lock = threading.Lock()
        finished = {"n": len(jobs) - len(pending)}
        with done_path.open("a", encoding="utf-8") as handle:
            def write(result: dict) -> None:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                finished["n"] += 1
                if finished["n"] == 1 or finished["n"] % 20 == 0 or finished["n"] == len(jobs):
                    print(f"scored {finished['n']}/{len(jobs)}", flush=True)

            def run_session(turns: list[dict]) -> None:
                messages = [{"role": "system", "content": turns[0]["system"]}]
                rerun = False
                for turn in turns:
                    question = turn["question"]
                    history = [*messages, {"role": "user", "content": question}]
                    saved = done.get(_record_key(turn))
                    if saved is not None and not rerun:
                        messages.append({"role": "user", "content": question})
                        messages.append({"role": "assistant", "content": saved.get("response") or ""})
                        continue
                    rerun = True
                    turn["messages"] = history
                    turn["prompt"] = transcript(history)
                    result = score_one(client, model, 0.0, thinking, turn)
                    reply = result.get("response") or ""
                    reasoning = result.get("reasoning") or ""
                    if reasoning and len(reasoning) > 40 and reasoning in question:
                        raise RuntimeError("思维链被写回了下一句的输入。会话只保留答案文本。")
                    messages.append({"role": "user", "content": question})
                    messages.append({"role": "assistant", "content": reply})
                    with lock:
                        write(result)
                        done[_record_key(result)] = result

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_session, turns) for turns in sessions]
                for future in as_completed(futures):
                    future.result()
    elif pending:
        client = make_client(timeout=600.0)
        lock = threading.Lock()
        finished = len(jobs) - len(pending)
        with done_path.open("a", encoding="utf-8") as handle:
            def write(result: dict) -> None:
                nonlocal finished
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                if finished == 1 or finished % 20 == 0 or finished == len(jobs):
                    print(f"scored {finished}/{len(jobs)}", flush=True)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(score_one, client, model, 0.0, thinking, job)
                    for job in pending
                ]
                for future in as_completed(futures):
                    result = future.result()
                    with lock:
                        write(result)
                        done[_record_key(result)] = result
    stored = [done[_record_key(job)] for job in jobs if _record_key(job) in done]
    preface = _preface(stored, model, thinking, 0.0, 1, 20261102, max_tokens, args.condition)
    if args.condition != "gold-trunc":
        note = "定义只写在会话的系统消息。之后逐题追加当前句子。session-gold 的前文是金标，session-model 的前文是模型自己的回答。"
    elif args.domain == "flow":
        note = "飞行力学空间坐标转换，统一默认。前四个子过程共用坐标定义。气动力和参考点各自注入，不携带前面子过程的正文。标准前缀，句号截断，思考关闭。"
    elif args.combined:
        note = "机器人学运动学。正运动学和逆运动学分开注入，分数合并为一条。子过程之间不共享正文。标准前缀，句号截断，思考关闭。"
    else:
        note = "机器人学运动学审阅稿。正样本和逆样本分开注入，子过程之间不共享正文。标准前缀，句号截断，思考关闭。"
    preface.append(note)
    report = summarize(stored, preface)
    (args.out / "report.md").write_text(report, encoding="utf-8")
    (args.out / "meta.json").write_text(json.dumps({
        "model": model,
        "thinking": thinking,
        "condition": args.condition,
        "review": str(args.review),
        "part": args.part,
        "sentences": len(jobs),
        "stored": len(stored),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report.split("## 按定义变更档")[0], flush=True)


if __name__ == "__main__":
    main()
