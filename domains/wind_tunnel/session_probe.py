# -*- coding: utf-8 -*-
"""Fork shallow and rederive probes, then clear matrix answers in the main chat.

The kept probes are the shallow ask and the deep rederive ask. The original-flow
deep probe is not used. After a component matrix or a total matrix is scored,
its answer in the main chat is replaced by a placeholder. Later manuscript
turns and later probes see that placeholder, not the matrix.
"""
from __future__ import annotations

import re

from domains.wind_tunnel.rt_isolation import (
    COMPONENT_IDS,
    DEEP_ANCHOR,
    MATRICES,
    PLACEHOLDER,
    PROCESS,
    RT_ID,
    STAGES,
    _compact,
    _next_slotfree,
    _question,
)

PROBE_LEAD = "这是一次旁路探针。只回答这一问。"


def _stages() -> list[dict]:
    stages = [
        {
            "process": PROCESS,
            "deep_anchor": DEEP_ANCHOR,
            "hide": COMPONENT_IDS,
            "totals": (RT_ID,),
            "matrices": MATRICES,
        }
    ]
    for stage in STAGES:
        total = stage.get("total_id") or ""
        stages.append({
            "process": stage["process"],
            "deep_anchor": stage["deep_anchor"],
            "hide": tuple(item["blank"] for item in stage["matrices"]),
            "totals": (total,) if total else (),
            "matrices": stage["matrices"],
        })
    return stages


def _gold_map(chain: dict) -> dict[str, str]:
    found = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            found[str(blank["blank_id"])] = str(blank.get("answer") or "")
    return found


def _target_blank(chain: dict, blank_id: str) -> dict:
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            if str(blank["blank_id"]) == str(blank_id):
                return blank
    raise KeyError(f"{chain['process']} 没有探针目标 ID{blank_id}")


def forks_after(chain: dict, blank_ids: list[str]) -> list[dict]:
    """Probes whose anchor was just committed on the main trajectory."""
    anchors = {str(item) for item in blank_ids}
    found = []
    for stage in _stages():
        if chain.get("process") != stage["process"]:
            continue
        for item in stage["matrices"]:
            if item["shallow_anchor"] in anchors:
                found.append(_spec(chain, stage, item, "shallow", item["shallow_anchor"]))
        if stage["deep_anchor"] in anchors:
            for item in stage["matrices"]:
                found.append(_spec(chain, stage, item, "deep_rederive", stage["deep_anchor"]))
    return found


def _spec(chain: dict, stage: dict, item: dict, kind: str, anchor: str) -> dict:
    name = item["name"]
    return {
        "probe_id": f"{stage['process']}:{name}:{kind}",
        "record_id": f"probe:{name}:{kind}",
        "kind": kind,
        "name": name,
        "anchor": str(anchor),
        "target": str(item["blank"]),
        "named": item["named"],
        "hide": stage["hide"] if kind == "deep_rederive" else (),
        "blank": _target_blank(chain, item["blank"]),
    }


def probe_question(chain: dict, spec: dict) -> str:
    """The branch user turn. It does not repeat the definition."""
    preamble = _next_slotfree(chain, spec["anchor"]).strip()
    parts = [
        PROBE_LEAD,
        "不要填写过程中的空，不要继续后面的句子。",
        "这个答案只作评分记录，不会写回主会话。",
    ]
    if preamble:
        parts.extend(["", "【题目前文】", preamble])
    parts.extend([
        "",
        "【探针】",
        _question(spec["name"]),
        "",
        f"只输出一个 JSON 对象，键是 {spec['target']}，值是矩阵字符串。不要解释。",
    ])
    question = "\n".join(parts)
    marker = chain["declaration"].strip()[:40]
    if "【定义声明】" in question or (len(marker) >= 12 and marker in question):
        raise RuntimeError(f"{spec['probe_id']} 的探针题干重复了定义声明")
    if "重新推导" in question or "重新写出" in question:
        raise RuntimeError(f"{spec['probe_id']} 的探针仍在要求重新推导")
    return question


def _snapshot(messages: list[dict]) -> list[tuple[str, str]]:
    return [(item["role"], item["content"]) for item in messages]


def _replace_fields(content: str, blank_ids: list[str], placeholder: str) -> tuple[str, bool]:
    updated = content
    for blank_id in blank_ids:
        pattern = re.compile(
            rf'((?:"{re.escape(blank_id)}"|{re.escape(blank_id)})\s*:\s*)'
            rf'(?:"(?:\\.|[^"\\])*"|[^,}}\s]+)'
        )
        updated, count = pattern.subn(lambda match: f'{match.group(1)}"{placeholder}"', updated, count=1)
        if not count:
            return content, False
    return updated, True


def answers_to_clear(chain: dict, blank_ids: list[str]) -> list[str]:
    """Original-flow shallow matrix fills and original-flow totals on this turn."""
    present = {str(item) for item in blank_ids}
    found = []
    for stage in _stages():
        if chain.get("process") != stage["process"]:
            continue
        for item in stage["matrices"]:
            if str(item["blank"]) in present:
                found.append(str(item["blank"]))
        for total in stage["totals"]:
            if str(total) in present:
                found.append(str(total))
    return found


def clear_turn(
    messages: list[dict],
    turn: dict,
    clear_ids: list[str],
    gold_by_id: dict[str, str],
) -> None:
    """Replace matrix answers in one main-session reply. The scored record stays intact."""
    if not clear_ids:
        return
    index = turn["assistant_index"]
    ids = [str(item) for item in clear_ids]
    alone = set(map(str, turn["blank_ids"])) <= set(ids)
    text = messages[index]["content"]
    rewritten, matched = _replace_fields(text, ids, PLACEHOLDER)
    if matched:
        text = rewritten
    elif alone:
        text = "{" + ", ".join(f'{blank_id}:"{PLACEHOLDER}"' for blank_id in ids) + "}"
    for blank_id in ids:
        gold = gold_by_id.get(blank_id, "")
        if gold and gold in text:
            text = text.replace(gold, PLACEHOLDER)
    if PLACEHOLDER not in text:
        raise RuntimeError("主会话没有清掉矩阵答案：" + "、".join(ids))
    for blank_id in ids:
        gold = _compact(gold_by_id.get(blank_id, ""))
        if gold and gold in _compact(text):
            raise RuntimeError(f"主会话仍保留 ID{blank_id} 的矩阵答案")
    messages[index]["content"] = text


def _reply_text(messages: list[dict], committed: list[dict], blank_id: str) -> str:
    for turn in committed:
        if str(blank_id) in {str(item) for item in turn["blank_ids"]}:
            return messages[turn["assistant_index"]]["content"]
    raise RuntimeError(f"主会话里还没有 ID{blank_id} 的回答")


def hide_replies(
    messages: list[dict],
    committed: list[dict],
    hide_ids: tuple[str, ...] | list[str],
    gold_by_id: dict[str, str],
    placeholder: str = PLACEHOLDER,
) -> list[dict]:
    """Copy the chat and replace component-matrix replies. The original list stays."""
    hidden = [{"role": item["role"], "content": item["content"]} for item in messages]
    hide = {str(item) for item in hide_ids}
    touched = False
    for turn in committed:
        ids = [str(item) for item in turn["blank_ids"] if str(item) in hide]
        if not ids:
            continue
        index = turn["assistant_index"]
        alone = set(map(str, turn["blank_ids"])) <= hide
        text = hidden[index]["content"]
        rewritten, matched = _replace_fields(text, ids, placeholder)
        if matched:
            text = rewritten
        elif alone:
            text = "{" + ", ".join(f'{blank_id}:"{placeholder}"' for blank_id in ids) + "}"
        for blank_id in ids:
            gold = gold_by_id.get(blank_id, "")
            if gold and gold in text:
                text = text.replace(gold, placeholder)
        if PLACEHOLDER not in text:
            raise RuntimeError("重推导分支没能隐去分矩阵回答")
        hidden[index]["content"] = text
        touched = True
    if not touched:
        raise RuntimeError("重推导分支没有找到要隐去的分矩阵回答")
    return hidden


def make_branch(
    chain: dict,
    messages: list[dict],
    committed: list[dict],
    spec: dict,
    gold_by_id: dict[str, str] | None = None,
) -> list[dict]:
    """Return the probe chat. ``messages`` is not modified."""
    before = _snapshot(messages)
    gold_by_id = gold_by_id if gold_by_id is not None else _gold_map(chain)
    source = messages
    if spec["kind"] == "deep_rederive":
        present = {str(blank_id) for turn in committed for blank_id in turn["blank_ids"]}
        missing = [blank_id for blank_id in spec["hide"] if str(blank_id) not in present]
        if missing:
            raise RuntimeError(f"{spec['probe_id']} 还没走完要隐去的分矩阵：" + "、".join(missing))
        source = hide_replies(messages, committed, spec["hide"], gold_by_id)
    question = probe_question(chain, spec)
    branch = [{"role": item["role"], "content": item["content"]} for item in source]
    branch.append({"role": "user", "content": question})
    if _snapshot(messages) != before:
        raise RuntimeError(f"{spec['probe_id']} 在构造分支时改写了主会话")
    _audit(chain, messages, branch, committed, spec, gold_by_id)
    return branch


def _audit(
    chain: dict,
    messages: list[dict],
    branch: list[dict],
    committed: list[dict],
    spec: dict,
    gold_by_id: dict[str, str],
) -> None:
    if branch[-1]["role"] != "user" or PROBE_LEAD not in branch[-1]["content"]:
        raise RuntimeError(f"{spec['probe_id']} 的分支末尾不是探针问题")
    if len(branch) != len(messages) + 1:
        raise RuntimeError(f"{spec['probe_id']} 的分支没有只多出一轮提问")
    transcript = "\n".join(item["content"] for item in branch)
    if spec["named"] not in transcript:
        raise RuntimeError(f"{spec['name']} 的探针还没看见矩阵名称")
    if "【定义声明】" in branch[-1]["content"]:
        raise RuntimeError(f"{spec['probe_id']} 在后续提问里重复注入了定义")
    own = _compact(gold_by_id[spec["target"]])
    if spec["kind"] == "shallow":
        if own and own in _compact(transcript):
            raise RuntimeError(f"{spec['name']} 的浅层探针看见了自己的原答案")
    elif spec["kind"] == "deep_rederive":
        hide = {str(item) for item in spec["hide"]}
        for turn in committed:
            ids = [str(item) for item in turn["blank_ids"] if str(item) in hide]
            if not ids:
                continue
            text = branch[turn["assistant_index"]]["content"]
            if PLACEHOLDER not in text:
                raise RuntimeError(f"{spec['name']} 的重推导分支没有隐去分矩阵")
            for blank_id in ids:
                gold = _compact(gold_by_id.get(blank_id, ""))
                if gold and gold in _compact(text):
                    raise RuntimeError(f"{spec['name']} 的重推导探针仍然看见分矩阵答案")
    else:
        raise RuntimeError(f"未知探针类型 {spec['kind']}")
    if chain["process"] != spec["probe_id"].split(":", 1)[0]:
        raise RuntimeError("探针过程与会话过程不一致")


def check_session(chain: dict) -> int:
    """Walk one gold session, fork every probe, and prove the main chat continues."""
    from cloze_eval import _session_question, _session_system, gold_reply

    messages = [{"role": "system", "content": _session_system(chain)}]
    committed: list[dict] = []
    previous = None
    count = 0
    gold_by_id = _gold_map(chain)
    cleared: list[str] = []
    sentences = chain["sentences"]
    for index, sentence in enumerate(sentences):
        question = _session_question(chain, sentence, previous)
        if PROBE_LEAD in question:
            raise RuntimeError(f"{chain['process']} 的主问题带了探针题干")
        for gold in cleared:
            if gold and gold in _compact(question):
                raise RuntimeError(f"{chain['process']} 的后续主问题又带上了已清除的矩阵答案")
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": gold_reply(sentence["blanks"])})
        committed.append({
            "blank_ids": [str(item) for item in sentence["blank_ids"]],
            "assistant_index": len(messages) - 1,
        })
        clear_ids = answers_to_clear(chain, sentence["blank_ids"])
        clear_turn(messages, committed[-1], clear_ids, gold_by_id)
        for blank_id in clear_ids:
            gold = _compact(gold_by_id.get(blank_id, ""))
            if gold:
                cleared.append(gold)
        before = _snapshot(messages)
        for spec in forks_after(chain, sentence["blank_ids"]):
            branch = make_branch(chain, messages, committed, spec, gold_by_id)
            branch.append({"role": "assistant", "content": f"《探针记录》{spec['probe_id']}"})
            if _snapshot(messages) != before:
                raise RuntimeError(f"{spec['probe_id']} 写回了主会话")
            if PROBE_LEAD in "\n".join(item["content"] for item in messages):
                raise RuntimeError(f"{spec['probe_id']} 的题干进入了主会话")
            if index + 1 < len(sentences):
                nxt = _session_question(chain, sentences[index + 1], sentence)
                if PROBE_LEAD in nxt or "《探针记录》" in nxt:
                    raise RuntimeError(f"{spec['probe_id']} 之后的主问题带了探针")
            count += 1
        previous = sentence
    expected = _expected_count(chain["process"])
    if expected and count != expected:
        raise RuntimeError(f"{chain['process']} 叉出 {count} 个探针，应当是 {expected} 个")
    return count


def _expected_count(process: str) -> int:
    for stage in _stages():
        if stage["process"] == process:
            return len(stage["matrices"]) * 2
    return 0
