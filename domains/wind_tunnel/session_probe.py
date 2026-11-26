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
    PROBE_MARK,
    _question,
)

PROBE_LEAD = "This is a side probe. Answer only this question."


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
    raise KeyError(f"{chain['process']} has no probe target ID{blank_id}")


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
        "Do not fill blanks in the process, and do not continue the later sentences.",
        "This answer is only a score record. It is not written back to the main session.",
    ]
    if preamble:
        parts.extend(["", "[Text so far]", preamble])
    parts.extend([
        "",
        "[Probe]",
        _question(spec["name"]),
        "",
        f"Output one JSON object only. The key is {spec['target']}, and the value is the matrix string. Do not explain.",
    ])
    question = "\n".join(parts)
    marker = chain["declaration"].strip()[:40]
    if "[Declaration]" in question or (len(marker) >= 12 and marker in question):
        raise RuntimeError(f"{spec['probe_id']} probe repeats the declaration")
    if "re-derive" in question or "write it again" in question:
        raise RuntimeError(f"{spec['probe_id']} probe still asks for a re-derivation")
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
        raise RuntimeError("The main session did not clear the matrix answers: " + ", ".join(ids))
    for blank_id in ids:
        gold = _compact(gold_by_id.get(blank_id, ""))
        if gold and gold in _compact(text):
            raise RuntimeError(f"The main session still keeps the matrix answer of ID{blank_id}")
    messages[index]["content"] = text


def _reply_text(messages: list[dict], committed: list[dict], blank_id: str) -> str:
    for turn in committed:
        if str(blank_id) in {str(item) for item in turn["blank_ids"]}:
            return messages[turn["assistant_index"]]["content"]
    raise RuntimeError(f"The main session does not yet have an answer for ID{blank_id}")


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
            raise RuntimeError("The rederive branch did not hide the component-matrix reply")
        hidden[index]["content"] = text
        touched = True
    if not touched:
        raise RuntimeError("The rederive branch found no component-matrix reply to hide")
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
            raise RuntimeError(f"{spec['probe_id']} has not finished the component matrices to hide: " + ", ".join(missing))
        source = hide_replies(messages, committed, spec["hide"], gold_by_id)
    question = probe_question(chain, spec)
    branch = [{"role": item["role"], "content": item["content"]} for item in source]
    branch.append({"role": "user", "content": question})
    if _snapshot(messages) != before:
        raise RuntimeError(f"{spec['probe_id']} rewrote the main session while building the branch")
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
        raise RuntimeError(f"{spec['probe_id']} branch does not end with the probe question")
    if len(branch) != len(messages) + 1:
        raise RuntimeError(f"{spec['probe_id']} branch did not add exactly one question turn")
    transcript = "\n".join(item["content"] for item in branch)
    if spec["named"] not in transcript:
        raise RuntimeError(f"{spec['name']} probe has not seen the matrix name")
    if "[Declaration]" in branch[-1]["content"]:
        raise RuntimeError(f"{spec['probe_id']} injects the declaration again in a later question")
    own = _compact(gold_by_id[spec["target"]])
    if spec["kind"] == "shallow":
        if own and own in _compact(transcript):
            raise RuntimeError(f"{spec['name']} shallow probe saw its own original answer")
    elif spec["kind"] == "deep_rederive":
        hide = {str(item) for item in spec["hide"]}
        for turn in committed:
            ids = [str(item) for item in turn["blank_ids"] if str(item) in hide]
            if not ids:
                continue
            text = branch[turn["assistant_index"]]["content"]
            if PLACEHOLDER not in text:
                raise RuntimeError(f"{spec['name']} rederive branch did not hide the component matrices")
            for blank_id in ids:
                gold = _compact(gold_by_id.get(blank_id, ""))
                if gold and gold in _compact(text):
                    raise RuntimeError(f"{spec['name']} rederive probe still sees a component-matrix answer")
    else:
        raise RuntimeError(f"Unknown probe kind {spec['kind']}")
    if chain["process"] != spec["probe_id"].split(":", 1)[0]:
        raise RuntimeError("The probe process does not match the session process")


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
            raise RuntimeError(f"{chain['process']} main question contains a probe stem")
        for gold in cleared:
            if gold and gold in _compact(question):
                raise RuntimeError(f"{chain['process']} later main question contains a cleared matrix answer")
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
            branch.append({"role": "assistant", "content": f"{PROBE_MARK}{spec['probe_id']}"})
            if _snapshot(messages) != before:
                raise RuntimeError(f"{spec['probe_id']} wrote back into the main session")
            if PROBE_LEAD in "\n".join(item["content"] for item in messages):
                raise RuntimeError(f"{spec['probe_id']} stem entered the main session")
            if index + 1 < len(sentences):
                nxt = _session_question(chain, sentences[index + 1], sentence)
                if PROBE_LEAD in nxt or PROBE_MARK in nxt:
                    raise RuntimeError(f"{spec['probe_id']} later main question contains a probe")
            count += 1
        previous = sentence
    expected = _expected_count(chain["process"])
    if expected and count != expected:
        raise RuntimeError(f"{chain['process']} forked {count} probes, expected {expected}")
    return count


def _expected_count(process: str) -> int:
    for stage in _stages():
        if stage["process"] == process:
            return len(stage["matrices"]) * 2
    return 0
