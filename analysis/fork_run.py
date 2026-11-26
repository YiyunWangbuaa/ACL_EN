# -*- coding: utf-8 -*-
"""Session fork for the ID196 kernel: sign, force, coordinate, denominator.

The control is one session-model chat of moment reference point, stopped at ID196.
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
PROCESS = "moment reference point"
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
    raise RuntimeError(f"{chain['process']} has no ID{stop_id}")


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
        raise RuntimeError("no kernel slot needs to be restored to gold")
    if messages[-1]["role"] != "assistant":
        raise RuntimeError("a fork should end on the assistant reply before ID196")
    fork = [{"role": item["role"], "content": item["content"]} for item in messages]
    for index, gold_text in replacements:
        if fork[index]["role"] != "assistant":
            raise RuntimeError("the kernel slot is not an assistant reply")
        if fork[index]["content"] == gold_text:
            raise RuntimeError("this kernel slot is already gold")
        fork[index]["content"] = gold_text
    fork.append({"role": "user", "content": question})
    return fork


def prepare(sample: str) -> tuple[dict, list[dict], dict[str, str]]:
    rows = load_samples(DATA, {sample})
    chain = next(item for item in contexts_for(rows[0]) if item["process"] == PROCESS)
    sentences = _sentences_through(chain, TARGET_ID)
    target = sentences[-1]
    if target["blank_ids"] != [TARGET_ID]:
        raise RuntimeError(f"ID{TARGET_ID} is not its own sentence")
    gold = {}
    for sentence in sentences:
        ids = [str(item) for item in sentence["blank_ids"]]
        if len(ids) == 1 and ids[0] in KERNEL:
            if ids[0] in gold:
                raise RuntimeError(f"ID{ids[0]} appears twice")
            gold[ids[0]] = gold_reply(sentence["blanks"])
    if set(gold) != set(KERNEL):
        raise RuntimeError(f"kernel slots are incomplete: {sorted(set(KERNEL) - set(gold))}")
    for sentence in sentences:
        if forks_after(chain, sentence["blank_ids"]):
            raise RuntimeError(f"{PROCESS} would insert a probe, so the main session would no longer be only manuscript turns")
        if answers_to_clear(chain, sentence["blank_ids"]):
            raise RuntimeError(f"ID{sentence['blank_ids']} assistant reply would be cleared by clear_turn")
    return chain, sentences, gold


def check_shape(sample: str) -> None:
    chain, sentences, gold = prepare(sample)
    if sample == "body-rule" and gold != BODY_GOLD:
        raise RuntimeError(f"body-rule kernel gold should be {BODY_GOLD}, got {gold}")
    order = [str(item) for sentence in sentences for item in sentence["blank_ids"]]
    for earlier, later in (("194", "201"), ("213", "195"), ("195", "196")):
        if order.index(earlier) > order.index(later):
            raise RuntimeError(f"ID{earlier} is not before ID{later}")
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
            raise RuntimeError(f"ID{ids} assistant reply was rewritten by the main session")
        if len(ids) == 1 and ids[0] in KERNEL:
            indexes[ids[0]] = len(messages) - 1
        previous = sentence
    if set(indexes) != set(KERNEL):
        raise RuntimeError("the four kernel slots did not each enter the session")
    question = _session_question(chain, sentences[-1], previous)
    for blank_id in KERNEL:
        if f"{{{blank_id}:" in question:
            raise RuntimeError(f"the ID196 user message includes ID{blank_id}")
    if "positive" in question or "negative" in question:
        raise RuntimeError("the ID196 user message includes a sign word")
    blanks = {
        "190": {"normalized": False},
        "202": {"normalized": True},
        "208": {"normalized": True},
        "213": {"normalized": False},
    }
    replacements = kernel_replacements(indexes, blanks, gold)
    if [gold_text for _index, gold_text in replacements] != [gold["190"], gold["213"]]:
        raise RuntimeError(f"only ID190 and ID213 should be replaced, got {replacements}")
    fork = build_fork(messages, replacements, question)
    if fork[indexes["190"]]["content"] != gold["190"] or fork[indexes["213"]]["content"] != gold["213"]:
        raise RuntimeError("the fork did not replace a wrong kernel slot with gold")
    if fork[indexes["202"]]["content"] != "{202:CZ}" or fork[indexes["208"]]["content"] != gold["208"]:
        raise RuntimeError("the fork rewrote a kernel slot that was already correct")
    if messages[indexes["190"]]["content"] != "{190:B}":
        raise RuntimeError("the fork wrote back into the control session")
    if fork[-1] != {"role": "user", "content": question}:
        raise RuntimeError("the fork did not stop on the ID196 user message")
    if any(item["role"] == "assistant" and item["content"].startswith("{196:") for item in fork):
        raise RuntimeError("the fork should not contain the old ID196 answer")
    for marker in ("{191:Z}", "{201:Z}"):
        index = next(i for i, item in enumerate(messages) if item["role"] == "assistant" and item["content"] == marker)
        if fork[index]["content"] != marker:
            raise RuntimeError(f"the fork changed an assistant reply outside the kernel {marker}")
    if kernel_replacements(indexes, {blank_id: {"normalized": True} for blank_id in KERNEL}, gold):
        raise RuntimeError("a replacement was required even though all four kernel slots were correct")
    try:
        build_fork(messages, [], question)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("an empty replacement still built a fork")
    print(
        f"shape check passed: {sample} {PROCESS}, kernel gold {gold}, ID196 user message {len(question)} characters.",
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
                raise RuntimeError(f"run {run + 1} ID{ids} question text disagrees with the saved copy")
            if not _failed(record):
                _replay(messages, question, record["committed"])
                if len(ids) == 1 and ids[0] in KERNEL:
                    indexes[ids[0]] = len(messages) - 1
                previous = sentence
                continue
        _invalidate_after(run_dir, index)
        print(f"run {run + 1} asking ID{','.join(ids)}", flush=True)
        result = _ask(
            client, model, temperature, thinking, chain, sentence, messages, question,
            run, seed, max_tokens, standard, SESSION_MODEL,
        )
        reply = result.get("response") or ""
        committed = _commit(messages, question, reply, sentence, chain) if not (result.get("error") or result.get("unanswered")) else reply
        record = _turn_record(result, question, committed)
        _save(path, record)
        if _failed(record):
            print(f"run {run + 1} ID{ids[0]} call failed: {record['error']}", flush=True)
            return {"run": run, "status": "failed", "where": ids}
        if committed != reply:
            raise RuntimeError(f"ID{ids} assistant reply was rewritten by the main session")
        if len(ids) == 1 and ids[0] in KERNEL:
            indexes[ids[0]] = len(messages) - 1
        previous = sentence
    if set(indexes) != set(KERNEL):
        raise RuntimeError("the four kernel slots did not each enter the session")
    question = _session_question(chain, target, previous)
    kernel = _kernel_snapshot(run_dir)
    wrong_ids = [blank_id for blank_id in KERNEL if not kernel[blank_id].get("normalized")]
    control_path = run_dir / "control.json"
    if control_path.is_file() and not _failed(_load(control_path)):
        control = _load(control_path)
    else:
        print(f"run {run + 1} control question ID196 (wrong slots {','.join(wrong_ids) or 'none'})", flush=True)
        result = _ask(
            client, model, temperature, thinking, chain, target, messages, question,
            run, seed, max_tokens, standard, SESSION_MODEL,
        )
        control = _turn_record(result, question, result.get("response") or "")
        control["blank"] = _picked(result, TARGET_ID) if result.get("blanks") else {}
        control["wrong_ids"] = wrong_ids
        _save(control_path, control)
    if _failed(control):
        print(f"run {run + 1} ID196 control call failed: {control.get('error')}", flush=True)
        return {"run": run, "status": "failed", "where": [TARGET_ID], "kernel": kernel, "wrong_ids": wrong_ids}
    control_blank = control.get("blank") or _picked_saved(control)
    if not wrong_ids:
        status = "residue" if _wrong(control_blank) else "clean"
        print(f"run {run + 1} all four kernel slots are gold, status {status}", flush=True)
        return {"run": run, "status": status, "kernel": kernel, "wrong_ids": [], "control": control_blank}
    if control_only:
        print(f"run {run + 1} control recorded, wrong slots {','.join(wrong_ids)}, not forking yet", flush=True)
        return {"run": run, "status": "control", "kernel": kernel, "wrong_ids": wrong_ids, "control": control_blank}
    replacements = kernel_replacements(indexes, kernel, gold)
    injected = {blank_id: gold[blank_id] for blank_id in wrong_ids}
    fork_path = run_dir / "fork.json"
    if fork_path.is_file() and not _failed(_load(fork_path)):
        fork = _load(fork_path)
    else:
        print(f"run {run + 1} fork: {injected}, re-ask ID196", flush=True)
        forked = build_fork(messages, replacements, question)
        for index, gold_text in replacements:
            if forked[index]["content"] != gold_text or messages[index]["content"] == gold_text:
                raise RuntimeError("the fork replacement does not match the control/intervention split")
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
        print(f"run {run + 1} ID196 fork call failed: {fork.get('error')}", flush=True)
        return {"run": run, "status": "failed", "where": [TARGET_ID], "kernel": kernel, "wrong_ids": wrong_ids, "control": control_blank}
    fork_blank = fork.get("blank") or _picked_saved(fork)
    print(
        f"run {run + 1} kept: control {control_blank.get('prediction')!r}, fork {fork_blank.get('prediction')!r}",
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
        "# Session fork: the four-layer kernel of ID196",
        "",
        f"Sample {sample}, subprocess {PROCESS}. Model {model}, thinking {thinking}, effort {effort}, temperature 0.",
        "The control is session-model: the definition is only in the system message, ID181 through ID196 are answered sentence by sentence, and assistant replies keep the model text.",
        "The fork copies that session, deletes the ID196 assistant reply, replaces the wrong slots among ID190, ID202, ID208, and ID213 with gold JSON, and asks ID196 again.",
        "A kernel slot already equal to gold after normalization keeps the model text. Other slots are not changed and not re-asked.",
        "A run whose four kernel slots are all gold is not forked: ID196 still wrong is a formula-layer residue, and ID196 also correct is clean.",
        "The chain of thought is not written into the next turn. This step tests the visible session, not hidden state inside one generation.",
        "",
        f"Planned {runs} runs, records completed {len(rows)}: forked {n}, waiting to fork {len(waiting)}, formula-layer residue {len(residue)}, clean {len(clean)}, call failed {len(failed)}.",
        "",
        "| slot | control wrong |",
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
        f"ID196 still wrong when all four slots are correct: {residual}. The denominator is the runs whose four kernel slots all match gold after normalization.",
        "",
        "The table above uses every control that finished the four kernel slots. The two error rates below use only runs with at least one wrong kernel slot that have already re-asked ID196.",
        "",
        "| quantity | value |",
        "| --- | --- |",
        f"| entered fork | {n}/{len(rows) or runs} |",
        f"| p_control | {rate(control_wrong)} |",
        f"| p_repair | {rate(fork_wrong)} |",
        f"| effect p_control - p_repair | {'—' if p_prop is None else f'{(p_prop - p_fix):.0%}'} |",
        f"| Fisher exact test | {'—' if fisher is None else f'{fisher:.4f}'} |",
        f"| two-proportion z | {'—' if z_test is None else f'z={z_test[0]:.2f}, p={z_test[1]:.4f}'} |",
        f"| paired exact test | {'—' if paired is None else f'{paired:.4f}'} |",
        "",
        "Fisher and z treat the control and the fork of the same run as two independent samples. The paired exact test counts only runs that disagree.",
        "",
        "| control wrong, fork right | control right, fork wrong | both wrong | both right |",
        "| ---: | ---: | ---: | ---: |",
        f"| {rescued} | {harmed} | {both_wrong} | {both_right} |",
        "",
        f"Of the {both_wrong} runs that are wrong on both sides, {same_text} have the same fork text as the control. The other {both_wrong - same_text} changed the expression and still do not equal gold after normalization.",
        f"Formula-layer residue {len(residue)} runs: all four kernel slots are gold, ID196 is still wrong, and those runs are not in the two error rates above.",
        "",
        "| run | status | wrong kernel slots | control ID196 | fork ID196 |",
        "| ---: | --- | --- | --- | --- |",
    ])
    labels = {
        "kept": "forked",
        "control": "waiting to fork",
        "residue": "formula-layer residue",
        "clean": "clean",
        "failed": "call failed",
    }
    for row in rows:
        control = (row.get("control") or {}).get("prediction") or ""
        fork = (row.get("fork") or {}).get("prediction") or ""
        if row["status"] == "failed":
            control = control or "call failed"
            fork = fork or "not done"
        elif row["status"] != "kept":
            fork = "—"
        wrong = ", ".join(f"ID{blank_id}" for blank_id in row.get("wrong_ids") or []) or "—"
        lines.append(f"| {row['run'] + 1} | {labels.get(row['status'], row['status'])} | {wrong} | {control} | {fork} |")
    lines.append("")
    if waiting and n == 0:
        lines.append("The control is recorded. The fork has not been run.")
    elif n == 0:
        lines.append("No run has a wrong kernel slot, so the repair rate has no denominator.")
    elif p_fix is not None and p_prop is not None and p_fix + 0.5 <= p_prop and fisher is not None and fisher < 0.05:
        lines.append("After the full kernel is restored to gold, the ID196 error rate drops, Fisher p<0.05.")
    elif p_fix is not None and p_prop is not None and abs(p_prop - p_fix) <= 0.1:
        lines.append("After the full kernel is restored to gold, the ID196 error rate is essentially unchanged.")
    else:
        lines.append("The two error rates differ, and this batch does not yet pin the difference as a drop or as no change.")
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
    parser.add_argument("--report", type=Path, default=ROOT / "doc" / "session_fork_ID196_kernel.md")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--control-only", action="store_true")
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    out = args.out or (ROOT / "results" / "fork_id196_kernel" / args.sample)
    check_shape(args.sample)
    if args.check:
        if fisher_exact(10, 0, 0, 10) >= 0.001:
            raise RuntimeError("Fisher did not treat a fork that fixes every case as a small probability")
        if fisher_exact(10, 0, 10, 0) != 1:
            raise RuntimeError("Fisher did not return 1 when both sides are entirely wrong")
        if mcnemar_exact(10, 0) >= 0.01:
            raise RuntimeError("the paired test did not treat 10 fork-only fixes as a small probability")
        from domains.wind_tunnel.injector_catalog import e3_rows
        variants = {
            row.question[0]: row.variant
            for row in e3_rows("body-rule")
            if row.question
        }
        found = {key: variants.get(key) for key in ("190", "202", "213")}
        if found != {"190": "sign", "202": "choice", "213": "choice"}:
            raise RuntimeError(f"moment-probe row marks are wrong: {found}")
        print("Test functions passed.", flush=True)
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
                raise SystemExit(f"saved results were run under different conditions: {comparable}")
        _save(meta_path, meta)
        print(
            f"sample {args.sample} model {args.model} thinking {args.thinking} effort {args.effort} "
            f"kernel {gold} runs {args.runs} {'control only' if args.control_only else 'fork after control'}",
            flush=True,
        )
        try:
            for run in range(args.runs):
                run_trial(
                    chain, sentences, gold, client, args.model, args.temperature, args.thinking,
                    run, args.seed + run, max_tokens, standard, out, args.control_only,
                )
        except RunStop as exc:
            print(f"call interrupted: {exc}", flush=True)
    rows = collect(out, args.runs)
    text = render(args.sample, args.model, args.thinking, args.effort, args.runs, rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
