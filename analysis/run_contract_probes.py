# -*- coding: utf-8 -*-
"""Load the five contract probes and ask them without writing the answers back.

The real matrices and formulas stay the answers already stored in the A/B runs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_score import normalize  # noqa: E402
from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from cloze_eval import THINKING_MAX_TOKENS, _prefix_phrase, _repair_phrase, contexts_for, load_env_file, load_samples  # noqa: E402
from domains.wind_tunnel.contract_probes import (  # noqa: E402
    CONDITIONS,
    CONSEQUENCE,
    PROBES,
    build_prompt,
    classify,
    contract_key,
    execution_class,
    filled_prefix,
    parse_probe,
    prompt_id,
    recognition_text,
    same_contract,
)
from semi_auto_eval import call_model, make_client  # noqa: E402

DATA = EVAL_ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = EVAL_ROOT / "results" / "contract_probes"
STANDARD_ID = "standard-default-unmarked"
EFFORT = "disabled"
RECORD_PATHS = {
    "roll-unmarked": {
        "model": EVAL_ROOT / "results" / "pair_roll" / "self-trunc" / "records.jsonl",
        "gold_run": EVAL_ROOT / "results" / "pair_roll" / "gold-trunc" / "records.jsonl",
    },
    "body-unmarked": {
        "model": EVAL_ROOT / "results" / "session_body" / "model_low" / "records.jsonl",
        "gold_run": EVAL_ROOT / "results" / "session_body" / "gold_low" / "records.jsonl",
    },
}


def _unwrap_blank(blank: dict) -> dict:
    """Use the single JSON value when a session reply was keyed by sentence number."""
    prediction = str(blank.get("prediction") or "").strip()
    if not prediction.startswith("{"):
        return blank
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError:
        return blank
    if not isinstance(parsed, dict) or len(parsed) != 1:
        return blank
    value = next(iter(parsed.values()))
    if not isinstance(value, str):
        return blank
    from cloze_score import judge

    repaired = dict(blank)
    repaired["prediction"] = value
    repaired.update(judge(
        value,
        str(blank.get("answer") or ""),
        str(blank.get("format_code") or ""),
        blank.get("options"),
    ))
    return repaired


def _records(sample_id: str, role: str) -> dict[str, dict]:
    path = RECORD_PATHS[sample_id][role]
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("branch") == "probe" or str(row.get("sample_id") or "") != sample_id:
            continue
        for blank in row.get("blanks") or []:
            found[str(blank["blank_id"])] = _unwrap_blank(blank)
    return found


def _chain(row: dict, process: str) -> dict:
    found = next(item for item in contexts_for(row) if item["process"] == process)
    found["prior"] = row.get("prior") or found.get("prior")
    return found


def _blanks(chain: dict) -> dict[str, dict]:
    found = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            found[str(blank["blank_id"])] = blank
    return found


def _ancestors(blanks: dict[str, dict], target: str) -> set[str]:
    seen = set()
    stack = [str(target)]
    while stack:
        current = stack.pop()
        for parent in blanks.get(current, {}).get("depends_on") or []:
            parent = str(parent)
            if parent in seen:
                continue
            seen.add(parent)
            stack.append(parent)
    return seen


def _gold_map(blanks: dict[str, dict]) -> dict[str, str]:
    return {blank_id: _repair_phrase(blank) for blank_id, blank in blanks.items()}


def _model_map(records: dict[str, dict]) -> dict[str, str]:
    return {blank_id: _prefix_phrase(blank) for blank_id, blank in records.items()}


def _memory(probe: dict, condition: str, model: dict[str, str], gold: dict[str, str], blanks: dict[str, dict], records: dict[str, dict]) -> tuple[dict[str, str], list[str]]:
    if condition == "deep_gold":
        return dict(gold), []
    base = dict(model)
    if condition in {"shallow", "deep_b", "replay"}:
        return base, []
    if condition == "repair_parent":
        changed = [item for item in probe["repair_parent"] if item in gold]
        for item in changed:
            base[item] = gold[item]
        return base, changed
    if condition == "repair_other":
        changed = list(probe["repair_other"])
        if not changed:
            anchor = probe["deep_after"]
            present = set()
            # The ids available to this probe are those whose gold phrase exists and whose
            # number is at most the deep anchor. Ancestors stay on the model answer.
            ancestors = _ancestors(blanks, probe["target"])
            wrong = [
                blank_id for blank_id, blank in records.items()
                if not blank.get("normalized") and blank_id not in ancestors and blank_id in gold
                and int(blank_id) <= int(anchor)
            ]
            changed = sorted(wrong, key=int)[-1:] if wrong else []
        for item in changed:
            base[item] = gold[item]
        return base, list(changed)
    if condition == "matched":
        ancestors = _ancestors(blanks, probe["target"])
        changed = []
        for blank_id, blank in records.items():
            if blank_id in ancestors or blank_id == probe["target"]:
                continue
            if int(blank_id) > int(probe["deep_after"]) or blank_id not in gold:
                continue
            if not blank.get("normalized"):
                base[blank_id] = gold[blank_id]
                changed.append(blank_id)
        return base, sorted(changed, key=int)
    raise KeyError(condition)


def _recognition_keys(probe: dict, blanks: dict[str, dict], standard_blanks: dict[str, dict]) -> tuple[dict, dict]:
    contract_text = recognition_text(probe, lambda blank_id: _repair_phrase(blanks[blank_id]))
    standard_text = recognition_text(probe, lambda blank_id: _repair_phrase(standard_blanks[blank_id]))
    return contract_key(probe, contract_text), contract_key(probe, standard_text)


def _phrases(probe: dict, gold_blanks: dict[str, dict]) -> dict[str, str]:
    return {}


def _jobs(bundles: dict[str, dict]) -> list[dict]:
    jobs = []
    for probe in PROBES:
        bundle = bundles[probe["sample_id"]]
        phrases = _phrases(probe, bundle["blanks"])
        for condition in CONDITIONS:
            memory, changed = _memory(
                probe, condition, bundle["model"], bundle["gold"], bundle["blanks"], bundle["b_records"],
            )
            prompt = build_prompt(bundle["chains"][probe["process"]], probe, condition, memory, phrases)
            jobs.append({
                "probe_id": probe["probe_id"],
                "condition": condition,
                "group": probe["group"],
                "target": probe["target"],
                "changed": changed,
                "prompt": prompt,
                "prompt_id": prompt_id(prompt),
            })
    replay = {item["probe_id"]: item for item in jobs if item["condition"] == "replay"}
    deep = {item["probe_id"]: item for item in jobs if item["condition"] == "deep_b"}
    for probe_id, item in replay.items():
        if item["prompt"] != deep[probe_id]["prompt"]:
            raise RuntimeError(f"{probe_id} 的重放提示和深层 B 提示不一致。")
    return jobs


def _audit(bundles: dict[str, dict], standard: dict[str, dict]) -> list[str]:
    lines = [
        "# 契约探针加载",
        "",
        "探针只问局部符号或负号位置，不要求写出目标矩阵或公式。答案只记录，不写回原流程。目标执行仍使用已经跑完的模型前缀和金标前缀记录。",
        "",
        "期望答案取自当前稿面。符号探针只比较 sin 项或整体符号，不把另一套定义的角差或手性写进题干。",
        "",
    ]
    problems = []
    for probe in PROBES:
        bundle = bundles[probe["sample_id"]]
        target = bundle["blanks"][probe["target"]]
        contract_text = _repair_phrase(target)
        standard_blank = standard[probe["process"]][probe["target"]]
        standard_text = _repair_phrase(standard_blank)
        try:
            contract, standard_key = _recognition_keys(probe, bundle["blanks"], standard[probe["process"]])
        except Exception as exc:  # noqa: BLE001 - the audit should report the manuscript mismatch
            problems.append(f"{probe['probe_id']} 无法从稿面读取契约：{exc}")
            continue
        b_pred = bundle["b_records"][probe["target"]].get("prediction") or ""
        a_pred = bundle["a_records"][probe["target"]].get("prediction") or ""
        parents = [str(item) for item in target.get("depends_on") or []]
        parent_state = [
            f"ID{item}{'正确' if bundle['b_records'].get(item, {}).get('normalized') else '错误'}"
            for item in parents
        ]
        lines.append(f"## {probe['probe_id']} → ID{probe['target']}")
        lines.append("")
        lines.append(f"- 稿面：{probe['sample_id']}，{probe['process']}")
        lines.append(f"- 浅层锚点：ID{probe['shallow_after']} 所在句结束。深层锚点：ID{probe['deep_after']} 所在句结束。")
        lines.append(f"- 契约：{contract}")
        lines.append(f"- 标准定义：{standard_key}")
        lines.append(f"- 模型前缀下的目标输出：{b_pred}")
        lines.append(f"- gold 前缀下的目标输出：{a_pred}")
        lines.append(f"- 模型前缀执行归类：{execution_class(b_pred, contract_text, standard_text)}")
        lines.append(f"- gold 前缀执行归类：{execution_class(a_pred, contract_text, standard_text)}")
        lines.append(f"- 直接父节点：{'、'.join(parent_state) if parent_state else '无'}")
        shallow = filled_prefix(bundle["chain"], probe["shallow_after"], bundle["model"])
        deep = filled_prefix(bundle["chain"], probe["deep_after"], bundle["model"])
        if probe["target"] in shallow or f"@@{probe['target']}@@" in shallow:
            problems.append(f"{probe['probe_id']} 浅层前缀含有目标空。")
        if f"@@{probe['target']}@@" in deep:
            problems.append(f"{probe['probe_id']} 深层前缀仍留着目标槽。")
        hidden = probe.get("hidden_stem") or ""
        if hidden and (hidden in shallow or hidden in deep):
            problems.append(f"{probe['probe_id']} 前缀露出了 {hidden}")
        lines.append(f"- 浅层题干：{probe['shallow_question']}")
        lines.append(f"- 深层题干：{probe['deep_question']}")
        lines.append("")
    consequence = bundles[CONSEQUENCE["sample_id"]]
    target = consequence["blanks"][CONSEQUENCE["target"]]
    lines.append("## ID58 组合后果")
    lines.append("")
    lines.append("ID58 不设探针。它只记录左手系约定有没有传到姿态分量。")
    lines.append(f"- 契约：{_repair_phrase(target)}")
    lines.append(f"- 模型前缀输出：{consequence['b_records']['58'].get('prediction')}")
    lines.append("")
    return lines, problems


def _load() -> tuple[dict[str, dict], dict[str, dict]]:
    wanted = {item["sample_id"] for item in PROBES} | {STANDARD_ID}
    rows = {row["id"]: row for row in load_samples(DATA, wanted)}
    bundles = {}
    for probe in PROBES:
        sample_id = probe["sample_id"]
        if sample_id in bundles:
            continue
        row = rows[sample_id]
        chain = _chain(row, probe["process"])
        # One sample has several processes. Keep every chain the probes use.
        bundles[sample_id] = {
            "row": row,
            "chains": {},
            "blanks": {},
            "b_records": _records(sample_id, "model"),
            "a_records": _records(sample_id, "gold_run"),
        }
    for probe in PROBES:
        bundle = bundles[probe["sample_id"]]
        if probe["process"] not in bundle["chains"]:
            chain = _chain(bundle["row"], probe["process"])
            bundle["chains"][probe["process"]] = chain
            bundle["blanks"].update(_blanks(chain))
        bundle["chain"] = bundle["chains"][probe["process"]]
        bundle["model"] = _model_map(bundle["b_records"])
        bundle["gold"] = _gold_map(bundle["blanks"])
    # The loop above leaves chain pointing at the last process. Jobs look up by probe.
    prepared = {}
    for probe in PROBES:
        bundle = bundles[probe["sample_id"]]
        prepared.setdefault(probe["sample_id"], {
            "b_records": bundle["b_records"],
            "a_records": bundle["a_records"],
            "chains": bundle["chains"],
            "blanks": bundle["blanks"],
            "model": _model_map(bundle["b_records"]),
            "gold": _gold_map(bundle["blanks"]),
        })
    # Flatten per probe access: each job uses the chain of its process.
    usable = {}
    for sample_id, bundle in prepared.items():
        usable[sample_id] = bundle
    # Attach the right chain inside _jobs via probe process. Store chains.
    for sample_id, bundle in usable.items():
        bundle["chain"] = next(iter(bundle["chains"].values()))
    standard_row = rows[STANDARD_ID]
    standard = {}
    for probe in PROBES:
        if probe["process"] in standard:
            continue
        standard[probe["process"]] = _blanks(_chain(standard_row, probe["process"]))
    # Rebind chain per probe inside a wrapper so _jobs sees the right process.
    wrapped = {}
    for probe in PROBES:
        bundle = usable[probe["sample_id"]]
        wrapped[probe["sample_id"]] = {
            **bundle,
            "chain": bundle["chains"][probe["process"]],
            "blanks": bundle["blanks"],
        }
    return wrapped, standard


def _write_design(bundles, standard) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text, problems = _audit_each(bundles, standard)
    (OUT / "design.md").write_text(text, encoding="utf-8")
    if problems:
        raise SystemExit("探针加载未通过：\n" + "\n".join(problems))


def _audit_each(bundles: dict, standard: dict) -> tuple[str, list[str]]:
    lines = [
        "# 契约探针加载",
        "",
        "探针只问局部符号或负号位置，不要求写出目标矩阵或公式。答案只记录，不写回原流程。目标执行仍使用已经跑完的模型前缀和金标前缀记录。",
        "",
        "期望答案取自当前稿面。符号探针只比较 sin 项或整体符号，不把另一套定义的角差或手性写进题干。",
        "",
    ]
    problems = []
    for probe in PROBES:
        base = bundles[probe["sample_id"]]
        chain = base["chains"][probe["process"]]
        blanks = base["blanks"]
        target = blanks[probe["target"]]
        contract_text = _repair_phrase(target)
        standard_text = _repair_phrase(standard[probe["process"]][probe["target"]])
        try:
            contract, standard_key = _recognition_keys(probe, blanks, standard[probe["process"]])
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{probe['probe_id']} 无法从稿面读取契约：{exc}")
            continue
        b_pred = base["b_records"][probe["target"]].get("prediction") or ""
        a_pred = base["a_records"][probe["target"]].get("prediction") or ""
        parents = [str(item) for item in target.get("depends_on") or []]
        parent_state = [
            f"ID{item}{'正确' if base['b_records'].get(item, {}).get('normalized') else '错误'}"
            for item in parents
        ]
        shallow = filled_prefix(chain, probe["shallow_after"], base["model"])
        deep = filled_prefix(chain, probe["deep_after"], base["model"])
        hidden = probe.get("hidden_stem") or ""
        if hidden and (hidden in shallow or hidden in deep):
            problems.append(f"{probe['probe_id']} 前缀露出了 {hidden}")
        if f"@@{probe['target']}@@" in shallow or f"@@{probe['target']}@@" in deep:
            problems.append(f"{probe['probe_id']} 前缀仍留着目标槽")
        phrases = _phrases(probe, blanks)
        shallow_prompt = build_prompt(chain, probe, "shallow", base["model"], phrases)
        deep_prompt = build_prompt(chain, probe, "deep_b", base["model"], phrases)
        if "【探针】" not in shallow_prompt or shallow_prompt.index("【探针】") < shallow_prompt.index("【过程】"):
            problems.append(f"{probe['probe_id']} 探针没有放在过程之后")
        if probe["shallow_question"].split("，")[0] in deep.split("【探针】")[0]:
            pass
        lines += [
            f"## {probe['probe_id']} → ID{probe['target']}",
            "",
            f"- 稿面：{probe['sample_id']}，{probe['process']}",
            f"- 浅层截到 ID{probe['shallow_after']}，深层截到 ID{probe['deep_after']}。",
            f"- 契约：`{json.dumps(contract, ensure_ascii=False)}`",
            f"- 标准定义：`{json.dumps(standard_key, ensure_ascii=False)}`",
            f"- 模型前缀下的目标输出：`{b_pred}`",
            f"- gold 前缀下的目标输出：`{a_pred}`",
            f"- 模型前缀执行归类：{execution_class(b_pred, contract_text, standard_text)}",
            f"- gold 前缀执行归类：{execution_class(a_pred, contract_text, standard_text)}",
            f"- 直接父节点：{'、'.join(parent_state) if parent_state else '无'}",
            f"- 浅层题干：{probe['shallow_question'].format(**phrases) if phrases else probe['shallow_question']}",
            f"- 深层题干：{probe['deep_question']}",
            "",
        ]
        _ = deep_prompt
    target = bundles[CONSEQUENCE["sample_id"]]["blanks"][CONSEQUENCE["target"]]
    lines += [
        f"## ID{CONSEQUENCE['target']} 组合后果",
        "",
        f"ID{CONSEQUENCE['target']} 不设探针。它只记录左手系约定有没有传到姿态分量。",
        f"- 契约：`{_repair_phrase(target)}`",
        f"- 模型前缀输出：`{bundles[CONSEQUENCE['sample_id']]['b_records'][CONSEQUENCE['target']].get('prediction')}`",
        "",
    ]
    return "\n".join(lines), problems


def _done() -> dict[tuple[str, str], dict]:
    path = OUT / "calls.jsonl"
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("probe_id") and row.get("condition") and "response" in row:
            found[(row["probe_id"], row["condition"])] = row
    return found


def _ask(jobs: list[dict]) -> None:
    load_env_file(EVAL_ROOT / ".env")
    client = make_client(timeout=1800.0)
    thinking = "enabled" if EFFORT != "disabled" else "disabled"
    stop = None if thinking == "enabled" else ["\n\n"]
    max_tokens = THINKING_MAX_TOKENS if thinking == "enabled" else 8192
    done = _done()
    pending = [job for job in jobs if (job["probe_id"], job["condition"]) not in done]
    path = OUT / "calls.jsonl"
    for index, job in enumerate(pending, start=1):
        print(f"probe {index}/{len(pending)} {job['probe_id']} {job['condition']}", flush=True)
        captured = {}
        response = ""
        reasoning = ""
        last_error = ""
        for attempt in range(4):
            try:
                response, reasoning = call_model(
                    client,
                    "deepseek-flash",
                    job["prompt"],
                    0.0,
                    thinking,
                    captured=captured,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=20261102,
                    reasoning_effort=None if EFFORT == "disabled" else EFFORT,
                )
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - retry a transient API error
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(8, 2 ** attempt))
        if last_error:
            raise SystemExit(f"{job['probe_id']} {job['condition']} 调用失败：{last_error}")
        probe = next(item for item in PROBES if item["probe_id"] == job["probe_id"])
        row = {
            "probe_id": job["probe_id"],
            "condition": job["condition"],
            "target": job["target"],
            "changed": job["changed"],
            "prompt_id": job["prompt_id"],
            "response": response,
            "reasoning": reasoning,
            "thinking": thinking,
            "reasoning_effort": EFFORT,
            "finish_reason": captured.get("finish_reason"),
            "parsed": parse_probe(probe, response),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[(job["probe_id"], job["condition"])] = row


def _report(bundles: dict, standard: dict) -> None:
    calls = _done()
    lines = [
        "# 契约探针结果",
        "",
        "探针答案没有写回原流程。目标节点的执行结果来自已经跑完的模型前缀和金标前缀。",
        f"本次探针调用：deepseek-flash，思考 {('enabled / ' + EFFORT) if EFFORT != 'disabled' else 'disabled'}。",
        "",
        "| 探针 | 浅层 | 深层model | 深层gold | 重放一致 | model执行 | gold前缀执行 | 父节点 | 解释 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for probe in PROBES:
        base = bundles[probe["sample_id"]]
        blanks = base["blanks"]
        contract_text = _repair_phrase(blanks[probe["target"]])
        standard_text = _repair_phrase(standard[probe["process"]][probe["target"]])
        contract, standard_key = _recognition_keys(probe, blanks, standard[probe["process"]])

        def hit(condition: str) -> tuple[bool, bool, dict]:
            row = calls[(probe["probe_id"], condition)]
            parsed = parse_probe(probe, row["response"])
            return (
                same_contract(probe, parsed, contract),
                same_contract(probe, parsed, standard_key),
                parsed,
            )

        shallow_hit, shallow_std, shallow_parsed = hit("shallow")
        deep_hit, deep_std, deep_parsed = hit("deep_b")
        gold_hit, gold_std, gold_parsed = hit("deep_gold")
        replay_hit, _replay_std, replay_parsed = hit("replay")
        b_pred = base["b_records"][probe["target"]].get("prediction") or ""
        a_pred = base["a_records"][probe["target"]].get("prediction") or ""
        parents = [str(item) for item in blanks[probe["target"]].get("depends_on") or []]
        parents_ok = all(base["b_records"].get(item, {}).get("normalized") for item in parents) if parents else True
        facts = {
            "shallow_hit": shallow_hit,
            "deep_b_hit": deep_hit,
            "deep_gold_hit": gold_hit,
            "deep_b_standard": deep_std and not deep_hit,
            "deep_b_parsed": bool(deep_parsed.get("parsed")),
            "target_b": execution_class(b_pred, contract_text, standard_text),
            "target_gold_run": execution_class(a_pred, contract_text, standard_text),
            "parents_correct": parents_ok,
        }
        labels = "；".join(classify(facts))

        def side(contract_hit: bool, standard_hit: bool, parsed: dict) -> str:
            if contract_hit:
                return "契约"
            if standard_hit:
                return "标准"
            if not parsed.get("parsed"):
                return "未解析"
            return "其他"

        if probe["kind"] == "expression":
            same_replay = normalize(deep_parsed.get("text") or "") == normalize(replay_parsed.get("text") or "")
        else:
            same_replay = side(deep_hit, deep_std, deep_parsed) == side(replay_hit, _replay_std, replay_parsed)
        lines.append(
            f"| {probe['probe_id']} | {side(shallow_hit, shallow_std, shallow_parsed)} | "
            f"{side(deep_hit, deep_std, deep_parsed)} | "
            f"{side(gold_hit, gold_std, gold_parsed)} | {'一致' if same_replay else '波动'} | "
            f"{facts['target_b']} | {facts['target_gold_run']} | "
            f"{'正确' if parents_ok else '有错'} | {labels} |"
        )
    lines += [
        "",
        "## 前缀对照",
        "",
        "matched 把非祖先错误换成正确答案。repair_parent 只改直接父节点。repair_other 只改指定的非父节点。",
        "",
        "| 探针 | 长度匹配 | 父节点修复 | 非父节点修复 |",
        "| --- | --- | --- | --- |",
    ]
    for probe in PROBES:
        base = bundles[probe["sample_id"]]
        blanks = base["blanks"]
        contract_text = _repair_phrase(blanks[probe["target"]])
        standard_text = _repair_phrase(standard[probe["process"]][probe["target"]])
        contract, standard_key = _recognition_keys(probe, blanks, standard[probe["process"]])

        def side_of(condition: str, contract=contract, standard_key=standard_key, probe=probe) -> str:
            row = calls[(probe["probe_id"], condition)]
            parsed = parse_probe(probe, row["response"])
            if same_contract(probe, parsed, contract):
                return "契约"
            if same_contract(probe, parsed, standard_key):
                return "标准"
            if not parsed.get("parsed"):
                return "未解析"
            return "其他"

        lines.append(
            f"| {probe['probe_id']} | {side_of('matched')} | {side_of('repair_parent')} | {side_of('repair_other')} |"
        )
    lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _configure() -> None:
    global OUT, EFFORT
    if "--effort" in sys.argv:
        EFFORT = sys.argv[sys.argv.index("--effort") + 1]
        if EFFORT not in {"low", "medium", "high", "disabled"}:
            raise SystemExit("--effort 只能是 low、medium、high 或 disabled。")
    if EFFORT != "disabled":
        OUT = EVAL_ROOT / "results" / f"contract_probes_{EFFORT}"


def _compare(bundles: dict, standard: dict) -> None:
    old_path = EVAL_ROOT / "results" / "contract_probes" / "calls.jsonl"
    if not old_path.is_file() or OUT.resolve() == old_path.parent.resolve():
        return
    previous = {}
    for line in old_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            previous[(row["probe_id"], row["condition"])] = row
    current = _done()
    lines = [
        "",
        "## 与关闭推理的对照",
        "",
        "两列都是探针判断。前缀相同，差别只在探针调用是否以 low 思考。",
        "",
        "| 探针 | 条件 | 关闭推理 | low |",
        "| --- | --- | --- | --- |",
    ]
    for probe in PROBES:
        blanks = bundles[probe["sample_id"]]["blanks"]
        contract, standard_key = _recognition_keys(
            probe,
            blanks,
            standard[probe["process"]],
        )

        def side(row: dict, contract=contract, standard_key=standard_key, probe=probe) -> str:
            parsed = parse_probe(probe, row.get("response") or "")
            if same_contract(probe, parsed, contract):
                return "契约"
            if same_contract(probe, parsed, standard_key):
                return "标准"
            if not parsed.get("parsed"):
                return "未解析"
            return "其他"

        for condition in ("shallow", "deep_b", "deep_gold"):
            lines.append(
                f"| {probe['probe_id']} | {condition} | {side(previous[(probe['probe_id'], condition)])} | "
                f"{side(current[(probe['probe_id'], condition)])} |"
            )
    report = OUT / "report.md"
    report.write_text(report.read_text(encoding="utf-8") + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    global PROBES
    _configure()
    if "--sample" in sys.argv:
        sample_id = sys.argv[sys.argv.index("--sample") + 1]
        PROBES = tuple(item for item in PROBES if item["sample_id"] == sample_id)
        if not PROBES:
            raise SystemExit(f"没有样本 {sample_id} 的探针。")
    print(f"探针思考：{EFFORT}，输出：{OUT}，样本：{sorted({item['sample_id'] for item in PROBES})}", flush=True)
    design_only = "--design-only" in sys.argv
    bundles, standard = _load()
    _write_design(bundles, standard)
    print(OUT / "design.md", flush=True)
    if design_only:
        return
    jobs = []
    for probe in PROBES:
        base = bundles[probe["sample_id"]]
        chain = base["chains"][probe["process"]]
        phrases = _phrases(probe, base["blanks"])
        for condition in CONDITIONS:
            memory, changed = _memory(
                probe, condition, base["model"], base["gold"], base["blanks"], base["b_records"],
            )
            prompt = build_prompt(chain, probe, condition, memory, phrases)
            jobs.append({
                "probe_id": probe["probe_id"],
                "condition": condition,
                "group": probe["group"],
                "target": probe["target"],
                "changed": changed,
                "prompt": prompt,
                "prompt_id": prompt_id(prompt),
            })
    deep = {item["probe_id"]: item["prompt"] for item in jobs if item["condition"] == "deep_b"}
    for item in jobs:
        if item["condition"] == "replay" and item["prompt"] != deep[item["probe_id"]]:
            raise SystemExit(f"{item['probe_id']} 重放提示和深层 B 不一致")
    _ask(jobs)
    _report(bundles, standard)
    _compare(bundles, standard)
    print(OUT / "report.md", flush=True)


if __name__ == "__main__":
    main()
