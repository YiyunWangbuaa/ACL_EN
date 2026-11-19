# -*- coding: utf-8 -*-
"""De-leak ID85, recompute repaired targets, and split moment signs from levers.

Probe answers are not written back. Target recomputes are separate calls.
Contract values come from the manuscript generator, not from a hand-written case.
"""
from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.run_contract_probes import _ancestors, _gold_map, _load, _model_map  # noqa: E402
from cloze_eval import _score_blanks, load_env_file, prompt_for  # noqa: E402
from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from domains.wind_tunnel.contract_probes import (  # noqa: E402
    _cell_sign,
    _groups,
    execution_class,
    filled_prefix,
)
from semi_auto_eval import call_model, make_client  # noqa: E402

OUT = EVAL_ROOT / "results" / "nothink_pair" / "probes_followup"
NAMED = ("R_(γ_M-γ_tp)", "R_(γ_M−γ_tp)")


def _through_ids(chain: dict, anchor: str) -> list[str]:
    groups = _groups(chain)
    index = next(i for i, group in enumerate(groups) if str(anchor) in group)
    return [blank_id for group in groups[: index + 1] for blank_id in group]


def _sentence(chain: dict, target: str) -> dict:
    for sentence in chain["sentences"]:
        if str(target) in [str(item) for item in sentence["blank_ids"]]:
            return sentence
    raise KeyError(target)


def _memory(base_model: dict, gold: dict, mode: str, ids: list[str]) -> dict[str, str]:
    if mode == "gold":
        return dict(gold)
    memory = dict(base_model)
    for item in ids:
        memory[item] = gold[item]
    return memory


def _nonancestor(records: dict, blanks: dict, gold: dict, target: str, anchor: str, exclude: set[str]) -> list[str]:
    banned = _ancestors(blanks, target) | exclude | {target}
    wrong = [
        blank_id for blank_id, blank in records.items()
        if blank_id not in banned and blank_id in gold and int(blank_id) <= int(anchor)
        and not blank.get("normalized")
    ]
    return sorted(wrong, key=int)[-1:]


def _deleak(chain: dict, target: str) -> dict:
    copied = copy.deepcopy(chain)
    text = copied["process_text"]
    stem = next((item for item in NAMED if item in text), "")
    if not stem:
        raise RuntimeError("题面里没有 R_(γ_M-γ_tp)，无法做去泄露替换。")
    old = f"旋转矩阵{stem}为@@{target}@@。"
    new = f"旋转矩阵R_θ为@@{target}@@，其中θ是前面已经确定的滚转角差。"
    if old not in text:
        raise RuntimeError(f"去泄露没有找到原句：{old}")
    copied["process_text"] = text.replace(old, new, 1)
    if any(item in copied["process_text"].split(old)[-1][:0] for item in NAMED):
        pass
    if stem in new:
        raise RuntimeError("去泄露替换仍把标准角组合写进目标句。")
    return copied


def _side_prompt(chain: dict, anchor: str, memory: dict[str, str], question: str, output: str = '{"probe":"答案"}') -> str:
    prefix = filled_prefix(chain, anchor, memory)
    if question in prefix:
        raise RuntimeError("探针题干写进了流程前缀。")
    return "\n".join([
        "这是一次旁路问题。只回答下面这一问。",
        "不要填写过程中的空，不要继续后面的句子。",
        "这个答案只作记录，不会写回原流程。",
        f"只输出一个 JSON 对象 {output}。",
        "",
        "【定义声明】",
        chain["declaration"],
        "",
        "【物理常识】",
        chain.get("prior") or "",
        "",
        "【子过程】",
        chain["process"],
        "",
        "【过程】",
        prefix,
        "",
        "【问题】",
        question,
        "",
        "【输出】",
        output,
    ])


def _structured_prompt(chain: dict, anchor: str, memory: dict[str, str], name: str, force: str) -> str:
    question = (
        f"只判断 {name} 的两个部分，不要把它们乘开。"
        f"整体符号 S 只能写正或负，负号必须放在整个式子最外侧。"
        f"力臂 ARM 只能写 Xr-Xg 或 Xg-Xr，力臂内部不能再带负号。"
        f"力系数固定为 {force}。"
        f'{name} = [S] {force} · [ARM] / L_RT。'
        '只输出 {"S":"正或负","ARM":"Xr-Xg或Xg-Xr"}。'
    )
    return _side_prompt(chain, anchor, memory, question, '{"S":"正或负","ARM":"Xr-Xg或Xg-Xr"}')


def _probe_body(response: str) -> str:
    match = re.search(r'"probe"\s*:\s*"(.*?)"', response, flags=re.S)
    return (match.group(1) if match else response).strip()


def _direction_label(text: str) -> str:
    if any(token in text for token in ("相反", "逆", "反向")):
        return "相反"
    if any(token in text for token in ("相同", "正向")) or text.strip() in {"正", "顺"}:
        return "相同"
    return ""


def _pole(text: str) -> str:
    if "左" in text:
        return "左侧"
    if "右" in text:
        return "右侧"
    return ""


def _sign_word(text: str) -> str:
    raw = text.strip()
    if raw in {"负", "负号", "-", "−"} or raw.startswith("负"):
        return "负"
    if raw in {"正", "正号", "+", "＋"} or raw.startswith("正"):
        return "正"
    return ""


def _matrix_signs(text: str, target: str) -> str:
    prepared = (text or "").replace("θ", "γ_M")
    cells = ((1, 3), (3, 1)) if target == "34" else ((2, 3), (3, 2))
    contract = ("正", "负") if target == "34" else ("负", "正")
    standard = ("负", "正") if target == "34" else ("正", "负")
    try:
        got = tuple(_cell_sign(prepared, row, col) for row, col in cells)
    except Exception:
        return ""
    if got == contract:
        return "契约符号"
    if got == standard:
        return "标准符号"
    return f"其他({got[0]}/{got[1]})"


def _formula_parts(text: str) -> dict:
    raw = (text or "").replace(" ", "").replace("−", "-").replace("－", "-")
    outer = "负" if raw.startswith("-") else "正"
    body = raw[1:] if raw.startswith("-") else raw
    force = "CN" if "CN" in body else "CZ" if "CZ" in body else "CA" if "CA" in body else ""
    if "Xg-Xr" in body or "Xg−Xr" in text:
        arm = "Xg-Xr"
    elif "Xr-Xg" in body or "Xr−Xg" in text:
        arm = "Xr-Xg"
    else:
        arm = ""
    buried = arm == "Xg-Xr" or ("-" in body[body.find("("):body.find(")") + 1] if "(" in body and ")" in body else False)
    return {"outer": outer, "force": force, "arm": arm, "buried_sign": arm == "Xg-Xr", "text": text}


def _structured_parts(response: str) -> dict:
    source = response.replace("−", "-").replace("－", "-")
    sign = re.search(r'"S"\s*:\s*"?(正|负)', source)
    arm = re.search(r'"ARM"\s*:\s*"?(Xr-Xg|Xg-Xr|Xr−Xg|Xg−Xr)', source)
    arm_text = ""
    if arm:
        arm_text = "Xr-Xg" if "Xr" in arm.group(1) and arm.group(1).find("Xr") < arm.group(1).find("Xg") else "Xg-Xr"
    return {"outer": sign.group(1) if sign else "", "arm": arm_text, "force": "", "buried_sign": arm_text == "Xg-Xr", "text": response}


def _jobs(bundles: dict) -> list[dict]:
    specs = [
        {
            "target": "85", "sample": "roll-unmarked", "group": "roll", "process": "天平系到体轴系",
            "anchor": "84",
            "repairs": [("raw_b", []), ("repair_id84", ["84"]), ("repair_graph_parent", ["82", "83"])],
            "exclude": {"84", "82", "83"},
            "deleak": True,
        },
        {
            "target": "34", "sample": "body-unmarked", "group": "body", "process": "风轴系到体轴系",
            "anchor": "33",
            "repairs": [("raw_b", []), ("repair_direct", ["32", "33"])],
            "exclude": {"32", "33"},
            "deleak": False,
        },
        {
            "target": "43", "sample": "body-unmarked", "group": "body", "process": "风轴系到体轴系",
            "anchor": "42",
            "repairs": [("raw_b", []), ("repair_direct", ["41", "42"])],
            "exclude": {"41", "42"},
            "deleak": False,
        },
        {
            "target": "192", "sample": "body-unmarked", "group": "body", "process": "力矩参考点",
            "anchor": "191",
            "repairs": [("raw_b", []), ("repair_186", ["186"]), ("repair_178_186", ["178", "186"])],
            "exclude": {"178", "186"},
            "deleak": False,
            "formula": True,
            "force": "CN",
            "sign_anchor": "186",
        },
        {
            "target": "194", "sample": "body-unmarked", "group": "body", "process": "力矩参考点",
            "anchor": "193",
            "repairs": [("raw_b", []), ("repair_188", ["188"]), ("repair_178_188", ["178", "188"])],
            "exclude": {"178", "188"},
            "deleak": False,
            "formula": True,
            "force": "CZ",
            "sign_anchor": "188",
        },
    ]
    jobs = []
    roll = bundles["roll-unmarked"]["chains"]["天平系到体轴系"]
    body_wind = bundles["body-unmarked"]["chains"]["风轴系到体轴系"]
    for sample_id, chain, anchor, question, name in (
        ("roll-unmarked", roll, "83", "以当前坐标轴正方向为基准，该旋转与右手定则相同还是相反？", "gamma_dir_before_84"),
        ("roll-unmarked", roll, "84", "在当前工程约定下，绕当前轴旋转θ时，矩阵右上角的 sin(θ) 项应为正还是负？θ是前面已经确定的滚转角差。", "gamma_sign_theta"),
        ("roll-unmarked", roll, "84", "在当前工程约定下，R_(γ_M-γ_tp) 的右上角 sin(γ_M-γ_tp) 项应为正还是负？", "gamma_sign_named"),
        ("body-unmarked", body_wind, "8", "当前体轴系的 ZTZ 指向机体左侧还是右侧？", "hand_after_8"),
        ("body-unmarked", body_wind, "34", "当前坐标系仍是左手系吗？ZTZ 指向机体左侧还是右侧？", "hand_after_34"),
    ):
        bundle = bundles[sample_id]
        for mode, ids in (("raw_b", []), ("gold", "gold")):
            memory = _memory(bundle["model"], bundle["gold"], "gold" if mode == "gold" else "raw", [])
            jobs.append({
                "name": name, "kind": "probe", "mode": mode, "target": anchor,
                "prompt": _side_prompt(chain, anchor, memory, question),
                "changed": [],
            })
    for spec in specs:
        bundle = bundles[spec["sample"]]
        chain = bundle["chains"][spec["process"]]
        other = _nonancestor(bundle["b_records"], bundle["blanks"], bundle["gold"], spec["target"], spec["anchor"], spec["exclude"])
        repairs = list(spec["repairs"]) + [("repair_nonancestor", other), ("gold", "gold")]
        for mode, ids in repairs:
            memory = _memory(bundle["model"], bundle["gold"], "gold" if ids == "gold" else "raw", [] if ids == "gold" else ids)
            sentence = _sentence(chain, spec["target"])
            prompt = prompt_for(chain, [str(item) for item in sentence["blank_ids"]], "self-trunc", memory)
            jobs.append({
                "name": f"target_{spec['target']}", "kind": "formula" if spec.get("formula") else "matrix",
                "mode": mode, "target": spec["target"], "prompt": prompt,
                "changed": [] if ids == "gold" else list(ids), "form": "original",
            })
            if spec.get("formula"):
                jobs.append({
                    "name": f"parts_{spec['target']}", "kind": "structured", "mode": mode,
                    "target": spec["target"], "force": spec["force"],
                    "prompt": _structured_prompt(chain, spec["anchor"], memory, "MCN_CZ" if spec["target"] == "192" else "MCZ_CN", spec["force"]),
                    "changed": [] if ids == "gold" else list(ids),
                })
            if spec.get("deleak"):
                leaked = prompt_for(_deleak(chain, spec["target"]), [str(item) for item in sentence["blank_ids"]], "self-trunc", memory)
                if any(item in leaked.split("【过程】")[-1] for item in NAMED):
                    raise RuntimeError("去泄露后的目标段仍含有 R_(γ_M-γ_tp)。")
                jobs.append({
                    "name": "target_85_theta", "kind": "matrix", "mode": mode, "target": "85",
                    "prompt": leaked, "changed": [] if ids == "gold" else list(ids), "form": "theta",
                })
        if spec.get("sign_anchor"):
            memory = _memory(bundle["model"], bundle["gold"], "raw", [])
            name = "MCN_CZ" if spec["target"] == "192" else "MCZ_CN"
            jobs.append({
                "name": f"sign_{spec['target']}", "kind": "probe", "mode": "raw_b", "target": spec["target"],
                "prompt": _side_prompt(chain, spec["sign_anchor"], memory, f"在当前左手体轴约定下，{name} 的整体符号是正还是负？"),
                "changed": [],
            })
    return jobs


def _done() -> dict[tuple[str, str], dict]:
    path = OUT / "calls.jsonl"
    found = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                found[(row["name"], row["mode"])] = row
    return found


def _ask(jobs: list[dict]) -> None:
    load_env_file(EVAL_ROOT / ".env")
    client = make_client(timeout=180.0)
    done = _done()
    pending = [job for job in jobs if (job["name"], job["mode"]) not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "calls.jsonl"
    for index, job in enumerate(pending, start=1):
        print(f"follow {index}/{len(pending)} {job['name']} {job['mode']}", flush=True)
        last_error = ""
        response = ""
        for attempt in range(4):
            try:
                response, _reasoning = call_model(
                    client, "deepseek-flash", job["prompt"], 0.0, "disabled",
                    max_tokens=8192, stop=["\n\n"], seed=20261102,
                )
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(8, 2 ** attempt))
        if last_error:
            raise SystemExit(f"{job['name']} {job['mode']} 调用失败：{last_error}")
        row = {
            "name": job["name"], "kind": job["kind"], "mode": job["mode"],
            "target": job["target"], "changed": job.get("changed") or [],
            "form": job.get("form", ""), "response": response,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prediction(bundle: dict, response: str, target: str) -> str:
    sentence = _sentence(bundle["chains"][_process(bundle, target)], target)
    scored = _score_blanks(response, sentence["blanks"], None, "")
    found = next(item for item in scored if str(item["blank_id"]) == target)
    return found.get("prediction") or ""


def _process(bundle: dict, target: str) -> str:
    for process, chain in bundle["chains"].items():
        try:
            _sentence(chain, target)
        except KeyError:
            continue
        return process
    raise KeyError(target)


def _report(bundles: dict, standard: dict) -> None:
    calls = _done()
    lines = [
        "# 去泄露、父节点修复与目标重算",
        "",
        "思考关闭。探针不写回原流程。契约和标准都取生成器金标。",
        "ID194 的生成器契约整体符号是负，力臂是 Xr-Xg。",
        "",
        "## 目标执行",
        "",
        "| 目标 | 条件 | 替换节点 | 原题面执行 | 符号归类 |",
        "| --- | --- | --- | --- | --- |",
    ]

    def bundle_of(target: str) -> dict:
        sample = "roll-unmarked" if target == "85" else "body-unmarked"
        return bundles[sample]

    def gold_pair(target: str) -> tuple[str, str]:
        bundle = bundle_of(target)
        process = _process(bundle, target)
        contract = bundle["gold"][target]
        standard_text = standard[process][target]
        from cloze_eval import _repair_phrase
        return contract, _repair_phrase(standard_text)

    for name in ("target_85", "target_85_theta", "target_34", "target_43", "target_192", "target_194"):
        rows = [row for row in calls.values() if row["name"] == name]
        for row in rows:
            target = row["target"]
            contract, standard_text = gold_pair(target)
            if name.endswith("theta"):
                pred = _prediction(bundle_of(target), row["response"], target)
                klass = _matrix_signs(pred or row["response"], target)
                whole = execution_class(pred, contract, standard_text) if pred else "未解析"
                shown = klass or whole
            elif name.startswith("target_19"):
                pred = _prediction(bundle_of(target), row["response"], target)
                parts = _formula_parts(pred or row["response"])
                shown = f"{execution_class(pred, contract, standard_text)}；符号{parts['outer']}；力臂{parts['arm'] or '其他'}；力{parts['force'] or '其他'}"
                if parts["buried_sign"]:
                    shown += "；负号在力臂内"
            else:
                pred = _prediction(bundle_of(target), row["response"], target)
                shown = execution_class(pred, contract, standard_text)
                signs = _matrix_signs(pred or row["response"], target) if target in {"85", "34", "43"} else ""
                if signs:
                    shown = f"{shown}；{signs}"
            changed = "、".join(f"ID{item}" for item in row["changed"]) or "无"
            lines.append(f"| {name} | {row['mode']} | {changed} | `{(pred or '')[:90]}` | {shown} |")
    lines += ["", "## 旁路探针", "", "| 探针 | 条件 | 判断 |", "| --- | --- | --- |"]
    expected = {
        "gamma_dir_before_84": ("相反", _direction_label),
        "gamma_sign_theta": ("负", _sign_word),
        "gamma_sign_named": ("负", _sign_word),
        "hand_after_8": ("左侧", _pole),
        "hand_after_34": ("左侧", _pole),
        "sign_192": ("正", _sign_word),
        "sign_194": ("负", _sign_word),
    }
    for row in sorted(calls.values(), key=lambda item: (item["name"], item["mode"])):
        if row["kind"] != "probe":
            continue
        gold, reader = expected[row["name"]]
        got = reader(_probe_body(row["response"]))
        label = "契约" if got == gold else ("标准" if got and got != gold else "其他")
        lines.append(f"| {row['name']} | {row['mode']} | {label}（{got or _probe_body(row['response'])[:40]}） |")
    lines += ["", "## 符号与力臂拆分", "", "| 目标 | 条件 | 整体符号 | 力臂 |", "| --- | --- | --- | --- |"]
    contract_parts = {"192": ("正", "Xr-Xg"), "194": ("负", "Xr-Xg")}
    for row in sorted((item for item in calls.values() if item["kind"] == "structured"), key=lambda item: (item["target"], item["mode"])):
        parts = _structured_parts(row["response"])
        sign_gold, arm_gold = contract_parts[row["target"]]
        sign = "契约" if parts["outer"] == sign_gold else ("标准" if parts["outer"] else "其他")
        arm = "契约形式" if parts["arm"] == arm_gold else ("反向力臂" if parts["arm"] == "Xg-Xr" else "其他")
        lines.append(f"| ID{row['target']} | {row['mode']} | {sign}（{parts['outer'] or '未解析'}） | {arm}（{parts['arm'] or '未解析'}） |")
    lines.append("")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    bundles, standard = _load()
    for sample_id, bundle in bundles.items():
        bundle["model"] = _model_map(bundle["b_records"])
        bundle["gold"] = _gold_map(bundle["blanks"])
    from cloze_eval import _repair_phrase
    usable_standard = {}
    for process, blanks in standard.items():
        usable_standard[process] = blanks
    jobs = _jobs(bundles)
    print(f"follow-up calls {len(jobs)}", flush=True)
    _ask(jobs)
    _report(bundles, usable_standard)
    print(OUT / "report.md", flush=True)
    _ = _repair_phrase


if __name__ == "__main__":
    main()
