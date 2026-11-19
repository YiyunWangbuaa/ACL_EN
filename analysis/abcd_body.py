# -*- coding: utf-8 -*-
"""Pair the unmarked A–D runs and walk each wrong blank back to a definition.

A is the standard definition with a gold prefix. B is the same definition with
the model's own prefix. C and D repeat those prefixes under the left-handed
body frame. Every call is truncated after the current sentence.

Correctness is the normalized score against that manuscript's own gold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.wind_tunnel.error_graph import NODE_BY_ID, format_trace, trace_error


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "abcd_body"
CONDITIONS = {
    "A": {"dir": "A", "definition_variant": "standard", "prefix_source": "gold", "factor": None},
    "B": {"dir": "B", "definition_variant": "standard", "prefix_source": "model", "factor": None},
    "C": {"dir": "C", "definition_variant": "perturbed", "prefix_source": "gold", "factor": "body_handedness"},
    "D": {"dir": "D", "definition_variant": "perturbed", "prefix_source": "model", "factor": "body_handedness"},
}
BODY = "body_handedness"


def _load(path: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for blank in row.get("blanks") or []:
            item = dict(blank)
            item["process"] = row.get("process")
            found[str(blank["blank_id"])] = item
    if not found:
        raise RuntimeError(f"{path} 里没有评分。")
    return found


def _correct(blank: dict | None) -> bool | None:
    if blank is None or blank.get("normalized") is None:
        return None
    return bool(blank["normalized"])


def _relations(blank_id: str) -> list[dict]:
    node = NODE_BY_ID[blank_id]
    slim = []
    for item in node.public_fields()["definition_relations"]:
        slim.append({
            "definition_id": item["definition_id"].removeprefix("D_"),
            "relation_type": item["relation_type"],
            "ground_truth_changes": item["ground_truth_changes"],
            "path": list(item["path"]),
        })
    return slim


def _item_row(code: str, blank_id: str, blank: dict, changed: bool) -> dict:
    spec = CONDITIONS[code]
    return {
        "item_id": f"ID{blank_id}",
        "condition": code,
        "definition_variant": spec["definition_variant"],
        "prefix_source": spec["prefix_source"],
        "future_context": "truncated",
        "answer_changed_under_perturbation": changed,
        "definition_relations": _relations(blank_id),
        "upstream_dependencies": [f"ID{item}" for item in NODE_BY_ID[blank_id].depends_on],
        "prediction": blank.get("prediction"),
        "gold": blank.get("answer"),
        "correct": _correct(blank),
        "role": NODE_BY_ID[blank_id].role,
        "process": blank.get("process"),
    }


def _flags(correct: dict[str, bool]) -> list[str]:
    flags = []
    if correct["A"] and not correct["C"]:
        flags.append("扰动下新增错误")
    if not correct["A"] and not correct["C"]:
        flags.append("基线错误仍在")
    if not correct["A"] and correct["C"]:
        flags.append("扰动下这条变对")
    if correct["A"] and not correct["B"]:
        flags.append("标准定义下的前缀传播")
    if correct["C"] and not correct["D"]:
        flags.append("扰动定义下的前缀传播")
    if correct["A"] and correct["C"] and correct["B"] and correct["D"]:
        flags.append("四组都对")
    return flags


def _interesting(trace: dict) -> bool:
    """Keep the body path, and any other path whose first wrong step is earlier."""
    if trace["where_error_appears"] in {"origin", "intermediate"}:
        return True
    return (
        trace["definition_key"] == BODY
        and trace["where_error_appears"] in {"direct", "target", "incomplete"}
    )


def _section(blank_id: str, observations: dict, factor: str | None) -> str:
    report = trace_error(blank_id, observations, factor)
    kept = [trace for trace in report["traces"] if _interesting(trace)]
    if not kept:
        return "定义路径上没有更早的错题。已记录的前置步骤是对的，或者这道题不在体轴系手性的下游。"
    report = dict(report)
    report["traces"] = kept
    return format_trace(report)


def main(out: Path = OUT, kind: str = "unmarked") -> None:
    loaded = {}
    for code, spec in CONDITIONS.items():
        path = out / spec["dir"] / "records.jsonl"
        if not path.is_file():
            raise SystemExit(f"还没有 {code} 的记录：{path}")
        loaded[code] = _load(path)
    ids = set(loaded["A"])
    for code, found in loaded.items():
        if set(found) != ids:
            missing = sorted(ids - set(found), key=int)
            extra = sorted(set(found) - ids, key=int)
            raise SystemExit(f"{code} 的空和 A 不一致，缺 {missing}，多 {extra}。")

    order = sorted(ids, key=int)
    changed = {
        blank_id for blank_id in order
        if loaded["A"][blank_id].get("answer") != loaded["C"][blank_id].get("answer")
    }
    rows = []
    for blank_id in order:
        for code in CONDITIONS:
            rows.append(_item_row(code, blank_id, loaded[code][blank_id], blank_id in changed))
    (out / "items.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    by_id: dict[str, dict[str, bool]] = {}
    for blank_id in order:
        by_id[blank_id] = {code: bool(_correct(loaded[code][blank_id])) for code in CONDITIONS}

    def count(predicate) -> list[str]:
        return [blank_id for blank_id in order if predicate(blank_id)]

    groups = {
        "扰动下新增错误": count(lambda i: by_id[i]["A"] and not by_id[i]["C"]),
        "基线错误仍在": count(lambda i: not by_id[i]["A"] and not by_id[i]["C"]),
        "扰动下这条变对": count(lambda i: not by_id[i]["A"] and by_id[i]["C"]),
        "标准定义下的前缀传播": count(lambda i: by_id[i]["A"] and not by_id[i]["B"]),
        "扰动定义下的前缀传播": count(lambda i: by_id[i]["C"] and not by_id[i]["D"]),
    }
    scope_new = [
        blank_id for blank_id in groups["扰动下新增错误"]
        if blank_id not in changed
    ]
    wrong = [blank_id for blank_id in order if not all(by_id[blank_id].values())]

    if kind == "mention":
        lines = [
            "# 规则提及稿上的四组实验",
            "",
            "标准稿是 standard-default，扰动稿是 body-rule。两稿保留全部 18 个规则询问。扰动只把体轴系改为左手系，定义声明与对应的无强调稿相同。",
            "A、C 用各自定义版本的标准答案做前缀。B、D 把模型此前的答案规范化成与标准前缀相同的答案原文后再注入，不保留对话历史。四组都在当前句的句号处截断。",
            "对错按归一化分数，对照的是该稿自己的标准答案。图上的关联只用来定位路径，不单独当成因果。",
            "",
        ]
    else:
        lines = [
            "# 无强调稿上的四组实验",
            "",
            "标准稿是 standard-default-unmarked，扰动稿是 body-unmarked。两稿都去掉全部 18 个规则询问，各留 176 个空。扰动只把体轴系改为左手系。",
            "A、C 用各自定义版本的标准答案做前缀。B、D 把模型此前的答案规范化成与标准前缀相同的答案原文后再注入，不保留对话历史。四组都在当前句的句号处截断。",
            "对错按归一化分数，对照的是该稿自己的标准答案。图上的关联只用来定位路径，不单独当成因果。",
            "",
        ]
    lines.extend([
        "## 正确数",
        "",
        "| 条件 | 定义 | 前缀 | 正确 |",
        "| --- | --- | --- | --- |",
    ])
    for code, spec in CONDITIONS.items():
        hit = sum(by_id[blank_id][code] for blank_id in order)
        lines.append(
            f"| {code} | {spec['definition_variant']} | {spec['prefix_source']} | {hit}/{len(order)} |"
        )
    lines.extend([
        "",
        f"体轴系手性改变了其中 {len(changed)} 个空的标准答案。其余 {len(order) - len(changed)} 个空的标准答案不变。",
        "",
        "## 配对",
        "",
    ])
    for name, found in groups.items():
        lines.append(f"- {name}：{len(found)} 个。{'、'.join(f'ID{item}' for item in found) or '无'}")
    lines.append(
        f"- 标准答案不变、但 A 对 C 错：{len(scope_new)} 个。"
        + ("、".join(f"ID{item}" for item in scope_new) or "无")
    )
    lines.extend(["", "## 扰动稿上前缀错误的落点", ""])
    landing: dict[str, list[str]] = {}
    for blank_id in groups["扰动定义下的前缀传播"]:
        report = trace_error(blank_id, loaded["D"], "body_handedness")
        body = next((item for item in report["traces"] if item["definition_key"] == BODY), None)
        if body is None:
            key = "不在体轴系手性的下游"
        elif body["where_error_appears"] == "incomplete":
            key = f"入口题未出现，已记录的第一次出错是 ID{body['first_wrong']}"
        elif body["first_wrong"] == blank_id:
            key = "体轴系路径上第一次出错就在本题"
        else:
            key = f"体轴系路径上第一次出错在 ID{body['first_wrong']}"
        landing.setdefault(key, []).append(blank_id)
    for key, found in landing.items():
        lines.append(f"- {key}：{'、'.join(f'ID{item}' for item in found)}")
    lines.extend(["", "## 错题路径", ""])
    if not wrong:
        lines.append("四组里没有错题。")
    shown = {
        "扰动下新增错误": ("C", "C"),
        "标准定义下的前缀传播": ("B", "B"),
        "扰动定义下的前缀传播": ("D", "D"),
    }
    for blank_id in wrong:
        flags = [name for name in groups if blank_id in groups[name]]
        node = NODE_BY_ID[blank_id]
        state = " ".join(f"{code}{'对' if by_id[blank_id][code] else '错'}" for code in CONDITIONS)
        lines.append(f"### ID{blank_id} {node.role}")
        lines.append("")
        lines.append(state + "。标准答案" + ("改变。" if blank_id in changed else "不变。"))
        if flags:
            lines.append("配对：" + "；".join(flags) + "。")
        body_rel = [
            item for item in _relations(blank_id) if item["definition_id"] == BODY
        ]
        if body_rel:
            item = body_rel[0]
            route = " → ".join(f"ID{step}" for step in item["path"])
            lines.append(
                f"体轴系手性：{item['relation_type']}，"
                f"{'会' if item['ground_truth_changes'] else '不会'}改变这道题的标准答案。"
                f"路径 {route}。"
            )
        else:
            lines.append("体轴系手性的定义图到不了这道题。")
        for name, (code, _label) in shown.items():
            if blank_id not in groups[name]:
                continue
            lines.append("")
            lines.append(f"沿 {code} 的记录向前看：")
            lines.append("")
            lines.append("```text")
            lines.append(_section(blank_id, loaded[code], CONDITIONS[code]["factor"]))
            lines.append("```")
        if not any(blank_id in groups[name] for name in shown):
            wrong_codes = [code for code in CONDITIONS if not by_id[blank_id][code]]
            code = wrong_codes[0]
            lines.append("")
            lines.append(f"沿 {code} 的记录向前看：")
            lines.append("")
            lines.append("```text")
            lines.append(_section(blank_id, loaded[code], CONDITIONS[code]["factor"]))
            lines.append("```")
        lines.append("")
    (out / "analysis.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out / 'items.jsonl'} and {out / 'analysis.md'}")
    print(f"wrong {len(wrong)} / {len(order)}")
    for name, found in groups.items():
        print(f"{name}: {len(found)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="配对 A–D 并沿定义路径核对错题。")
    parser.add_argument("--kind", choices=("unmarked", "mention"), default="unmarked")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    target = args.out or (
        ROOT / "results" / "abcd_mention" if args.kind == "mention" else OUT
    )
    main(target, args.kind)
