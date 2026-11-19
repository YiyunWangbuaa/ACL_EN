# -*- coding: utf-8 -*-
"""Compare shallow and deep-rederive probes from one session-gold run and one session-model run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import standard_for  # noqa: E402
from cloze_score import judge  # noqa: E402

DATA = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
ORDER = ("风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点")


def _load(path: Path) -> list[dict]:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _probes(directory: Path) -> list[dict]:
    named = directory / "probes.jsonl"
    rows = _load(named)
    if rows:
        return [item for item in rows if item.get("branch") == "probe"]
    return [item for item in _load(directory / "records.jsonl") if item.get("branch") == "probe"]


def _blank(row: dict) -> dict:
    blanks = row.get("blanks") or []
    if not blanks:
        raise RuntimeError(f"{row.get('probe_id')} 没有评分")
    return blanks[0]


def _label(prediction: str, gold: str, standard: str) -> str:
    hit_gold = judge(prediction, gold, "矩阵")["normalized"]
    same = bool(standard) and judge(gold, standard, "矩阵")["normalized"]
    hit_standard = bool(standard) and judge(prediction, standard, "矩阵")["normalized"]
    if same:
        return "契约与标准相同" if hit_gold else "两者都不是"
    if hit_gold:
        return "契约"
    if hit_standard:
        return "标准"
    return "两者都不是"


def _same(left: str, right: str) -> str:
    if not left or not right:
        return "缺答"
    return "同一张" if judge(left, right, "矩阵")["normalized"] else "不一致"


def _index(rows: list[dict], standard: dict) -> dict[tuple[str, str], dict[str, dict]]:
    found: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        probe_id = str(row.get("probe_id") or "")
        process, name, kind = probe_id.split(":", 2)
        blank = _blank(row)
        gold = str(blank.get("answer") or "")
        prediction = str(blank.get("prediction") or "")
        std = standard.get((process, str(row.get("probe_target") or blank.get("blank_id"))), "")
        found.setdefault((process, name), {})[kind] = {
            "label": _label(prediction, gold, std),
            "prediction": prediction,
            "error": row.get("error") or "",
        }
    return found


def _keys(gold: dict, model: dict) -> list[tuple[str, str]]:
    keys = set(gold) | set(model)
    return sorted(keys, key=lambda item: (ORDER.index(item[0]) if item[0] in ORDER else len(ORDER), item[1]))


def render(gold_dir: Path, model_dir: Path, note: str) -> str:
    standard = standard_for("wind_tunnel", DATA)
    gold = _index(_probes(gold_dir), standard)
    model = _index(_probes(model_dir), standard)
    lines = [
        "# 体轴左手系·无标记，会话注入",
        "",
        f"样本 body-unmarked。模型 deepseek-flash。{note}",
        "定义只在系统消息出现一次。session-gold 的前文是金标，session-model 的前文是模型自己的回答。",
        "分矩阵和总矩阵记分后，主会话里的答案换成占位符。浅层探针和重推导深层探针是分支，答案不写回。",
        "",
        "| 环节 | 矩阵 | gold 浅层 | gold 重推导 | gold 两次 | model 浅层 | model 重推导 | model 两次 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    both_contract = 0
    shallow_only = 0
    deep_only = 0
    neither = 0
    for process, name in _keys(gold, model):
        grow = gold.get((process, name), {})
        mrow = model.get((process, name), {})
        shallow_g = grow.get("shallow", {})
        deep_g = grow.get("deep_rederive", {})
        shallow_m = mrow.get("shallow", {})
        deep_m = mrow.get("deep_rederive", {})
        lines.append(
            "| {process} | {name} | {gs} | {gd} | {gsame} | {ms} | {md} | {msame} |".format(
                process=process,
                name=name,
                gs=shallow_g.get("label", "缺答"),
                gd=deep_g.get("label", "缺答"),
                gsame=_same(shallow_g.get("prediction", ""), deep_g.get("prediction", "")),
                ms=shallow_m.get("label", "缺答"),
                md=deep_m.get("label", "缺答"),
                msame=_same(shallow_m.get("prediction", ""), deep_m.get("prediction", "")),
            )
        )
        for pair in ((shallow_g, deep_g), (shallow_m, deep_m)):
            labels = {item.get("label") for item in pair}
            hit = {"契约", "契约与标准相同"}
            shallow_hit = pair[0].get("label") in hit
            deep_hit = pair[1].get("label") in hit
            if shallow_hit and deep_hit:
                both_contract += 1
            elif shallow_hit:
                shallow_only += 1
            elif deep_hit:
                deep_only += 1
            elif pair[0].get("label") and pair[1].get("label"):
                neither += 1
    lines.extend([
        "",
        "浅层与重推导都判为契约（含契约与标准相同）的有 "
        f"{both_contract} 组。只有浅层是契约的有 {shallow_only} 组。"
        f"只有重推导是契约的有 {deep_only} 组。两边都不是契约的有 {neither} 组。",
        "一组是一种注入方式下的一张矩阵。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, default=ROOT / "results" / "session_body" / "gold")
    parser.add_argument("--model", type=Path, default=ROOT / "results" / "session_body" / "model")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "session_body" / "compare.md")
    parser.add_argument("--note", default="思考关闭，temperature 0。")
    args = parser.parse_args()
    text = render(args.gold, args.model, args.note)
    args.out.write_text(text, encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    main()
