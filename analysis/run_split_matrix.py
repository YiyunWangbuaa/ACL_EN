# -*- coding: utf-8 -*-
"""One definition, A/B, thinking effort low. Compare first matrix answers with the pre-product expansions."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_eval import ROOT as EVAL_ROOT  # noqa: E402
from cloze_eval import load_samples  # noqa: E402
from cloze_score import sympy_equal  # noqa: E402

NAMED = {
    "R_α_M": "迎角旋转矩阵",
    "R_β_M": "侧滑旋转矩阵",
    "R_γ_M": "滚转旋转矩阵",
    "RP1": "前置矩阵RP1",
    "RP2": "前置矩阵RP2",
    "RTP_TZ1": "对齐矩阵",
    "RTP_TZ2": "正装或倒装的前置矩阵",
    "R_(γ_M-γ_tp)": "安装滚转矩阵",
}


def _symbol_name(item: dict) -> str:
    options = [str(option) for option in (item.get("options") or [])]
    answer = str(item.get("answer") or "")
    if options and len(answer) == 1 and answer.isalpha():
        return options[ord(answer.upper()) - ord("A")]
    return answer

DATA = EVAL_ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = EVAL_ROOT / "results" / "think_low_body"
SAMPLE = "body-unmarked"
PY = sys.executable


def _questions() -> list[dict]:
    row = load_samples(DATA, {SAMPLE})[0]
    return row["questions"]


def _pairs(questions: list[dict]) -> list[dict]:
    by_id = {str(item["id"]): item for item in questions}
    symbols = []
    expansions = []
    matrices = []
    for item in questions:
        label = str(item.get("label") or "")
        role = str(item.get("role") or "")
        text = role + label
        if "展开式" in text:
            expansions.append(item)
        elif "连乘式中的R" in text:
            symbols.append(item)
        elif "矩阵" in text:
            matrices.append(item)
    found = []
    for expansion in expansions:
        key = ""
        for token in ("R1", "R2", "R3", "R4", "R5"):
            if token in (expansion.get("label") or "") or token in (expansion.get("role") or ""):
                key = token
                break
        symbol = next((
            item for item in symbols
            if key and item["process"] == expansion["process"] and (item.get("role") or "").endswith(key)
        ), None)
        name = _symbol_name(symbol) if symbol else ""
        needle = NAMED.get(name, name)
        earlier = [
            item for item in matrices
            if item["process"] == expansion["process"]
            and int(item["id"]) < int(expansion["id"])
            and needle and needle in (item.get("role") or "")
            and "展开式" not in (item.get("role") or "")
            and "连乘式" not in (item.get("role") or "")
        ]
        source = earlier[-1] if earlier else None
        found.append({
            "process": expansion["process"],
            "factor": key,
            "symbol": name,
            "first_id": "" if source is None else str(source["id"]),
            "expand_id": str(expansion["id"]),
            "gold_same": bool(source) and sympy_equal(str(source["answer"]), str(expansion["answer"])),
        })
        _ = by_id
    if not found:
        raise SystemExit("这份稿里没有“计算前分矩阵展开”题。")
    return found


def _complete(path: Path) -> bool:
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("status") == "complete" and meta.get("thinking") == "enabled"


def _launch(condition: str, folder: str) -> int:
    out = OUT / folder
    if _complete(out):
        print(f"skip {folder}", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    ids = OUT / "ids.json"
    ids.write_text(json.dumps([SAMPLE], ensure_ascii=False), encoding="utf-8")
    env = os.environ.copy()
    env["DEEPSEEK_REASONING_EFFORT"] = "low"
    command = [
        PY, "cloze_eval.py",
        "--domain", "wind_tunnel",
        "--model", "deepseek-flash",
        "--thinking", "enabled",
        "--condition", condition,
        "--runs", "1",
        "--workers", "4",
        "--temperature", "0",
        "--seed", "20261102",
        "--data", str(DATA),
        "--ids-file", str(ids),
        "--out", str(out),
    ]
    print(f"start {folder} {condition} thinking=low", flush=True)
    log = open(out / "run.log", "w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    code = process.wait()
    print(f"done {folder} exit={code}", flush=True)
    if code != 0 and not _complete(out):
        raise SystemExit(f"{folder} failed with {code}")
    return code


def _records(folder: str) -> dict[str, dict]:
    found = {}
    for line in (OUT / folder / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for blank in row.get("blanks") or []:
            found[str(blank["blank_id"])] = blank
    return found


def _same(left: str, right: str) -> bool:
    if not left or not right:
        return False
    return sympy_equal(left, right)


def _report(pairs: list[dict]) -> None:
    stored = {name: _records(name) for name in ("A", "B")}
    lines = [
        "# 分矩阵前后一致性",
        "",
        "定义稿：体轴左手系·无标记。模型 deepseek-flash，思考开启，强度 low。",
        "A 是标准答案前缀，B 是模型自身前缀。比较的是同一张分矩阵在最初提问时的回答，和计算总矩阵前再次展开时的回答。",
        "",
        "| 过程 | 因子 | 最初题号 | 计算前题号 | 符号 | A最初对金标 | A计算前对金标 | A两次一致 | B最初对金标 | B计算前对金标 | B两次一致 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair in pairs:
        cells = []
        for name in ("A", "B"):
            first = stored[name].get(pair["first_id"], {})
            second = stored[name].get(pair["expand_id"], {})
            cells.append("是" if first.get("normalized") else "否")
            cells.append("是" if second.get("normalized") else "否")
            cells.append("一致" if _same(first.get("prediction") or "", second.get("prediction") or "") else "不一致")
        lines.append(
            f"| {pair['process']} | {pair['factor']} | ID{pair['first_id']} | ID{pair['expand_id']} | {pair['symbol']} | "
            + " | ".join(cells) + " |"
        )
    missing = [pair for pair in pairs if not pair["first_id"]]
    if missing:
        lines += ["", "这些展开式没有对上最初的矩阵题：" + "、".join(pair["expand_id"] for pair in missing)]
    drifted = [pair for pair in pairs if pair["first_id"] and not pair["gold_same"]]
    if drifted:
        lines += ["", "这些配对的两份金标代数上不相同：" + "、".join(f"ID{pair['first_id']}/ID{pair['expand_id']}" for pair in drifted)]
    lines.append("")
    (OUT / "consistency.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pairs = _pairs(_questions())
    print(f"pairs {len(pairs)}", flush=True)
    for pair in pairs:
        print(f"{pair['process']} {pair['factor']} ID{pair['first_id']} -> ID{pair['expand_id']} {pair['symbol']}", flush=True)
    _launch("gold-trunc", "A")
    _launch("self-trunc", "B")
    _report(pairs)
    print(OUT / "consistency.md", flush=True)


if __name__ == "__main__":
    main()
