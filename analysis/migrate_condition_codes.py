# -*- coding: utf-8 -*-
"""Migrate experiment codes A/B/C'/C''/D -> A/B/C/D/E.

Mapping (old -> new):
    A -> A   gold prefix, truncated
    B -> B   own prefix, truncated
    C' -> C  gold prefix, later text kept
    C'' -> D own prefix, later text kept
    D -> E   end to end

The rename is collision-safe (C'' becomes D while old D becomes E), so it goes
through a temporary directory prefix and placeholder tokens.

Run:
    python analysis/migrate_condition_codes.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

CODE_OLD_TO_NEW = {"A": "A", "B": "B", "C'": "C", "C''": "D", "D": "E"}
LEGEND = (
    "A 空前用标准答案并在句号截断。B 空前用模型自己的原始输出并在句号截断。"
    "C 保留后续正文且空前仍是标准答案。D 保留后续正文且空前是模型自己的原始输出。"
    "E 每个子过程一次调用，全部空都待填，只输出一个 JSON 对象。"
)

# Directory prefixes, ordered so that no rename collides.
DIR_RENAMES = [("run_Cpp", "run_TMPX"), ("run_D", "run_E"), ("run_TMPX", "run_D"),
               ("run_Cp", "run_C")]


def map_text(text: str) -> str:
    """Replace condition codes and run-directory names inside free text."""
    def _cond(match):
        token = match.group(1)
        new = {"Cp": "C", "Cpp": "D", "D": "E", "C'": "C", "C''": "D"}.get(token, token)
        # triple letters keep the temporary token immune to the bare-D rewrite
        return "@@COND" + new * 3 + "@@"

    text = re.sub(r"--condition\s+(\S+)", _cond, text)
    text = text.replace('"C\'\'"', '"@@CPP@@"')
    text = text.replace("C''", "@@CPP@@")
    text = re.sub(r"(?<![A-Za-z])D(?![A-Za-z])", "@@D@@", text)
    text = text.replace("C'", "@@CP@")
    text = text.replace("@@CPP@@", "D").replace("@@D@@", "E").replace("@@CP@", "C")
    text = re.sub(r"@@COND(AAA|BBB|CCC|DDD|EEE)@@",
                  lambda m: "--condition " + m.group(1)[0], text)
    for old, new in (("run_Cpp", "@@RCPP@@"), ("run_D", "@@RD@@"), ("run_Cp", "@@RCP@@")):
        text = text.replace(old, new)
    return text.replace("@@RCPP@@", "run_D").replace("@@RD@@", "run_E").replace("@@RCP@@", "run_C")


def map_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key == "condition" and isinstance(value, str) and value in CODE_OLD_TO_NEW:
                out[key] = CODE_OLD_TO_NEW[value]
            elif isinstance(value, str):
                needs = any(token in value for token in
                            ("run_C", "run_D", "C'", "--condition"))
                out[key] = map_text(value) if needs else value
            else:
                out[key] = map_obj(value)
        return out
    if isinstance(obj, list):
        return [map_obj(item) for item in obj]
    return obj


def rename_dirs():
    moved = []
    for old, new in DIR_RENAMES:
        for path in sorted(RES.iterdir()):
            if not path.is_dir() or not path.name.startswith(old):
                continue
            target = RES / (new + path.name[len(old):])
            if target.exists():
                raise RuntimeError(f"target already exists: {target}")
            path.rename(target)
            moved.append((path.name, target.name))
    return moved


def rewrite_json(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    new = map_obj(data)
    path.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")


def rewrite_records(path: Path, expected: str):
    tmp = path.with_suffix(".jsonl.tmp")
    n = 0
    seen = set()
    with path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cond = rec.get("condition")
            if cond in CODE_OLD_TO_NEW:
                rec["condition"] = CODE_OLD_TO_NEW[cond]
            elif cond not in {"A", "B", "C", "D", "E"}:
                rec["condition"] = expected
            seen.add(rec["condition"])
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n, seen


def rewrite_report_md(path: Path):
    text = path.read_text(encoding="utf-8")
    # the legend line is regenerated from the new definition, not remapped
    lines = []
    for line in text.splitlines():
        if line.startswith("A 空前用标准答案"):
            lines.append(LEGEND)
        else:
            lines.append(map_text(line))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def self_check():
    cases = {
        "条件 C'": "条件 C",
        "条件 C''": "条件 D",
        "条件 D": "条件 E",
        "A/B/C'/C''/D": "A/B/C/D/E",
        "run_Cp_kimi_k3": "run_C_kimi_k3",
        "run_Cpp_qwen_27b": "run_D_qwen_27b",
        "run_D_kimi_k3": "run_E_kimi_k3",
        "--condition Cpp --runs 1": "--condition D --runs 1",
        "--condition Cp --workers 4": "--condition C --workers 4",
        "DeepSeek V4 Flash 的 D（56.6%）": "DeepSeek V4 Flash 的 E（56.6%）",
        "C'−A 与 C''−B": "C−A 与 D−B",
        "D−B 与 D−A": "E−B 与 E−A",
        "C1 530 空、C2 230": "C1 530 空、C2 230",
        "C:/Users/minec/workflow/results/run_Cp": "C:/Users/minec/workflow/results/run_C",
    }
    bad = 0
    for src, want in cases.items():
        got = map_text(src)
        flag = "ok " if got == want else "BAD"
        if got != want:
            bad += 1
        print(f"{flag} {src!r} -> {got!r}" + ("" if got == want else f"  (want {want!r})"))
    return bad


def main():
    marker = RES / ".condition_codes_v2"
    if marker.is_file():
        raise SystemExit(f"{marker} exists: codes were already migrated, refusing to run twice.")
    moved = rename_dirs()
    print(f"renamed {len(moved)} run directories")
    for old, new in moved:
        print(f"  {old} -> {new}")

    counts = {}
    problems = []
    for d in sorted(RES.glob("run_*")):
        if not d.is_dir():
            continue
        for name in ("meta.json", "checkpoint.json"):
            p = d / name
            if p.is_file():
                rewrite_json(p)
        meta_p = d / "meta.json"
        expected = ""
        if meta_p.is_file():
            expected = json.loads(meta_p.read_text(encoding="utf-8")).get("condition") or ""
        rec = d / "records.jsonl"
        if rec.is_file():
            n, seen = rewrite_records(rec, expected)
            counts[d.name] = (n, sorted(seen))
            if seen != {expected}:
                problems.append((d.name, expected, sorted(seen)))
        rep = d / "report.md"
        if rep.is_file():
            rewrite_report_md(rep)

    panel = RES / "panel_checkpoint.json"
    if panel.is_file():
        rewrite_json(panel)

    print(f"rewrote {len(counts)} records.jsonl files, {sum(v[0] for v in counts.values())} records")
    for name, (n, seen) in counts.items():
        print(f"  {name:26s} {n:5d} records  condition={','.join(seen)}")
    if problems:
        print("!! condition mismatch between meta and records:")
        for item in problems:
            print("  ", item)
    else:
        print("all runs consistent: records condition == meta condition")
    # verification
    seen = {}
    for d in sorted(RES.glob("run_*")):
        meta = d / "meta.json"
        if meta.is_file():
            cond = json.loads(meta.read_text(encoding="utf-8")).get("condition")
            seen.setdefault(cond, []).append(d.name)
    for cond in ("A", "B", "C", "D", "E"):
        print(f"  {cond}: {len(seen.get(cond, []))} runs")
    marker.write_text("A/B/C/D/E\n", encoding="utf-8")


if __name__ == "__main__":
    main()
