# -*- coding: utf-8 -*-
"""Thinking-off A/B runs for the five unmarked definition pairs plus standards."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT = ROOT / "results" / "nothink_pair"
PY = sys.executable

# Order pairs with the ball-wrist standard, not the unified default.
GROUPS = (
    ("standard", "standard-default-unmarked"),
    ("standard-order", "standard-order-unmarked"),
    ("alpha", "alpha-unmarked"),
    ("roll", "roll-unmarked"),
    ("order", "order-unmarked"),
    ("mount", "mount-unmarked"),
    ("body", "body-unmarked"),
)
PHASES = (
    ("A", "gold-trunc", 2),
    ("B", "self-trunc", 4),
)


def complete(path: Path) -> bool:
    meta = path / "meta.json"
    if not meta.is_file() or not (path / "records.jsonl").is_file():
        return False
    payload = json.loads(meta.read_text(encoding="utf-8"))
    return payload.get("status") == "complete" and payload.get("thinking") == "disabled"


def launch(slug: str, sample: str, code: str, condition: str) -> subprocess.Popen:
    out = OUT / slug / code
    out.mkdir(parents=True, exist_ok=True)
    ids = OUT / "ids" / f"{slug}.json"
    ids.parent.mkdir(parents=True, exist_ok=True)
    ids.write_text(json.dumps([sample], ensure_ascii=False), encoding="utf-8")
    log = open(out / "run.log", "w", encoding="utf-8")
    command = [
        PY, "cloze_eval.py",
        "--domain", "wind_tunnel",
        "--model", "deepseek-flash",
        "--thinking", "disabled",
        "--condition", condition,
        "--runs", "1",
        "--workers", "4",
        "--temperature", "0",
        "--seed", "20261102",
        "--data", str(DATA),
        "--ids-file", str(ids),
        "--out", str(out),
    ]
    print(f"start {slug} {code} {condition}", flush=True)
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def run_phase(code: str, condition: str, limit: int) -> None:
    pending = []
    for slug, sample in GROUPS:
        out = OUT / slug / code
        if complete(out):
            print(f"skip {slug} {code}", flush=True)
            continue
        pending.append((slug, sample))
    active: list[tuple[str, subprocess.Popen]] = []
    while pending or active:
        while pending and len(active) < limit:
            slug, sample = pending.pop(0)
            active.append((slug, launch(slug, sample, code, condition)))
        if not active:
            break
        slug, process = active.pop(0)
        code_return = process.wait()
        print(f"done {slug} {code} exit={code_return}", flush=True)
        if code_return != 0 and not complete(OUT / slug / code):
            raise SystemExit(f"{slug} {code} failed with {code_return}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for code, condition, limit in PHASES:
        run_phase(code, condition, limit)
    print("全部完成", flush=True)


if __name__ == "__main__":
    main()
