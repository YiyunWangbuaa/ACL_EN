# -*- coding: utf-8 -*-
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CJK = re.compile(r"[\u4e00-\u9fff]")
files = []
for path in ROOT.rglob("*.py"):
    rel = path.relative_to(ROOT).as_posix()
    if "__pycache__" in rel or rel == "analysis/_cjk_scan.py":
        continue
    text = path.read_text(encoding="utf-8")
    n = len(CJK.findall(text))
    if n:
        files.append((n, rel))
files.sort()
print(f"{len(files)} files")
for n, rel in files:
    print(f"{n:5} {rel}")
