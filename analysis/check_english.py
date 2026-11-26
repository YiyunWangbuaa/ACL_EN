# -*- coding: utf-8 -*-
"""Check whether the four domain calculators still return Chinese literals."""
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")

files = [
    "domains/aerodynamics/contract_verifiers.py",
    "domains/computer_vision/contract_verifiers.py",
    "domains/robotics/contract_verifiers.py",
    "domains/wind_tunnel/contract_verifiers.py",
    "domains/wind_tunnel/contract_executor.py",
]

pat = re.compile(r'["\'][^"\']*[\u4e00-\u9fff][^"\']*["\']')

for f in files:
    p = ROOT / f
    if not p.exists():
        continue
    hits = []
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        for m in pat.finditer(line):
            hits.append((i, m.group(0)))
    if hits:
        print(f"{f}: {len(hits)} Chinese literals")
        for i, s in hits[:6]:
            print(f"   L{i}: {s}")
    else:
        print(f"{f}: no Chinese literals OK")
