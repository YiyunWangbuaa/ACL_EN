# -*- coding: utf-8 -*-
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))
from cloze_score import sympy_equal

RECON = Path("results/recon_vision_flash_think/20261113_pilot10_A/records.jsonl")
CROSS = Path("results/cv_flash_think/20261113_pilot10_A/records.jsonl")


def latex_plain(text: str) -> str:
    value = text or ""
    value = value.replace("\\varphi", "φ").replace("\\phi", "φ").replace("\\vartheta", "θ")
    value = value.replace("\\theta", "θ").replace("\\psi", "ψ")
    value = value.replace("\\alpha", "α").replace("\\beta", "β").replace("\\gamma", "γ")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\ ", "")
    value = re.sub(r"\\(sin|cos|tan)\b", r"\1", value)
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    value = re.sub(r"\b(sin|cos|tan)([αβγθφψ](?:_\d+)?)", r"\1(\2)", value)
    value = re.sub(r"\)(?=[A-Za-zαβγθφψ(])", ")*", value)
    value = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", value)
    return value


def rows(path):
    for line in path.open(encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


cross = {}
for item in rows(CROSS):
    for blank in item["blanks"]:
        cross[(item["sample_id"], str(blank["blank_id"]))] = blank

print("latex matrix equivalence")
for item in rows(RECON):
    for blank in item["blanks"]:
        bid = str(blank["blank_id"])
        if bid not in {"37", "76"}:
            continue
        other = cross[(item["sample_id"], bid)]
        if not (other.get("normalized") and not blank.get("normalized")):
            continue
        pred = blank.get("prediction") or ""
        fixed = latex_plain(pred)
        ok = sympy_equal(fixed, blank.get("answer") or "")
        print(item["sample_id"], bid, "equal" if ok else "DIFF", "latex" if "\\" in pred else "plain")

print("\n145/146 responses")
shown = 0
for item in rows(RECON):
    ids = [str(b["blank_id"]) for b in item["blanks"]]
    if "145" not in ids:
        continue
    if shown >= 3:
        break
    shown += 1
    print(item["sample_id"], ids, "len", len(item.get("response") or ""))
    print((item.get("response") or "")[:300])
