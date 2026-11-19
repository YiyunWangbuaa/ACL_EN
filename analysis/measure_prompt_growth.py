# -*- coding: utf-8 -*-
"""Measure how the sentence-level prompt grows, and check for reasoning leakage."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

import cloze_eval as ce  # noqa: E402

SAMPLES = {
    "wind_tunnel": ROOT / "data/wind_tunnel/batch/samples.jsonl",
    "flight_pilot": ROOT / "data/flight/basic/pilot/samples.jsonl",
}

for name, path in SAMPLES.items():
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    chains = ce.contexts_for_flight(row) if name == "flight_pilot" else ce.contexts_for(row)
    print(f"\n== {name}: {len(chains)} chains ==")
    for chain in chains[:2]:
        lengths = []
        for index, sentence in enumerate(chain["sentences"]):
            prompt = ce.prompt_for(chain, sentence["blank_ids"], "A", None, (sentence.get("local_ids") or [None])[0])
            lengths.append(len(prompt))
        print(f"  {chain['process']}: sentences={len(lengths)} "
              f"first={lengths[0]} last={lengths[-1]} growth={lengths[-1] - lengths[0]}")
