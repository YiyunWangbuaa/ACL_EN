# -*- coding: utf-8 -*-
"""Run one real chain with a fake thinking client and check nothing leaks back."""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\minec\Desktop\ACL\workflow")
sys.path.insert(0, str(ROOT))

import cloze_eval as ce

row = json.loads((ROOT / "data/wind_tunnel/batch/samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
chain = ce.contexts_for(row)[0]
records = []
counter = {"n": 0}


def fake_call(client, model, prompt, temperature, thinking, captured=None, max_tokens=None,
              stop=None, seed=None, reasoning_effort=None):
    counter["n"] += 1
    captured["finish_reason"] = "stop"
    ids = list(chain["sentences"][min(counter["n"] - 1, len(chain["sentences"]) - 1)]["blank_ids"])
    body = {blank_id: "A" for blank_id in ids if len(ids) == 1}
    content = json.dumps(body) if body else "A"
    return content, f"LONG-REASONING-{counter['n']}-" + "思考" * 200


ce.call_model = fake_call
ran = ce._execute_chain(chain, "B", 0, {}, None, "fake", 0.0, "enabled", 0, 8192, None, {},
                        records.append)
leaks = [r["blank_ids"] for r in records if "LONG-REASONING" in r["prompt"]]
print(f"sentences run {ran}, prompts {len(records)}, leaks {len(leaks)}")
print("max prompt chars:", max(len(r["prompt"]) for r in records))
assert not leaks
print("no reasoning in any prompt")
