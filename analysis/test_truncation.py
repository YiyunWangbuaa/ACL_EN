# -*- coding: utf-8 -*-
"""Offline checks: a truncated thinking reply is wrong, with no second call."""
import sys

sys.path.insert(0, r"C:\Users\minec\Desktop\ACL\workflow")

import cloze_eval as ce

calls = []


def fake_call(client, model, prompt, temperature, thinking, captured=None, max_tokens=None,
              stop=None, seed=None, reasoning_effort=None):
    calls.append({"max_tokens": max_tokens, "effort": reasoning_effort})
    captured["finish_reason"] = "length"
    return "A", "thinking..." * 50


ce.call_model = fake_call

job = {
    "sample_id": "s1", "tier": 25, "process": "test",
    "blank_ids": ["1"], "blanks": [{"blank_id": "1", "label": "A1", "format_code": "无",
                                     "options": ["A", "B"], "answer": "A"}],
    "prompt": "问题", "condition": "A", "run": 0, "max_tokens": ce.THINKING_MAX_TOKENS, "standard": {},
}

assert ce.THINKING_MAX_TOKENS == 32768 * 2
assert ce._resolve_max_tokens(0, "enabled", announce=False) == 65536
assert ce._resolve_max_tokens(16384, "enabled", announce=False) == 65536
assert ce._resolve_max_tokens(100000, announce=False) == 32768

record = ce.score_one(None, "fake-model", 0.0, "enabled", job)
print("calls:", calls)
print("unanswered:", record["unanswered"], "response:", repr(record["response"]),
      "exact:", record["blanks"][0]["exact"], "error:", repr(record["error"]))
assert calls == [{"max_tokens": 65536, "effort": None}], calls
assert record["truncated"] is True
assert record["unanswered"] is True
assert record["response"] == ""
assert record["error"] == ""
assert record["attempts"] == 1
assert record["blanks"][0]["exact"] is False
assert record["blanks"][0]["normalized"] is False
print("strip:", repr(ce._strip_reasoning("先想一段" * 20 + "答案 A", "先想一段" * 20)))
print("memory:", ce._memory_text("x" * 1000)[-3:], len(ce._memory_text("x" * 1000)))
print("truncation ok")
