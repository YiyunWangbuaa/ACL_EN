# -*- coding: utf-8 -*-
"""Re-derive panel structure facts from the run directories in results/.

The Word report must not trust a single summary file, so this script walks every
run directory, reads meta.json / records.jsonl, and writes an independent
structure summary to results/panel_structure.json.

Run:
    python analysis/panel_structure.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

MODEL_DIRS = {
    "DeepSeek V4 Flash": {
        "A": "run_A",
        "B": "run_B",
        "C": "run_C",
        "D": "run_D",
        "E": "run_E",
    },
    "Qwen3.8 Flash": {
        "A": "run_A_qwen_flash",
        "B": "run_B_qwen_flash",
        "C": "run_C_qwen_flash",
        "D": "run_D_qwen_flash",
        "E": "run_E_qwen_flash",
    },
    "Qwen3.8 27B": {
        "A": "run_A_qwen_27b",
        "B": "run_B_qwen_27b",
        "C": "run_C_qwen_27b",
        "D": "run_D_qwen_27b",
        "E": "run_E_qwen_27b",
    },
    "Kimi K3": {
        "A": "run_A_kimi_k3",
        "B": "run_B_kimi_k3",
        "C": "run_C_kimi_k3",
        "D": "run_D_kimi_k3",
        "E": "run_E_kimi_k3",
    },
}

EXTRA_DIRS = {}


def load_records(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_dir(dirname: str):
    d = RES / dirname
    if not d.is_dir():
        return None
    meta_p = d / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    recs = load_records(d / "records.jsonl") if (d / "records.jsonl").exists() else []
    blanks = [b for r in recs for b in r.get("blanks", [])]
    exact = sum(1 for b in blanks if b.get("exact"))
    norm = sum(1 for b in blanks if b.get("normalized"))
    def _cost(r):
        c = r.get("cost")
        if isinstance(c, dict):
            c = c.get("usd") or 0.0
        return float(c or 0.0)

    cost = sum(_cost(r) for r in recs)
    errors = Counter(b.get("error_kind") or "" for b in blanks)
    per_call = Counter(len(r.get("blanks", [])) for r in recs)
    per_process = Counter(r.get("process", "") for r in recs)
    per_sample = Counter(r.get("sample_id", "") for r in recs)
    tiers = Counter(r.get("tier") for r in recs)
    labels = Counter(b.get("label", "") for b in blanks)
    return {
        "dir": dirname,
        "model_field": meta.get("model"),
        "condition": meta.get("condition"),
        "thinking": meta.get("thinking"),
        "samples": meta.get("samples", []),
        "n_samples": len(meta.get("samples", [])),
        "sentences_meta": meta.get("sentences"),
        "blanks_meta": meta.get("blanks"),
        "calls": len(recs),
        "blanks": len(blanks),
        "exact": exact,
        "norm": norm,
        "cost": round(cost, 6),
        "exact_rate": exact / len(blanks) if blanks else None,
        "norm_rate": norm / len(blanks) if blanks else None,
        "meta_exact": meta.get("exact"),
        "meta_norm": meta.get("normalized"),
        "per_call": dict(sorted(per_call.items())),
        "per_process_calls": dict(per_process),
        "per_sample_calls": dict(per_sample),
        "tier_samples": dict(tiers),
        "labels": dict(labels),
        "error_kinds": {k: v for k, v in errors.items() if k},
        "status": meta.get("status"),
    }


def numbering_scheme(dirname: str):
    """Per-process restart (1..n) vs continuous (1..193) blank numbering."""
    recs = load_records(RES / dirname / "records.jsonl")
    ids_by_proc = defaultdict(list)
    for r in recs:
        ids_by_proc[r.get("process", "")].extend(int(x) for x in r.get("blank_ids", []))
    starts = {p: min(v) for p, v in ids_by_proc.items() if v}
    maxima = {p: max(v) for p, v in ids_by_proc.items() if v}
    restarts = sum(1 for p, v in starts.items() if v == 1)
    return {
        "process_starts": starts,
        "process_max": maxima,
        "restart_at_1": restarts,
        "scheme": "per-process" if restarts > 1 else "continuous",
    }


def prompt_parity(dir1: str, dir2: str):
    """Align calls per (sample, process) by first blank id and compare prompt bytes."""
    def build(dirname):
        g = defaultdict(list)
        for r in load_records(RES / dirname / "records.jsonl"):
            ids = [int(x) for x in r.get("blank_ids", [])]
            if not ids:
                continue
            g[(r["sample_id"], r.get("process", ""))].append(
                (min(ids), hashlib.sha1((r.get("prompt") or "").encode("utf-8")).hexdigest())
            )
        for k in g:
            g[k].sort()
        return g

    a, b = build(dir1), build(dir2)
    groups = sorted(set(a) | set(b))
    aligned = identical = 0
    for k in groups:
        x, y = a.get(k, []), b.get(k, [])
        if len(x) != len(y):
            continue
        aligned += 1
        identical += sum(1 for p, q in zip(x, y) if p[1] == q[1])
    return {"groups": len(groups), "aligned_groups": aligned, "identical_prompts": identical}


def main():
    out = {"models": {}, "extra": {}, "checks": {}}
    for model, conds in MODEL_DIRS.items():
        out["models"][model] = {}
        for cond, dirname in conds.items():
            s = summarize_dir(dirname)
            if s:
                out["models"][model][cond] = s
    for dirname, label in EXTRA_DIRS.items():
        s = summarize_dir(dirname)
        if s:
            out["extra"][label] = s

    # consistency: blanks per sample must be 193, samples 10, meta counts equal records counts
    checks = []
    for group in ("models", "extra"):
        for model, conds in out[group].items():
            items = conds.items() if group == "models" else []
            for cond, s in items:
                tag = f"{model}/{cond}"
                checks.append({
                    "run": tag,
                    "samples": s["n_samples"],
                    "blanks": s["blanks"],
                    "blanks_per_sample": round(s["blanks"] / s["n_samples"], 2) if s["n_samples"] else None,
                    "calls": s["calls"],
                    "meta_matches_records": (s["meta_exact"] == s["exact"] and s["meta_norm"] == s["norm"])
                    if s["meta_exact"] is not None else None,
                })
    out["checks"] = checks

    # global structure facts
    a_run = out["models"]["Kimi K3"]["A"]
    out["structure"] = {
        "samples": a_run["n_samples"],
        "blanks_per_sample": a_run["blanks"] // a_run["n_samples"],
        "calls_per_sample_sentence_conditions": a_run["calls"] // a_run["n_samples"],
        "sentence_calls_per_sample": a_run["sentences_meta"] // a_run["n_samples"],
        "per_call_blank_hist": a_run["per_call"],
        "labels": a_run["labels"],
        "tier_samples": a_run["tier_samples"],
        "per_process_calls": a_run["per_process_calls"],
        "e_calls": out["models"]["Kimi K3"]["E"]["calls"],
        "e_blanks": out["models"]["Kimi K3"]["E"]["blanks"],
    }

    # protocol checks: numbering scheme and prompt parity against the reference model
    protocol = {"numbering": {}, "parity_vs_kimi": {}}
    for model, conds in MODEL_DIRS.items():
        d = conds["A"]
        if (RES / d / "records.jsonl").exists():
            protocol["numbering"][model] = numbering_scheme(d)
    for cond, d in MODEL_DIRS["DeepSeek V4 Flash"].items():
        ref = MODEL_DIRS["Kimi K3"].get(cond)
        if ref and (RES / d / "records.jsonl").exists() and (RES / ref / "records.jsonl").exists():
            protocol["parity_vs_kimi"][cond] = prompt_parity(d, ref)
    for cond in ("A", "B", "C", "D"):
        d = MODEL_DIRS["Qwen3.8 Flash"][cond]
        ref = MODEL_DIRS["Kimi K3"][cond]
        if (RES / d / "records.jsonl").exists():
            protocol["parity_vs_kimi"].setdefault("qwen_" + cond, prompt_parity(d, ref))
    out["protocol"] = protocol

    (RES / "panel_structure.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print("== per run ==")
    for model, conds in out["models"].items():
        for cond, s in conds.items():
            print(f"{model:22s} {cond:3s} calls={s['calls']:5d} blanks={s['blanks']:5d} "
                  f"exact={s['exact_rate']*100:5.1f}% norm={s['norm_rate']*100:5.1f}% "
                  f"cost=${s['cost']:.4f} meta_ok={s['meta_exact']==s['exact'] and s['meta_norm']==s['norm']}")
    for label, s in out["extra"].items():
        print(f"{label:22s} {'':3s} calls={s['calls']:5d} blanks={s['blanks']:5d} "
              f"exact={s['exact_rate']*100:5.1f}% norm={s['norm_rate']*100:5.1f}% cost=${s['cost']:.4f}")
    print("\n== structure ==")
    print(json.dumps(out["structure"], ensure_ascii=False, indent=1))
    print("\nwrote results/panel_structure.json")


if __name__ == "__main__":
    main()
