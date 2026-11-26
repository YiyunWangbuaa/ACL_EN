# -*- coding: utf-8 -*-
"""Stage 0: regenerated gold must match the review manuscript for 1–59."""
from __future__ import annotations

from domains.aerodynamics.flow_generate import _check_standard
from domains.aerodynamics.manuscript import gold_scored, manuscript_answers, record, samples_json_answers
from domains.aerodynamics.pipeline_scope import NODE_IDS


def check_stage0() -> None:
    built = record()
    _check_standard(built)
    scored = gold_scored()
    from_md = manuscript_answers()
    from_json = samples_json_answers()
    missing = [node for node in NODE_IDS if node not in from_md]
    if missing:
        raise RuntimeError(f"The review manuscript is missing blanks: {missing[:8]}")
    md_diff = {node: (scored[node], from_md[node]) for node in NODE_IDS if scored[node] != from_md[node]}
    json_diff = {node: (scored[node], from_json[node]) for node in NODE_IDS if scored[node] != from_json.get(node)}
    if md_diff:
        raise RuntimeError(f"Generator and review manuscript differ: {dict(list(md_diff.items())[:5])}")
    if json_diff:
        raise RuntimeError(f"Generator and samples.json differ: {dict(list(json_diff.items())[:5])}")
    print(f"Stage 0 manuscript: gold difference is empty for 1-131, {len(NODE_IDS)} blanks.", flush=True)


if __name__ == "__main__":
    check_stage0()
