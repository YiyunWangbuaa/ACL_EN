# -*- coding: utf-8 -*-
"""Compile the case catalog into cards, an evidence matrix and experiment inputs.

    python analysis/organize_cases.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domains.wind_tunnel.case_cards import (  # noqa: E402
    check_standard_answers,
    compile_catalog,
    load_catalog,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="整理高价值案例为可复核的案例卡。")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "wind_tunnel" / "cases" / "catalog.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "case_cards")
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    problems = check_standard_answers(catalog, args.samples)
    if problems:
        raise SystemExit("标准答案和配对样本不一致：\n" + "\n".join(problems))
    cards = compile_catalog(catalog, ROOT)
    write_outputs(cards, args.out)
    for card in cards:
        print(
            f"{card['case_id']}\t{card['label']}\t{card['evidence']['evidence_strength']}"
            f"\tparent={card['direct_parent_correct']}\toutput={card['output_relation']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
