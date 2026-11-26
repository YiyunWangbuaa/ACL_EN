# -*- coding: utf-8 -*-
"""Write a human-readable review of the synced flight-mechanics control."""
from __future__ import annotations

import json

from domain_layout import SETS
from domains.flight_ctrl.fm_ctrl_dataset import FORMAT_EXAMPLE, FORMAT_TEXT, _injection, _render_process

OUT = SETS["reconstruction.flight"]["review"]


def _sample_markdown(row: dict, index: int) -> str:
    source = row["spec"]["source_id"]
    by_id = {item["id"]: item for item in row["questions"]}
    lines = [
        f"# Control {index:02d} {row['id']}",
        "",
        (
            f"Synced with wind-tunnel sample {source}. Installation is fixed upright. "
            "The question is one continuous cloze, split at sentence periods. "
            "Group A fills the current sentence; earlier blanks are filled with the gold text, and later paragraphs are omitted."
        ),
        "",
        "The full process is below. Labels and gold answers inside slots are for review only. The model sees open slots, without answers.",
        "",
        "## Declaration",
        "",
        row["declaration"].strip(),
        "",
        "## Full process",
        "",
        _render_process(by_id),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _index(rows: list[dict]) -> str:
    lines = [
        "# Flight-mechanics control review",
        "",
        (
            "These 10 items are synced with the wind-tunnel pilot_10: the ground frame matches the wind frame, "
            "the body frame matches the body-axis frame, and the positive senses of the theoretical angles and the three mechanism angles agree. "
            "Installation is always upright. Each item has 81 blanks, parallel to wind-tunnel \"wind frame to body-axis frame\" blanks 1-81. "
            "Options use the ground frame, the body frame, the airflow frame, and their axes."
        ),
        "",
        "Sentences follow one another at periods. Blanks in the same sentence are answered together. Earlier sentences are filled with known answers on the next input.",
        "",
        "The live input format is in 00_injection_format.md.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"- [Control {index:02d}](sample_{index:02d}.md): {row['id']}, synced with wind tunnel {row['spec']['source_id']}."
        )
    lines.append("")
    return "\n".join(lines)


def _format_document(example: str) -> str:
    lines = [
        "# Control injection format",
        "",
        (
            "Group A fills the current sentence. The process is written from paragraph 1 through the period of this sentence. "
            "Earlier blanks are filled with the gold text, not option letters. Later paragraphs are omitted. "
            "Blanks in the same sentence go in one JSON object."
        ),
        "",
        "Answers go in one JSON object. A multiple-choice value is a letter. A matrix, vector, expression, arctan, or arcsin is a string in English double quotes.",
        "",
        "Labels and gold answers in the review draft are not shown to the model.",
        "",
        "## Format names",
        "",
    ]
    for code, text in FORMAT_TEXT.items():
        if code in {"choice", "matrix", "vector", "expression", "arctan", "arcsin"}:
            lines.append(f"- {code}: {text} Example: {FORMAT_EXAMPLE[code]}")
    lines.extend([
        "",
        "## One real injection",
        "",
        (
            "Below is the group-A input for control 01, the sentence that contains blank 5. "
            "That sentence fills blanks 5, 6, and 7 together. Earlier blanks are already filled with the gold text."
        ),
        "",
        "```text",
        example,
        "```",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    rows = [
        json.loads(line)
        for line in SETS["reconstruction.flight"]["synced"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 10:
        raise RuntimeError(f"The synced control set must contain 10 items, got {len(rows)}.")
    OUT.mkdir(exist_ok=True)
    for index, row in enumerate(rows, start=1):
        (OUT / f"sample_{index:02d}.md").write_text(_sample_markdown(row, index), encoding="utf-8")
    (OUT / "00_index.md").write_text(_index(rows), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (OUT / "00_injection_format.md").write_text(_format_document(example), encoding="utf-8")
    print(f"wrote 10 reviews -> {OUT}")


if __name__ == "__main__":
    main()
