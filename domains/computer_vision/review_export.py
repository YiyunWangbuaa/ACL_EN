# -*- coding: utf-8 -*-
"""Write a human-readable review of the computer-vision pilot."""
from __future__ import annotations

import json
from pathlib import Path

from domain_layout import SETS
from domains.computer_vision.cv_dataset import _injection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = SETS["cross.wind_vision"]["review"]
SEGMENT_TITLE = {
    "frame": "Absolute directions and coordinate axes",
    "camera_attitude": "Camera attitude",
    "body_attitude": "Object attitude",
    "projection": "Projection of the object nose",
    "axis_wind": "Direction of an image line",
    "extrinsic": "Stereo extrinsics",
    "epipolar": "Epipolar geometry and triangulation",
}


def _options(item: dict) -> str:
    options = item.get("options") or []
    if not options:
        return "none"
    return "; ".join(f"{chr(65 + index)}: {option}" for index, option in enumerate(options))


def _slot(item: dict) -> str:
    return (
        f"{{ID:{item['id']}, options:{_options(item)}, format:{item['format_code']}, "
        f"label:{item['label']}, answer:{item['answer']}}}"
    )


def _process(questions: list[dict], titles: dict | None = None) -> str:
    lines = []
    current = None
    paragraph: list[dict] = []

    def flush() -> None:
        if not paragraph:
            return
        number = paragraph[0].get("paragraph")
        text = f"{number}."
        for item in paragraph:
            text += item.get("lead", "") + _slot(item) + item.get("tail", "")
        lines.append(text)
        lines.append("")
        paragraph.clear()

    for item in questions:
        segment = item.get("segment") or ""
        if segment != current:
            flush()
            current = segment
            title = (titles or SEGMENT_TITLE).get(segment, segment or "process")
            if lines:
                lines.append("")
            lines.extend([f"### {title}", ""])
        if item.get("paragraph"):
            if paragraph and paragraph[0].get("paragraph") != item.get("paragraph"):
                flush()
            paragraph.append(item)
            continue
        flush()
        lines.append(f"{item['id']}.{item['prompt']}{_slot(item)}")
    flush()
    return "\n".join(lines).strip()


def _declaration(text: str) -> str:
    body = text.strip()
    if body.startswith("[Declaration]"):
        body = body[len("[Declaration]"):].strip()
    return body


def _sample_markdown(row: dict, index: int, intro: str | None = None, titles: dict | None = None) -> str:
    lines = [
        f"# Data {index:02d}  {row['id']}",
        "",
        intro or (
            "This file is the review draft for one computer-vision sample. It shows the declaration and every blank. Camera attitude is split into six paragraphs per camera, and the pose translation follows that camera's rotation. Object attitude follows the nine paragraphs of the source draft. Imaging is split into formula, intrinsics, nose position, and skew matrices, with several slots in one paragraph. The remaining blanks stay one question each. The label and answer inside a slot are for review; the blank shown to a model omits both."
        ),
        "",
        row["change"],
        "",
        "## Declaration",
        "",
        _declaration(row["declaration"]),
        "",
        "## Full process",
        "",
        "Every slot below includes its label and gold answer for review. Those fields are not shown on the blank a model fills.",
        "",
        _process(row["questions"], titles),
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _index(rows: list[dict]) -> str:
    lines = [
        "# Computer vision review",
        "",
        "These 10 rows are the scored pilot: tiers 25%, 50%, 75%, and 100%, with 3, 3, 2, and 2 samples. Object attitude follows the world-frame to object-frame process, and the angles are the mechanism angles. The process name is stereo pose.",
        "",
        "The formal input format is in 00_injection_format.md. The end-to-end question is in [end to end](end_to_end.md). Edit the slot in the matching sample to change a stem or a gold answer.",
        "",
        "Camera attitude is split into six paragraphs per camera, and the pose translation follows that camera's rotation. The ids moved relative to the previous round. A low-accuracy blank counted under an old id is not the same question now.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [Data {index:02d}](sample_{index:02d}.md): {row['id']}, {row['change']}")
    lines.append("")
    return "\n".join(lines)


def _format_document(example: str) -> str:
    lines = [
        "# Computer vision injection format",
        "",
        "Group A fills one blank at a time. The input is the declaration, that blank's stem, and the output requirement. Earlier blanks are not carried forward, because each stem is self-contained.",
        "",
        "A choice answer is one option letter. A matrix is written [[a, b, c], [d, e, f], [g, h, i]]. A vector is written [a, b, c], not as a matrix.",
        "",
        "The label and gold answer in the review draft do not appear in the model input.",
        "",
        "## One real injection",
        "",
        "Below is the group-A input for data 01, blank 5.",
        "",
        "```text",
        example,
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rows = [
        json.loads(line)
        for line in SETS["cross.wind_vision"]["samples"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 10:
        raise RuntimeError(f"the computer-vision pilot should have 10 rows, got {len(rows)}.")
    OUT.mkdir(exist_ok=True)
    for index, row in enumerate(rows, start=1):
        if len(row["questions"]) != 165:
            raise RuntimeError(f"{row['id']} has {len(row['questions'])} blanks, expected 165.")
        (OUT / f"sample_{index:02d}.md").write_text(_sample_markdown(row, index), encoding="utf-8")
    (OUT / "00_index.md").write_text(_index(rows), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (OUT / "00_injection_format.md").write_text(_format_document(example), encoding="utf-8")
    print(f"wrote 10 reviews -> {OUT}")


if __name__ == "__main__":
    main()
