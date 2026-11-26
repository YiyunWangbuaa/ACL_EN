# -*- coding: utf-8 -*-
"""Vision-context copy of a stereo-pose record.

The generator already writes English. This step keeps question order, labels,
and answers, renames the mechanism symbols to the manuscript letters, and
marks the process as pure stereo pose.
"""
from __future__ import annotations

import json

from domain_layout import SETS


def rewrite(text: str) -> str:
    if not text:
        return text
    return text.replace("α_M", "θ_M").replace("β_M", "ψ_M").replace("γ_M", "φ_M")


def apply_record(row: dict) -> dict:
    out = dict(row)
    out["title"] = rewrite(row["title"])
    out["change"] = rewrite(row["change"])
    out["declaration"] = rewrite(row["declaration"])
    questions = []
    for item in row["questions"]:
        copied = dict(item)
        for key in ("prompt", "lead", "tail", "answer"):
            if key in copied and isinstance(copied[key], str):
                copied[key] = rewrite(copied[key])
        if copied.get("options"):
            copied["options"] = [rewrite(option) for option in copied["options"]]
        copied["process"] = "pure stereo pose"
        questions.append(copied)
    out["questions"] = questions
    return out


def leftover(text: str) -> list[str]:
    words = (
        "schlieren",
        "wind tunnel",
        "airspeed",
        "shock",
        "fuselage",
        "sting",
        "balance",
    )
    return [word for word in words if word in text]


def export_review(rows: list[dict]) -> None:
    from domains.computer_vision.cv_dataset import _injection
    from domains.computer_vision import review_export as rx

    out = SETS["reconstruction.vision"]["review"]
    out.mkdir(exist_ok=True)
    intro = (
        "This file is the review draft in the pure vision wording. Question order, labels, "
        "and gold answers match the corrected vision data. The label and answer inside a slot "
        "are for review; the blank shown to a model omits both."
    )
    titles = dict(rx.SEGMENT_TITLE)
    titles["body_attitude"] = "Object attitude"
    titles["projection"] = "Projection of the object nose"
    for index, row in enumerate(rows, start=1):
        (out / f"sample_{index:02d}.md").write_text(
            rx._sample_markdown(row, index, intro=intro, titles=titles), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (out / "00_injection_format.md").write_text(rx._format_document(example), encoding="utf-8")
    lines = [
        "# Pure vision review",
        "",
        f"These {len(rows)} rows match the corrected vision data: {len(rows[0]['questions'])} blanks, with the same labels and gold answers. The world frame replaces the former wind frame, the object frame replaces the former body frame, the reference direction replaces the former airspeed, and the object reference point replaces the former shock location.",
        "",
        "The formal input format is in 00_injection_format.md. The end-to-end question is in [end to end](end_to_end.md). Edit a slot in the matching sample to change a stem.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [Data {index:02d}](sample_{index:02d}.md): {row['id']}, {row['change']}")
    lines.append("")
    (out / "00_index.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    source = SETS["cross.wind_vision"]["samples"]
    target = SETS["reconstruction.vision"]["samples"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = []
    for row in rows:
        vision = apply_record(row)
        if len(vision["questions"]) != len(row["questions"]):
            raise RuntimeError(f"{row['id']} question count changed")
        for old, new in zip(row["questions"], vision["questions"]):
            if old["label"] != new["label"] or old["id"] != new["id"]:
                raise RuntimeError(f"{row['id']} blank {old['id']} label changed")
            if len(old.get("options") or []) != len(new.get("options") or []):
                raise RuntimeError(f"{row['id']} blank {old['id']} option count changed")
            renamed = old["answer"].replace("α_M", "θ_M").replace("β_M", "ψ_M").replace("γ_M", "φ_M")
            if new["answer"] != renamed:
                raise RuntimeError(f"{row['id']} blank {old['id']} answer changed")
            if old["answer"] in set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and new["answer"] != old["answer"]:
                raise RuntimeError(f"{row['id']} blank {old['id']} option letter changed")
        blob = vision["declaration"] + vision["change"] + vision["title"]
        for item in vision["questions"]:
            blob += item["prompt"] + item.get("lead", "") + item.get("tail", "")
            blob += "".join(item.get("options") or [])
        bad = leftover(blob)
        if bad:
            raise RuntimeError(f"{row['id']} still uses wind-tunnel wording: {bad}")
        written.append(vision)
        print(f"ok {row['id']}", flush=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written), encoding="utf-8")
    print(f"wrote {target}")
    export_review(written)


if __name__ == "__main__":
    main()
