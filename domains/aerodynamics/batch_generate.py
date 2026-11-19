# -*- coding: utf-8 -*-
"""Declaration samples for the ground-to-body subprocess.

Five definitions enter these blanks: the signs of yaw, pitch, and roll, plus
the ground frame and the body frame. Tiers change 1, 2, 3, 4, or all 5 of them.
Eight samples are drawn in each tier.

Run::

    python -m domains.aerodynamics.batch_generate
"""

from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import process_dirs
from flight_frame_model import PHI, PSI, STANDARD_SPEC, THETA
from domains.aerodynamics.dataset_generate import BLANK_COUNT, PROCESS, build_record
from domains.aerodynamics.numbering import public_span
from domains.aerodynamics.ground_body_generate import legal_body_frames, legal_ground_frames


OUT_DIR = process_dirs("aerodynamics", "ground_to_body")["batch"]
SEED = 20261105
PER_TIER = 8
TIERS = (1, 2, 3, 4, 5)
FACTORS = ["psi", "theta", "phi", "ground_frame", "body_frame"]
SCALAR_FLIP = {
    PSI: {"sign": "negative"},
    THETA: {"sign": "negative"},
    PHI: {"sign": "negative"},
}
SCALAR_ZH = {
    PSI: "偏航角改为机头向地面右方偏时为负",
    THETA: "俯仰角改为机头向地面上方偏时为负",
    PHI: "滚转角改为向机体右侧滚转时为负",
}


def _frame_phrase(title: str, spec: dict) -> str:
    shown = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
    declared = [
        f"{shown.get(name, name)}沿{axis['along'].split('_', 1)[1]}指向{axis['points'].split('_', 1)[1]}"
        for name, axis in spec.items()
        if name != "chirality"
    ]
    hand = "右手系" if spec["chirality"] == "right" else "左手系"
    return f"{title}改为{hand}，" + "，".join(declared)


def _samples() -> list[dict]:
    grounds = [item for item in legal_ground_frames() if item != STANDARD_SPEC["ground_frame"]]
    bodies = legal_body_frames()
    if len(grounds) != 15 or len(bodies) != 48:
        raise RuntimeError(f"Unexpected frame counts: ground {len(grounds)}, body {len(bodies)}.")
    layouts = {"ground_frame": grounds, "body_frame": bodies}
    rng = random.Random(SEED)
    seen = set()
    samples = []
    for tier in TIERS:
        pool = list(itertools.combinations(FACTORS, tier))
        rng.shuffle(pool)
        made = 0
        cursor = 0
        while made < PER_TIER:
            chosen = list(pool[cursor % len(pool)])
            cursor += 1
            spec = deepcopy(STANDARD_SPEC)
            for key in chosen:
                if key in SCALAR_FLIP:
                    spec[key] = deepcopy(SCALAR_FLIP[key])
                else:
                    spec[key] = deepcopy(rng.choice(layouts[key]))
            signature = json.dumps(spec, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                if cursor > len(pool) * 30:
                    raise RuntimeError(f"Could not draw a unique spec for tier {tier}.")
                continue
            changed = [key for key in FACTORS if spec[key] != STANDARD_SPEC[key]]
            if set(changed) != set(chosen):
                raise RuntimeError(f"Tier {tier} changed {changed}, expected {chosen}.")
            seen.add(signature)
            made += 1
            phrases = []
            for key in chosen:
                if key in SCALAR_ZH:
                    phrases.append(SCALAR_ZH[key])
                elif key == "ground_frame":
                    phrases.append(_frame_phrase("地面系", spec[key]))
                else:
                    phrases.append(_frame_phrase("机体系", spec[key]))
            samples.append({
                "id": f"{tier}-{made:02d}",
                "tier": tier,
                "changed_count": tier,
                "changed": chosen,
                "title": f"变更{tier}项",
                "change": f"这一子过程的 5 项定义中变更 {tier} 项：" + "；".join(phrases) + "。",
                "spec": spec,
            })
    return samples


def _slim(sample: dict, record: dict) -> dict:
    questions = []
    for item in record["questions"]:
        questions.append({
            "id": item["id"],
            "options": item["options"],
            "format_code": item["format_code"],
            "label": item["label"],
            "answer": item["answer"],
        })
    return {
        "id": sample["id"],
        "tier": sample["tier"],
        "changed_count": sample["changed_count"],
        "changed": sample["changed"],
        "title": sample["title"],
        "change": sample["change"],
        "spec": sample["spec"],
        "declaration": record["declaration"],
        "questions": questions,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    path = OUT_DIR / "samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            record = build_record(sample)
            if len(record["questions"]) != BLANK_COUNT:
                raise RuntimeError(f"{sample['id']} has {len(record['questions'])} questions.")
            handle.write(json.dumps(_slim(sample, record), ensure_ascii=False) + "\n")
            if index == 1 or index == len(samples):
                print(f"wrote {index}/{len(samples)}", flush=True)
    lines = [
        "# 空气动力学定义变更数据",
        "",
        f"子过程：地面系到机体系，每条 {BLANK_COUNT} 个空，公共编号 {public_span(PROCESS)[0]} 到 {public_span(PROCESS)[1]}。进入这些空的定义有 5 项：偏航符号、俯仰符号、滚转符号、地面系、机体系。其余飞行力学定义保持当前默认，因为这一子过程不读取它们。",
        "",
        "地面系在含默认的 16 套合法声明里抽取非默认的 15 套。机体系抽取两轴声明的 48 套，它们都和默认的 xb、zb 声明不同。",
        "",
        "| 档 | 变更项 | 条数 | 编号 |",
        "| --- | --- | --- | --- |",
    ]
    for tier in TIERS:
        lines.append(f"| {tier}/5 | {tier} | {PER_TIER} | {tier}-01 至 {tier}-0{PER_TIER} |")
    lines.extend([
        "",
        f"抽样种子 {SEED}。数据在 data/flight/basic/ground_to_body/batch/samples.jsonl，每行一条，含定义声明和 28 个空的标准答案。",
        "",
    ])
    (OUT_DIR / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
