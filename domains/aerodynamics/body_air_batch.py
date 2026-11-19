# -*- coding: utf-8 -*-
"""Declaration samples for the body frame to the airflow frame.

Five definitions enter these blanks: the signs of angle of attack and
sideslip, plus the body frame, the stability frame, and the airflow frame.
Tiers change 1 through all 5 of them. Eight samples are drawn in each tier.

Run::

    python -m domains.aerodynamics.body_air_batch
"""

from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import process_dirs
from domains.aerodynamics.body_air_dataset import BLANK_COUNT, PROCESS, build_record
from domains.aerodynamics.numbering import public_span
from domains.aerodynamics.body_air_generate import legal_stability_frames
from domains.aerodynamics.ground_body_generate import legal_body_frames
from domains.aerodynamics.ground_track_air_generate import legal_airflow_frames
from flight_frame_model import ALPHA, BETA, STANDARD_SPEC


OUT_DIR = process_dirs("aerodynamics", "body_to_airflow")["batch"]
SEED = 20261107
PER_TIER = 8
TIERS = (1, 2, 3, 4, 5)
FACTORS = ["alpha", "beta", "body_frame", "stability_frame", "airflow_frame"]
SCALAR_FLIP = {
    ALPHA: {"sign": "negative"},
    BETA: {"sign": "negative"},
}
SCALAR_ZH = {
    ALPHA: "迎角改为空速投影处于机体轴向下方时为负",
    BETA: "侧滑角改为空速处于飞行器右方时为负",
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
    bodies = legal_body_frames()
    stables = legal_stability_frames()
    airs = legal_airflow_frames()
    if len(bodies) != 48 or len(stables) != 48 or len(airs) != 48:
        raise RuntimeError(
            f"Unexpected frame counts: body {len(bodies)}, stability {len(stables)}, airflow {len(airs)}."
        )
    layouts = {"body_frame": bodies, "stability_frame": stables, "airflow_frame": airs}
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
                if cursor > len(pool) * 40:
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
                elif key == "body_frame":
                    phrases.append(_frame_phrase("机体系", spec[key]))
                elif key == "stability_frame":
                    phrases.append(_frame_phrase("稳定性坐标系", spec[key]))
                else:
                    phrases.append(_frame_phrase("气流系", spec[key]))
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
            if index == 1 or index % 16 == 0 or index == len(samples):
                print(f"wrote {index}/{len(samples)}", flush=True)
    lines = [
        "# 空气动力学定义变更数据",
        "",
        f"子过程：体轴系到气流系，每条 {BLANK_COUNT} 个空，公共编号 {public_span(PROCESS)[0]} 到 {public_span(PROCESS)[1]}。进入这些空的定义有 5 项：迎角符号、侧滑符号、机体系、稳定性坐标系、气流系。地面系、航迹系以及姿态角、航迹角保持当前默认，因为这一子过程不读取它们。",
        "",
        "机体系、稳定性坐标系和气流系各抽取两轴声明的 48 套，它们都和默认的轴向、法向声明不同。",
        "",
        "| 档 | 变更项 | 条数 | 编号 |",
        "| --- | --- | --- | --- |",
    ]
    for tier in TIERS:
        lines.append(f"| {tier}/5 | {tier} | {PER_TIER} | {tier}-01 至 {tier}-0{PER_TIER} |")
    lines.extend([
        "",
        f"抽样种子 {SEED}。数据在 data/flight/basic/body_to_airflow/batch/samples.jsonl，每行一条，含定义声明和 35 个空的标准答案。",
        "",
    ])
    (OUT_DIR / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
