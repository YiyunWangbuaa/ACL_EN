# -*- coding: utf-8 -*-
"""Ten flight-mechanics samples in the same four tiers as the wind-tunnel pilot.

The wind-tunnel test draws 3, 3, 2, and 2 samples from the 25%, 50%, 75%,
and 100% tiers. Here the domain has 13 definitions that enter a cloze: eight
angle signs and five frames. A tier of k percent changes exactly 13*k//100 of
them. Euler order is fixed and is not a definition.

Each sample carries one declaration and all three subprocesses. Public blank
numbers run from 1 through 103: ground to body, then ground to track and
airflow, then body to airflow.

Run::

    python -m domains.aerodynamics.pilot_generate
"""

from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import SETS
from domains.aerodynamics.body_air_dataset import build_record as build_body_air
from domains.aerodynamics.body_air_generate import legal_stability_frames
from domains.aerodynamics.dataset_generate import build_record as build_ground_body
from domains.aerodynamics.ground_body_generate import BLANK_COUNT as BODY_BLANKS
from domains.aerodynamics.ground_body_generate import legal_body_frames, legal_ground_frames
from domains.aerodynamics.ground_track_air_dataset import PRIOR, _check_declaration, _declaration
from domains.aerodynamics.ground_track_air_dataset import build_record as build_track_air
from domains.aerodynamics.ground_track_air_generate import BLANK_COUNT as TRACK_BLANKS
from domains.aerodynamics.ground_track_air_generate import legal_airflow_frames, legal_track_frames
from domains.aerodynamics.body_air_generate import BLANK_COUNT as AIR_BLANKS
from flight_frame_model import (
    ALPHA,
    BETA,
    GAMMA,
    MU,
    PHI,
    PSI,
    STANDARD_SPEC,
    THETA,
    VARPHI,
    FlightFrameModel,
)


OUT_DIR = SETS["basic.flight"]["pilot"]
SEED = 20261108
TIERS = (25, 50, 75, 100)
TIER_COUNTS = {25: 3, 50: 3, 75: 2, 100: 2}
FACTORS = [
    PSI, THETA, PHI, VARPHI, MU, GAMMA, ALPHA, BETA,
    "ground_frame", "body_frame", "stability_frame", "track_frame", "airflow_frame",
]
BLANK_COUNT = BODY_BLANKS + TRACK_BLANKS + AIR_BLANKS
BUILDS = (build_ground_body, build_track_air, build_body_air)
SCALAR_FLIP = {name: {"sign": "negative"} for name in FACTORS if not name.endswith("_frame")}
SCALAR_ZH = {
    PSI: "偏航角改为机头向地面右方偏时为负",
    THETA: "俯仰角改为机头向地面上方偏时为负",
    PHI: "滚转角改为向机体右侧滚转时为负",
    VARPHI: "航迹方位角改为空速投影在 xg 右侧时为负",
    MU: "航迹倾斜角改为空速向地面上方偏时为负",
    GAMMA: "航迹滚转角改为飞行器向机体右方滚转时为负",
    ALPHA: "迎角改为空速投影处于机体轴向下方时为负",
    BETA: "侧滑角改为空速处于飞行器右方时为负",
}
FRAME_TITLE = {
    "ground_frame": "地面系",
    "body_frame": "机体系",
    "stability_frame": "稳定性坐标系",
    "track_frame": "航迹系",
    "airflow_frame": "气流系",
}
_SPIN = "航迹旋转采用动轴内旋。"
_SPIN_ALL = "姿态旋转、航迹旋转和气流角旋转采用动轴内旋。"


def _width(tier: int) -> int:
    return len(FACTORS) * tier // 100


def _frame_phrase(title: str, spec: dict) -> str:
    shown = {"Xb": "xb", "Yb": "yb", "Zb": "zb"}
    declared = [
        f"{shown.get(name, name)}沿{axis['along'].split('_', 1)[1]}指向{axis['points'].split('_', 1)[1]}"
        for name, axis in spec.items()
        if name != "chirality"
    ]
    hand = "右手系" if spec["chirality"] == "right" else "左手系"
    return f"{title}改为{hand}，" + "，".join(declared)


def _change_text(tier: int, changed: list[str], spec: dict) -> str:
    phrases = []
    for key in changed:
        if key in SCALAR_ZH:
            phrases.append(SCALAR_ZH[key])
        else:
            phrases.append(_frame_phrase(FRAME_TITLE[key], spec[key]))
    return f"定义变更档{tier}%：13项可变定义中变更{len(changed)}项：" + "；".join(phrases) + "。"


def _layouts() -> dict[str, list[dict]]:
    grounds = [item for item in legal_ground_frames() if item != STANDARD_SPEC["ground_frame"]]
    bodies = legal_body_frames()
    stables = legal_stability_frames()
    tracks = legal_track_frames()
    airs = legal_airflow_frames()
    if (len(grounds), len(bodies), len(stables), len(tracks), len(airs)) != (15, 48, 48, 48, 48):
        raise RuntimeError("Unexpected legal frame counts.")
    for key, frames in (
        ("body_frame", bodies),
        ("stability_frame", stables),
        ("track_frame", tracks),
        ("airflow_frame", airs),
    ):
        if STANDARD_SPEC[key] in frames:
            raise RuntimeError(f"{key} includes the standard declaration.")
    return {
        "ground_frame": grounds,
        "body_frame": bodies,
        "stability_frame": stables,
        "track_frame": tracks,
        "airflow_frame": airs,
    }


def _samples() -> list[dict]:
    if len(FACTORS) != 13 or sum(TIER_COUNTS.values()) != 10:
        raise RuntimeError("The four tiers must cover 13 definitions and 10 samples.")
    if [_width(tier) for tier in TIERS] != [3, 6, 9, 13]:
        raise RuntimeError("Tier widths drifted from 3, 6, 9, 13.")
    layouts = _layouts()
    rng = random.Random(SEED)
    seen = set()
    samples = []
    for tier in TIERS:
        width = _width(tier)
        pool = list(itertools.combinations(FACTORS, width))
        rng.shuffle(pool)
        made = 0
        cursor = 0
        while made < TIER_COUNTS[tier]:
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
            samples.append({
                "id": f"{tier:03d}-{made:03d}",
                "tier": tier,
                "changed_count": len(chosen),
                "changed": chosen,
                "title": f"定义变更{tier}%",
                "change": _change_text(tier, chosen, spec),
                "spec": spec,
            })
    return samples


def _domain_declaration(spec: dict) -> str:
    model = FlightFrameModel().declare(deepcopy(spec))
    text = _declaration(model)
    if _SPIN not in text:
        raise RuntimeError("The shared declaration no longer has the expected fixed lines.")
    text = text.replace(_SPIN, _SPIN_ALL, 1)
    _check_declaration(text)
    leaked = [item for item in ("欧拉顺序", "航迹顺序", "气流顺序") if item in text]
    if leaked:
        raise RuntimeError("The composition order is still in the declaration.")
    return text


def _combine(sample: dict) -> tuple[str, list[dict], list[tuple[str, str]]]:
    declaration = _domain_declaration(sample["spec"])
    questions = []
    sections = []
    expected = (BODY_BLANKS, TRACK_BLANKS, AIR_BLANKS)
    for build, size in zip(BUILDS, expected):
        record = build(sample)
        if len(record["questions"]) != size:
            raise RuntimeError(f"{record['process']} has {len(record['questions'])} blanks.")
        sections.append((record["process"], record["process_text"]))
        questions.extend(record["questions"])
    public = [item["id"] for item in questions]
    if public != [str(number) for number in range(1, BLANK_COUNT + 1)]:
        raise RuntimeError(f"Public ids are {public[0]}..{public[-1]}, expected 1..{BLANK_COUNT}.")
    return declaration, questions, sections


def _slim(sample: dict, declaration: str, questions: list[dict]) -> dict:
    stored = []
    for item in questions:
        stored.append({
            "process": item["process"],
            "id": item["id"],
            "local_id": item["local_id"],
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
        "declaration": declaration,
        "prior": PRIOR,
        "questions": stored,
        "domain": "飞行力学",
    }


def _markdown(sample: dict, declaration: str, sections: list[tuple[str, str]]) -> str:
    lines = [
        f"# 数据 {sample['id']} {sample['title']}",
        "",
        "本文件是飞行力学领域的试跑样本。三个子过程编号连成 1 到 121。",
        "",
        sample["change"],
        "",
        "## 定义声明",
        "",
        declaration.strip(),
        "",
        "## 物理常识",
        "",
        PRIOR,
        "",
        "## 完整过程",
        "",
        "下面每个槽都带标签和标准答案，便于直接审阅。这是审阅字段，不是模型在待填空上看到的字段。",
        "",
    ]
    for process, text in sections:
        lines.extend([f"### {process}", "", text, ""])
    return "\n".join(lines)


def _index(samples: list[dict]) -> str:
    lines = [
        "# 飞行力学试跑样本",
        "",
        "分档方式和风洞试跑相同：25%、50%、75%、100% 四档，条数按 3、3、2、2，共 10 条。飞行力学进入填空的可变定义有 13 项：八个角度符号，以及地面系、机体系、稳定性坐标系、航迹系、气流系。一档改掉的项数是 13 乘以该档百分比后取整，即 3、6、9、13 项。欧拉顺序固定，不是可变定义。",
        "",
        "每条包含三个子过程。公共编号 1 到 32 是地面系到机体系，33 到 80 是地面系转航迹系和气流系，81 到 121 是体轴系到气流系。",
        "",
        "| 档 | 变更项 | 条数 | 编号 |",
        "| --- | --- | --- | --- |",
    ]
    for tier in TIERS:
        ids = [sample["id"] for sample in samples if sample["tier"] == tier]
        lines.append(f"| {tier}% | {_width(tier)}/13 | {len(ids)} | {'、'.join(ids)} |")
    lines.extend([
        "",
        f"抽样种子 {SEED}。数据在 data/flight/basic/pilot/samples.jsonl，每行一条，含定义声明和 103 个空的标准答案。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = _samples()
    path = OUT_DIR / "samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            declaration, questions, sections = _combine(sample)
            handle.write(json.dumps(_slim(sample, declaration, questions), ensure_ascii=False) + "\n")
            (OUT_DIR / f"sample_{sample['id']}.md").write_text(
                _markdown(sample, declaration, sections),
                encoding="utf-8",
            )
            print(f"wrote {sample['id']}: {sample['changed_count']}/13", flush=True)
    ids = [sample["id"] for sample in samples]
    (OUT_DIR / "pilot_10.json").write_text(
        json.dumps({"ids": ids, "counts": TIER_COUNTS, "seed": SEED, "factors": len(FACTORS)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "00_索引.md").write_text(_index(samples), encoding="utf-8")
    print(f"Wrote {len(samples)} samples, {BLANK_COUNT} blanks each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
