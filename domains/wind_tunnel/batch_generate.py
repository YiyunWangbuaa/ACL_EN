# -*- coding: utf-8 -*-
"""100 declaration samples in four change tiers: 25%, 50%, 75%, 100%.

A definition is one of 16 factors: 13 binary declarations that enter the cloze,
plus the body, wind, and balance frames. gamma_model is declared but no blank
reads it, so it stays at the standard value and is not one of the 16.
Euler order is fixed and is not a definition.

Each tier has 25 samples. A tier of k percent changes exactly 16*k/100 factors
away from the standard declaration. A changed frame is any of the 47 legal
frames other than the standard one.

Run::

    python -m domains.wind_tunnel.batch_generate
"""

from __future__ import annotations

import itertools
import json
import random
from copy import deepcopy

from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import AXIS_ZH, DIRECTION_ZH, HAND_ZH, build_record
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

OUT_DIR = DOMAINS["wind_tunnel"]["batch"]
SEED = 20261102
PER_TIER = 25
TIERS = (25, 50, 75, 100)
PILOT_COUNTS = {25: 3, 50: 3, 75: 2, 100: 2}
PILOT_SEED = 20261103

SCALAR_FLIP = {
    "alpha": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma": {"right_wing_toward": "body_up", "sign": "positive"},
    "alpha_m": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta_m": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma_m": {"right_wing_toward": "body_up", "sign": "positive"},
    "placement": "inverted",
    "alpha_e": {"axial_bends": "balance_down", "sign": "positive"},
    "beta_e": {"axial_bends": "balance_right", "sign": "positive"},
    "gamma_e": {"lateral_moves": "balance_up", "sign": "positive"},
    "alpha_v": {"airspeed_toward": "body_up", "sign": "positive"},
    "beta_v": {"airspeed_toward": "body_left", "sign": "positive"},
    "gamma_tp": {"sense": "opposite"},
}
SCALAR_ZH = {
    "alpha": "理论迎角改为空速指向机体上方为正",
    "beta": "理论侧滑角改为空速指向机体左侧为正",
    "gamma": "理论滚转角改为右翼指向机体上方为正",
    "alpha_m": "机构迎角改为天平偏向风洞下方为正",
    "beta_m": "机构侧滑角改为天平偏向风洞右侧为正",
    "gamma_m": "机构滚转改为与绕模型绝对轴向正方向（前方）的右手定则相反",
    "placement": "安装改为反装",
    "alpha_e": "弹性迎角改为轴向弯向天平下方为正",
    "beta_e": "弹性侧滑改为轴向弯向天平右侧为正",
    "gamma_e": "弹性滚转改为侧向移向天平上方为正",
    "alpha_v": "气流偏角α_V改为空速指向机体上方为正",
    "beta_v": "气流偏角β_V改为空速指向机体左侧为正",
    "gamma_tp": "天平安装角改为与绕天平绝对轴向正方向（前方）的右手定则相反",
}
FRAMES = (
    ("body_frame", "body", "XTZ", "YTZ", True, "体轴系"),
    ("wind_frame", "tunnel", "XFD", "YFD", False, "风轴系"),
    ("balance_frame", "balance", "XTP", "YTP", False, "天平系"),
)
FRAME_KEYS = [item[0] for item in FRAMES]
FACTORS = list(SCALAR_FLIP) + FRAME_KEYS
POINTS = {
    "axial": ("forward", "rearward"),
    "normal": ("up", "down"),
    "lateral": ("left", "right"),
}
AXES = ("axial", "normal", "lateral")


def _layouts(owner: str, x_name: str, y_name: str, with_order: bool, standard: dict) -> list[dict]:
    found = []
    for chirality in ("right", "left"):
        for x_axis in AXES:
            for x_sign in (0, 1):
                for y_axis in AXES:
                    if y_axis == x_axis:
                        continue
                    for y_sign in (0, 1):
                        spec = {
                            "chirality": chirality,
                            x_name: {
                                "along": f"{owner}_{x_axis}",
                                "points": f"{owner}_{POINTS[x_axis][x_sign]}",
                            },
                            y_name: {
                                "along": f"{owner}_{y_axis}",
                                "points": f"{owner}_{POINTS[y_axis][y_sign]}",
                            },
                        }
                        if with_order:
                            spec["euler_order"] = ["alpha", "beta", "gamma"]
                        if spec != standard:
                            found.append(spec)
    if len(found) != 47:
        raise RuntimeError(f"{owner} has {len(found)} non-standard frames, expected 47.")
    return found


def _frame_phrase(title: str, spec: dict, x_name: str, y_name: str) -> str:
    x_axis = spec[x_name]
    y_axis = spec[y_name]
    return (
        f"{title}改为{HAND_ZH[spec['chirality']]}，"
        f"{x_name}沿{AXIS_ZH[x_axis['along']]}指向{DIRECTION_ZH[x_axis['points']]}，"
        f"{y_name}沿{AXIS_ZH[y_axis['along']]}指向{DIRECTION_ZH[y_axis['points']]}"
    )


def _change_text(tier: int, changed: list[str], spec: dict) -> str:
    phrases = []
    for key in changed:
        if key in SCALAR_ZH:
            phrases.append(SCALAR_ZH[key])
        else:
            _key, _owner, x_name, y_name, _order, title = next(item for item in FRAMES if item[0] == key)
            phrases.append(_frame_phrase(title, spec[key], x_name, y_name))
    kept = "模型安装角γ_model保持与绕机体绝对轴向正方向（前方）的右手定则相同，不计入这16项。"
    return (
        f"定义变更档{tier}%：16项可变定义中变更{len(changed)}项。"
        + "；".join(phrases)
        + f"。其余计入的可变定义保持统一默认。{kept}欧拉顺序由功角机构固定，不是可变定义。"
    )


def _samples() -> list[dict]:
    if len(FACTORS) != 16:
        raise RuntimeError(f"Expected 16 factors, found {len(FACTORS)}.")
    layouts = {}
    for key, owner, x_name, y_name, with_order, _title in FRAMES:
        layouts[key] = _layouts(owner, x_name, y_name, with_order, STANDARD_SPEC[key])
    rng = random.Random(SEED)
    seen = set()
    samples = []
    for tier in TIERS:
        width = 16 * tier // 100
        pool = list(itertools.combinations(FACTORS, width))
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
                if cursor > len(pool) * 20:
                    raise RuntimeError(f"Could not draw a unique spec for tier {tier}.")
                continue
            changed = [key for key in FACTORS if spec[key] != STANDARD_SPEC[key]]
            if changed != chosen and set(changed) != set(chosen):
                raise RuntimeError(f"Tier {tier} changed {changed}, expected {chosen}.")
            if spec["gamma_model"] != STANDARD_SPEC["gamma_model"]:
                raise RuntimeError("gamma_model must stay at the standard value.")
            seen.add(signature)
            made += 1
            sample_id = f"{tier:03d}-{made:03d}"
            samples.append({
                "id": sample_id,
                "tier": tier,
                "changed_count": len(chosen),
                "changed": chosen,
                "title": f"定义变更{tier}%",
                "change": _change_text(tier, chosen, spec),
                "spec": spec,
            })
    return samples


def _slim(sample: dict, record: dict) -> dict:
    questions = []
    for item in record["questions"]:
        questions.append({
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
        "declaration": record["declaration"],
        "questions": questions,
    }


def _pilot(samples: list[dict]) -> list[str]:
    rng = random.Random(PILOT_SEED)
    picked = []
    for tier in TIERS:
        ids = [sample["id"] for sample in samples if sample["tier"] == tier]
        picked.extend(rng.sample(ids, PILOT_COUNTS[tier]))
    return picked


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    samples = _samples()
    path = OUT_DIR / "samples.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, sample in enumerate(samples, start=1):
            record = build_record(sample)
            if len(record["questions"]) != 200:
                raise RuntimeError(f"{sample['id']} has {len(record['questions'])} questions.")
            handle.write(json.dumps(_slim(sample, record), ensure_ascii=False) + "\n")
            if index == 1 or index % 50 == 0:
                print(f"wrote {index}/{len(samples)}", flush=True)
    pilot = _pilot(samples)
    (OUT_DIR / "pilot_10.json").write_text(
        json.dumps({"ids": pilot, "counts": PILOT_COUNTS, "seed": PILOT_SEED}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# 100条定义变更数据",
        "",
        "相对统一默认，按变更项数分成四档，每档 25 条。16 项可变定义是：理论迎角、理论侧滑、理论滚转、机构迎角、机构侧滑、机构滚转、安装方式、弹性迎角、弹性侧滑、弹性滚转、气流偏角 α_V、气流偏角 β_V、天平安装角，以及体轴系、风轴系、天平系。",
        "",
        "一项坐标系算作已变更，是指它不是统一默认的那一套合法坐标轴。每套坐标系有 48 种合法声明，变更时从其余 47 种里抽取。模型安装角 γ_model 不进入任何一个空，四档都保持默认。欧拉顺序由功角机构固定，不是可变定义。",
        "",
        "| 档 | 变更项 | 条数 | 编号 |",
        "| --- | --- | --- | --- |",
        "| 25% | 4/16 | 25 | 025-001 至 025-025 |",
        "| 50% | 8/16 | 25 | 050-001 至 050-025 |",
        "| 75% | 12/16 | 25 | 075-001 至 075-025 |",
        "| 100% | 16/16 | 25 | 100-001 至 100-025 |",
        "",
        "抽样种子 20261102。25%、50%、75% 三档的变更组合互不重复；100% 只有一种组合，25 条靠三套坐标系的不同声明区分。被变更的坐标系从 47 种非默认声明里抽取。数据在 samples.jsonl，每行一条，含定义声明和 200 个空的标准答案。",
        "",
        "本地试跑的分层样本在 pilot_10.json。四档按 3、3、2、2 抽取，共 10 条：" + "、".join(pilot) + "。",
        "",
    ]
    (OUT_DIR / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {OUT_DIR}")
    print("Pilot: " + ", ".join(pilot))


if __name__ == "__main__":
    main()
