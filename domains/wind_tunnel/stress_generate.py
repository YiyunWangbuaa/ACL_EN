# -*- coding: utf-8 -*-
"""Ten strong perturbations. Every variable definition leaves the standard value.

Euler order is fixed by the angle mechanism and is not a definition, so it stays
alpha, then beta, then gamma. The three frames are left-handed, and both declared
axes of each frame lie on a different absolute axis from the standard frame.
The ten samples share that full reversal and differ in how the axes are assigned.

Run::

    python -m domains.wind_tunnel.stress_generate
"""

from __future__ import annotations

import json
from copy import deepcopy

from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    AXIS_ZH,
    DIRECTION_ZH,
    _format_document,
    _render_markdown,
    build_record,
)
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC

OUT_DIR = DOMAINS["wind_tunnel"]["stress"]
POINTS = {
    "axial": ("forward", "rearward"),
    "normal": ("up", "down"),
    "lateral": ("left", "right"),
}
# X leaves the axial axis. Y leaves the normal axis. Signs select the two pointings.
LAYOUTS = [
    ("normal", 0, "lateral", 0),
    ("normal", 1, "lateral", 1),
    ("normal", 0, "lateral", 1),
    ("normal", 1, "lateral", 0),
    ("normal", 0, "axial", 0),
    ("normal", 1, "axial", 1),
    ("normal", 0, "axial", 1),
    ("normal", 1, "axial", 0),
    ("lateral", 0, "axial", 0),
    ("lateral", 1, "axial", 1),
]
FRAME_SLOTS = (
    ("body_frame", "body", "XTZ", "YTZ", True),
    ("wind_frame", "tunnel", "XFD", "YFD", False),
    ("balance_frame", "balance", "XTP", "YTP", False),
)


def _frame(owner: str, x_name: str, y_name: str, with_order: bool, layout: tuple) -> dict:
    x_axis, x_sign, y_axis, y_sign = layout
    if x_axis == "axial" or y_axis == "normal" or x_axis == y_axis:
        raise RuntimeError(f"Layout {layout} still uses a standard absolute axis.")
    spec = {
        "chirality": "left",
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
    return spec


def _scalars() -> dict:
    spec = deepcopy(STANDARD_SPEC)
    spec["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma"] = {"right_wing_toward": "body_up", "sign": "positive"}
    spec["alpha_m"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta_m"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}
    spec["placement"] = "inverted"
    spec["alpha_e"] = {"axial_bends": "balance_down", "sign": "positive"}
    spec["beta_e"] = {"axial_bends": "balance_right", "sign": "positive"}
    spec["gamma_e"] = {"lateral_moves": "balance_up", "sign": "positive"}
    spec["alpha_v"] = {"airspeed_toward": "body_up", "sign": "positive"}
    spec["beta_v"] = {"airspeed_toward": "body_left", "sign": "positive"}
    spec["gamma_tp"] = {"sense": "opposite"}
    spec["gamma_model"] = {"sense": "opposite"}
    return spec


def _axis_phrase(frame: dict, name: str) -> str:
    axis = frame[name]
    return f"{name}沿{AXIS_ZH[axis['along']]}指向{DIRECTION_ZH[axis['points']]}"


def _check_against_standard(spec: dict) -> None:
    if set(spec) != set(STANDARD_SPEC):
        raise RuntimeError("The stress spec does not cover the same definitions as the standard spec.")
    for key, value in spec.items():
        if value == STANDARD_SPEC[key]:
            raise RuntimeError(f"{key} is still the standard definition.")
    for key, _owner, x_name, y_name, _order in FRAME_SLOTS:
        frame = spec[key]
        standard = STANDARD_SPEC[key]
        if frame["chirality"] == standard["chirality"]:
            raise RuntimeError(f"{key} chirality was not changed.")
        if frame[x_name]["along"] == standard[x_name]["along"]:
            raise RuntimeError(f"{key} {x_name} is still on the standard absolute axis.")
        if frame[y_name]["along"] == standard[y_name]["along"]:
            raise RuntimeError(f"{key} {y_name} is still on the standard absolute axis.")
        if "euler_order" in frame and frame["euler_order"] != ["alpha", "beta", "gamma"]:
            raise RuntimeError("Euler order is fixed and must stay alpha, beta, gamma.")


def _samples() -> list[dict]:
    samples = []
    shared = (
        "相对统一默认，每项可变定义都改离原值。"
        "理论迎角改为空速指向机体上方为正，理论侧滑改为空速指向机体左侧为正，理论滚转改为右翼指向机体上方为正。"
        "机构迎角、机构侧滑、机构滚转与上述理论角同向改反。"
        "安装改为反装。"
        "弹性迎角改为轴向弯向天平下方为正，弹性侧滑改为轴向弯向天平右侧为正，弹性滚转改为侧向移向天平上方为正。"
        "气流偏角 α_V 改为空速指向机体上方为正，β_V 改为空速指向机体左侧为正。"
        "天平安装角改为与绕天平绝对轴向正方向（前方）的右手定则相反，模型安装角改为与绕机体绝对轴向正方向（前方）的右手定则相反。"
        "体轴、风轴、天平都改为左手系，声明的第一轴离开轴向，第二轴离开法向。"
        "欧拉顺序由功角机构固定，不是可变定义。"
    )
    for index in range(10):
        spec = _scalars()
        phrases = []
        for offset, (key, owner, x_name, y_name, with_order) in zip((0, 3, 6), FRAME_SLOTS):
            spec[key] = _frame(owner, x_name, y_name, with_order, LAYOUTS[(index + offset) % 10])
            phrases.append(_axis_phrase(spec[key], x_name) + "，" + _axis_phrase(spec[key], y_name))
        _check_against_standard(spec)
        change = shared + "本条坐标轴：" + "；".join(phrases) + "。"
        samples.append({
            "id": f"{index + 1:02d}",
            "title": f"全定义反向·轴排列{index + 1}",
            "change": change,
            "spec": spec,
        })
    return samples


def _standard_answers() -> list[tuple[str, str]]:
    record = build_record({
        "id": "00",
        "title": "统一默认",
        "change": "统一默认。",
        "spec": deepcopy(STANDARD_SPEC),
    })
    return [(item["id"], item["answer"]) for item in record["questions"]]


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    standard = _standard_answers()
    records = []
    for sample in _samples():
        record = build_record(sample)
        answers = [(item["id"], item["answer"]) for item in record["questions"]]
        if len(answers) != len(standard):
            raise RuntimeError(f"{sample['id']} question count does not match the standard set.")
        changed = sum(left != right for left, right in zip(answers, standard))
        record["changed_answers"] = changed
        records.append(record)
        print(f"sample {sample['id']}: {changed}/{len(answers)} answers differ from the standard")
    stored = []
    for record in records:
        stored.append({
            key: value
            for key, value in record.items()
            if key not in ("injection_example", "changed_answers")
        })
    (OUT_DIR / "samples.json").write_text(
        json.dumps(stored, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    index = [
        "# 10条强扰动数据",
        "",
        "每条都把全部可变定义改离统一默认，用来排查定义组合下的程序错误。欧拉顺序由功角机构固定，不在扰动之内。单因素校核仍看 data/wind_tunnel/review，不要和这里混用。",
        "",
        "正式注入格式见 00_注入格式.md。每条审阅稿给出定义声明和四个子过程的完整正文。",
        "",
    ]
    for record in records:
        path = OUT_DIR / f"sample_{record['id']}.md"
        text = _render_markdown(record).replace(
            "data/wind_tunnel/review/00_注入格式.md",
            "data/wind_tunnel/stress/00_注入格式.md",
        )
        path.write_text(text, encoding="utf-8")
        index.append(
            f"- 数据 {record['id']} {record['title']}："
            f"{record['changed_answers']} 个空的答案与统一默认不同。"
        )
    (OUT_DIR / "00_索引.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    (OUT_DIR / "00_注入格式.md").write_text(
        _format_document(records[0]["injection_example"]),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} samples, {len(records[0]['questions'])} questions each, to {OUT_DIR}")


if __name__ == "__main__":
    main()
