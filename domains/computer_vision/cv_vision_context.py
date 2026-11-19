# -*- coding: utf-8 -*-
"""同构出纯机器视觉语境。题序、标签、标准答案与当前机器视觉数据一一对应，只改说法。

风轴系 → 世界坐标系
机体系 → 物体系
空速 → 世界坐标系中的参考方向
激波发生位置 → 物体上的参考点
"""
from __future__ import annotations

import json

from domain_layout import SETS

_PINHOLE = "@@PINHOLE@@"
_STEREO = "@@STEREO@@"

_SCENE_1 = (
    "初始位置时，飞机模型平正处于风洞中，模型前方朝向来流方向。"
    "两台已标定、同步曝光的相机从不同方位拍摄纹影图。"
    "相机光轴沿相机轴向，像面横向沿相机侧向，像面纵向沿相机法向。"
    "飞机尾部固定在了一个机械结构装置上，该装置会采取一种全新的欧拉角变换方式将机体变换到一种特殊姿态。"
    "变换的顺序与正常变换顺序不同，必须先理解机构角度的变换特性，然后推断出正确的角度变换顺序。"
    "这些角度的定义方式已在下方的角度定义给出。"
)
_SCENE_1_VISION = (
    "初始位置时，物体平正处于世界坐标系中，物体前方朝向世界前向。"
    "两台已标定、同步曝光的相机从不同方位拍摄该物体。"
    "相机光轴沿相机轴向，像面横向沿相机侧向，像面纵向沿相机法向。"
    "物体后端固定在了一个机械结构装置上，该装置会采取一种全新的欧拉角变换方式将物体变换到一种特殊姿态。"
    "变换的顺序与正常变换顺序不同，必须先理解姿态角的变换特性，然后推断出正确的角度变换顺序。"
    "这些角度的定义方式已在下方的角度定义给出。"
)
_SCENE_2 = (
    "其中有两个特殊任务，其一是将风洞中的模型纵向轴线与法向轴线投影到相机图像中，确定来两轴线在图像坐标系中的表达式。"
    "其二是给定一个相机图像中一处激波发生位置的坐标。根据双目相机模型，反推激波发生位置在实际机器系中的坐标。"
    "其中涉及到的子任务包括，根据机体系与风轴系的姿态角，将机体的纵轴和法向轴转换到风轴系中表达；"
    "根据相机坐标系与风轴系的位置关系，推导风轴系向相机坐标系的转换矩阵，并据此进一步推导两个双目相机之间的本质矩阵和基本矩阵；"
    "根据风轴系汇总机体两轴的表达式与双目相机模型，计算图像坐标系中，两轴的表达式；"
    "根据激波在一个相机图像中的坐标位置，重构激波发生位置在机体坐标系中的坐标。"
)
_SCENE_2_VISION = (
    "其中有两个特殊任务，物体在世界坐标系中存在一定的放置姿态。其一是将世界坐标系中的物体纵轴与法向轴投影到相机图像中，确定这两条轴线在图像坐标系中的表达式。"
    "其二是给定一个相机图像中物体上一处参考点的坐标。根据双目相机模型，反推该参考点在物体系中的坐标。"
    "其中涉及到的子任务包括，根据物体系与世界坐标系的姿态角，将物体的纵轴和法向轴转换到世界坐标系中表达；"
    "根据相机坐标系与世界坐标系的位置关系，推导世界坐标系向相机坐标系的转换矩阵，并据此进一步推导两个双目相机之间的本质矩阵和基本矩阵；"
    "根据世界坐标系中物体两轴的表达式与双目相机模型，计算图像坐标系中两轴的表达式；"
    "根据参考点在一个相机图像中的坐标位置，重构该参考点在物体系中的坐标。"
)

# Longer phrases first. 针孔模型、双目相机模型在替换前先占位。
_PAIRS = [
    ("为支撑杆（机体轴向）与水平面的夹角。机体轴向偏向风洞",
     "为物体纵轴与水平面的夹角。物体纵轴偏向风洞"),
    ("β_M是侧滑角支撑杆与铅垂面的夹角。使机体轴向偏向风洞",
     "该角是物体纵轴与铅垂面的夹角。使物体纵轴偏向风洞"),
    ("绕侧滑角支撑杆的转角", "绕侧向机构的转角"),
    ("风轴系与世界坐标系等价，以下一律在风轴系中描述。", "以下一律在世界坐标系中描述。"),
    ("体轴系原点在机头；模型顶点到支杆底座的距离为 L。",
     "物体系原点在物体前端；物体前端到世界坐标系原点的距离为 L。"),
    ("相机光心在风轴系中的坐标为：", "相机光心在世界坐标系中的坐标为："),
    ("功角支撑杆在铅垂面上运动，可视为绕", "纵轴机构在铅垂面上运动，可视为绕"),
    ("侧滑角支撑杆在垂直于铅垂面且过功角支撑杆的平面上运动，即可视为侧滑角机构基于功角机构在作动。",
     "侧向机构在垂直于铅垂面且过纵轴机构的平面上运动，即可视为侧向机构基于纵轴机构在作动。"),
    ("动作机构产生了", "姿态变化产生了"),
    ("三个机构角", "三个姿态角"),
    ("机构迎角", "机构俯仰角"),
    ("机构侧滑角", "机构方位角"),
    ("根据机构结构", "根据姿态定义"),
    ("对应机构角", "对应姿态角"),
    ("机构角", "姿态角"),
    ("模型纵向轴线", "物体纵轴"),
    ("模型纵轴", "物体纵轴"),
    ("模型顶点", "物体前端"),
    ("模型平正", "物体平正"),
    ("模型安装方式", "物体安置方式"),
    ("模型只考虑正装", "物体只考虑正装"),
    ("模型的绝对", "物体的绝对"),
    ("模型前方", "物体前方"),
    ("风洞中的模型", "场景中的物体"),
    ("来流方向", "世界前向"),
    ("前方是来流。", "前方为世界前向。"),
    ("激波发生位置", "物体参考点"),
    ("激波点", "物体参考点"),
    ("激波", "物体参考点"),
    ("纹影图", "图像"),
    ("实际机器系", "物体系"),
    ("体轴系", "物体系"),
    ("风轴系", "世界坐标系"),
    ("机体系", "物体系"),
    ("风洞", "世界"),
    ("风轴", "世界"),
    ("机体", "物体"),
    ("机头", "物体前端"),
    ("机尾", "物体后端"),
    ("右翼", "物体右缘"),
    ("空速", "参考方向"),
    ("模型", "物体"),
]


def rewrite(text: str) -> str:
    if not text:
        return text
    text = text.replace(_SCENE_1, _SCENE_1_VISION).replace(_SCENE_2, _SCENE_2_VISION)
    text = text.replace("针孔模型", _PINHOLE).replace("双目相机模型", _STEREO)
    for old, new in _PAIRS:
        text = text.replace(old, new)
    text = text.replace(_PINHOLE, "针孔模型").replace(_STEREO, "双目相机模型")
    text = text.replace(
        "坐标轴按地面系的方式建立。沿轴向、指向前方的那根坐标轴是参考轴，相当于地面坐标轴 Xg。另外两根坐标轴在法向与侧向之间绑定，绑定结果必须符合该系的手性。\n",
        "\n",
    )
    text = text.replace("单独变换迎角（俯仰）时", "单独变换俯仰角时")
    text = text.replace("单独变换侧滑角（偏航）时", "单独变换方位角时")
    text = text.replace("世界轴线", "世界前后轴")
    text = text.replace("参考方向偏物体的", "参考方向偏向物体的")
    text = text.replace(
        "- 轴向：世界前后轴。正方向朝前，前方为世界前向。后方是其反方向。",
        "- 轴向：世界前后轴。水平面内，正方向朝前，前方为世界前向。后方是其反方向。",
    )
    text = text.replace(
        "- 侧向：世界水平横轴。正方向朝右。左侧是其反方向。",
        "- 侧向：世界水平横轴。水平面内，正方向朝右。左侧是其反方向。",
    )
    # 重构数据里物体姿态角改用 θ_M、ψ_M、φ_M。相机角仍是 θ1、ψ1、φ1。
    text = text.replace("α_M", "θ_M").replace("β_M", "ψ_M").replace("γ_M", "φ_M")
    return text


def apply_record(row: dict) -> dict:
    out = dict(row)
    out["title"] = rewrite(row["title"]) + "（纯视觉语境）"
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
        copied["process"] = "纯机器视觉双目姿态估计"
        questions.append(copied)
    out["questions"] = questions
    return out


def leftover(text: str) -> list[str]:
    words = ["风轴", "风洞", "机体", "空速", "激波", "机头", "机尾", "右翼", "纹影", "来流", "支撑杆", "天平"]
    return [word for word in words if word in text]


def export_review(rows: list[dict]) -> None:
    from domains.computer_vision.cv_dataset import _injection
    from domains.computer_vision import review_export as rx

    out = SETS["reconstruction.vision"]["review"]
    out.mkdir(exist_ok=True)
    intro = (
        "本文件是纯机器视觉语境的同构审阅稿。题序、标签和标准答案与校正后的机器视觉数据一一对应。"
        "槽里的标签和标准答案只给审阅用，模型看到的待填槽不含这两项。"
    )
    titles = dict(rx.SEGMENT_TITLE)
    titles["body_attitude"] = "物体姿态"
    titles["projection"] = "成像与物体前端投影"
    for index, row in enumerate(rows, start=1):
        (out / f"sample_{index:02d}.md").write_text(
            rx._sample_markdown(row, index, intro=intro, titles=titles), encoding="utf-8")
    first = rows[0]
    by_id = {item["id"]: item for item in first["questions"]}
    example = _injection(first["declaration"], by_id, "5")
    (out / "00_注入格式.md").write_text(rx._format_document(example), encoding="utf-8")
    lines = [
        "# 纯机器视觉审阅",
        "",
        f"这 10 条与校正后的机器视觉数据同构：{len(rows[0]['questions'])} 个空，标签和标准答案不变。世界坐标系对应原风轴系，物体系对应原机体系，参考方向对应原空速，物体参考点对应原激波发生位置。",
        "",
        "正式输入格式见 00_注入格式.md。端到端提问见 [端到端](端到端.md)。改题面时直接改对应样本里的槽。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(f"- [数据 {index:02d}](sample_{index:02d}.md)：{row['id']}，{row['change']}")
    lines.append("")
    (out / "00_索引.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")


def main() -> None:
    source = SETS["cross.wind_vision"]["samples"]
    target = SETS["reconstruction.vision"]["samples"]
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    written = []
    for row in rows:
        vision = apply_record(row)
        if len(vision["questions"]) != len(row["questions"]):
            raise RuntimeError(f"{row['id']} 题数变化")
        for old, new in zip(row["questions"], vision["questions"]):
            if old["label"] != new["label"] or old["id"] != new["id"]:
                raise RuntimeError(f"{row['id']} 空 {old['id']} 的标签变了")
            if len(old.get("options") or []) != len(new.get("options") or []):
                raise RuntimeError(f"{row['id']} 空 {old['id']} 的选项数变了")
            renamed = old["answer"].replace("α_M", "θ_M").replace("β_M", "ψ_M").replace("γ_M", "φ_M")
            if new["answer"] != renamed:
                raise RuntimeError(f"{row['id']} 空 {old['id']} 的答案变了")
            if old["answer"] in set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") and new["answer"] != old["answer"]:
                raise RuntimeError(f"{row['id']} 空 {old['id']} 的选项字母变了")
        blob = vision["declaration"] + vision["change"] + vision["title"]
        for item in vision["questions"]:
            blob += item["prompt"] + item.get("lead", "") + item.get("tail", "")
            blob += "".join(item.get("options") or [])
        bad = leftover(blob)
        if bad:
            raise RuntimeError(f"{row['id']} 仍有风洞用语：{bad}")
        written.append(vision)
        print(f"ok {row['id']}", flush=True)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in written), encoding="utf-8")
    print(f"wrote {target}")
    export_review(written)


if __name__ == "__main__":
    main()
