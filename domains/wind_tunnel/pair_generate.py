# -*- coding: utf-8 -*-
"""Single-factor contract pairs for the wind-tunnel and flight-mechanics set.

Five abnormal definitions. Each one is written three ways:

- unmarked: the abnormal rule stays in the declaration, once, with no extra warning,
  and the rule questions of every definition are left out;
- marked: the same text as unmarked, plus one warning in front of the changed rule;
- rule: the unmarked declaration, with every definition's rule questions put back.

Unmarked and marked are the contrast. Both drop every definition's rule question,
including on the standard-definition manuscripts. Rule-first puts those questions back.

The spherical-wrist order is the standard side of the order pair. The current
angle-rod description is the abnormal side. The other four abnormal definitions
are paired with the unified default.

Run::

    python -m domains.wind_tunnel.pair_generate
"""

from __future__ import annotations

import json
import re
from copy import deepcopy

from body_frame_model import BodyFrameModel
from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    PRIOR,
    PROCESS_OFFSET,
    PROCESSES,
    ROD_ORDER_PARAGRAPH,
    SLOT_MARK,
    WRIST_ORDER_PARAGRAPH,
    _check_declaration,
    _declaration,
    _mechanism_phrases,
    _process_template,
    _question,
    _render_markdown,
    _render_process,
    _sentence_groups,
    build_record,
)
from domains.wind_tunnel.tz_fd_generate import STANDARD_SPEC


OUT_DIR = DOMAINS["wind_tunnel"]["pair"]
TZ = "风轴系到体轴系"
BODY = "天平系到体轴系"
WIND = "天平系到风轴系"
HEADING = re.compile(r"(?m)^(\d+)\.")
_CACHE: dict[str, dict[str, list[dict]]] = {}


def _alpha(spec: dict) -> None:
    spec["alpha"] = {"airspeed_toward": "body_up", "sign": "positive"}


def _roll(spec: dict) -> None:
    spec["gamma_m"] = {"right_wing_toward": "body_up", "sign": "positive"}


def _mount(spec: dict) -> None:
    spec["placement"] = "inverted"


def _body(spec: dict) -> None:
    spec["body_frame"] = {
        "chirality": "left",
        "euler_order": ["alpha", "beta", "gamma"],
        "XTZ": {"along": "body_axial", "points": "body_forward"},
        "YTZ": {"along": "body_normal", "points": "body_up"},
    }


def _wrist(spec: dict) -> None:
    frame = deepcopy(spec["body_frame"])
    frame["euler_order"] = ["beta", "alpha", "gamma"]
    spec["body_frame"] = frame


def _close_order_gap(template: str) -> str:
    """After deleting section 3, number the later sections 3, 4, 5, ..."""
    def repl(match: re.Match) -> str:
        number = int(match.group(1))
        if number < 4:
            return match.group(0)
        return f"{number - 1}."

    return HEADING.sub(repl, template)


MARK_ALPHA = (
    "本题特别约定：以下理论迎角的正负方向可能不同于常见约定。"
    "后续所有步骤必须严格按照本题给出的定义执行，不得使用默认约定替代。"
)
MARK_ROLL = (
    "本题特别约定：以下机构滚转角正方向可能不同于常见右手定则。"
    "后续所有步骤必须严格按照本题给出的定义执行，不得使用默认约定替代。"
)
MARK_ORDER = (
    "本题特别约定：以下机构姿态角的执行顺序由本题的机构结构决定，可能不同于常见的偏航、俯仰、滚转顺序。"
    "后续所有步骤必须严格按照本题给出的机构描述执行，不得使用默认顺序替代。"
)
MARK_MOUNT = (
    "本题特别约定：以下模型安装方式可能不同于常见正装。"
    "后续所有步骤必须严格按照本题给出的安装方式执行，不得使用默认安装替代。"
)
MARK_BODY = (
    "本题特别约定：以下体轴系可能不同于常见右手体轴系。"
    "后续所有步骤必须严格按照本题给出的体轴系定义执行，不得使用默认约定替代。"
)
MARK_ALL = (
    "本题每一条定义都以此处声明为准。"
    "后续所有步骤必须严格按照本题给出的全部定义执行，不得使用默认约定替代。"
)

# Each edit is (process, old, new). old must occur once in that process template.
_ALPHA_EDITS = (
    (TZ, "理论迎角>0时，空速偏机体的@@21@@。", ""),
    (
        WIND,
        "正理论迎角时，空速指向机体的@@39@@。"
        "在该空速方向和风轴系轴向、法向、侧向都已确定后，Δα_e应使用的风轴分量为@@33@@。",
        "Δα_e应使用的风轴分量为@@33@@。",
    ),
)
_ROLL_EDITS = (
    (
        TZ,
        "当γ_M>0时，以模型绝对轴向正方向（前方）为旋转轴，与右手定则@@35@@。此时右翼偏机体的@@36@@。",
        "当γ_M>0时，右翼偏机体的@@36@@。",
    ),
    (
        BODY,
        "天平安装角以天平绝对轴向正方向（前方）的右手定则@@2@@为正。"
        "机体滚转角以机体绝对轴向正方向（前方）的右手定则@@3@@为正。"
        "此时天平绝对轴向与机体绝对轴向的指向可能相同，也可能相反。"
        "机体滚转角γ_M的绝对值远大于天平安装角γ_tp。"
        "在天平绝对轴向、机体绝对轴向和两个正方向都已确定的条件下，天平系与机体系相差的滚转角度为@@4@@。",
        "天平安装角以天平绝对轴向正方向（前方）的右手定则@@2@@为正。"
        "此时天平绝对轴向与机体绝对轴向的指向可能相同，也可能相反。"
        "机体滚转角γ_M的绝对值远大于天平安装角γ_tp。"
        "在天平绝对轴向、机体绝对轴向和两个正方向都已确定的条件下，根据定义声明，天平系与机体系相差的滚转角度为@@4@@。",
    ),
)
_ORDER_EDITS = (
    (TZ, ROD_ORDER_PARAGRAPH, ""),
)
_MOUNT_EDITS = (
    (
        TZ,
        "此外，模型安装方式为@@11@@。"
        "此时模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞的绝对法向正方向和绝对侧向正方向@@12@@，对应的前置变换矩阵为RP2。",
        "模型的绝对法向正方向（上方）和绝对侧向正方向（右侧）与风洞的绝对法向正方向和绝对侧向正方向@@12@@，对应的前置变换矩阵为RP2。",
    ),
    (TZ, "模型采用@@19@@。", ""),
    (TZ, "模型采用@@27@@。", ""),
    (
        BODY,
        "在该定义下，模型是@@15@@的。因此该定义下的前置矩阵为RTP_TZ2。前置矩阵RTP_TZ2是@@16@@。",
        "因此前置矩阵为RTP_TZ2。前置矩阵RTP_TZ2是@@16@@。",
    ),
    (
        WIND,
        "当产生形变α_e、β_e、γ_e时，模型是@@35@@的。产生这些形变后，空速位于模型的@@36@@。",
        "当产生α_e、β_e、γ_e形变后，空速位于模型的@@36@@。",
    ),
)
_BODY_EDITS = (
    (
        TZ,
        "机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@5@@、@@6@@、@@7@@。"
        "所以有风轴系向体轴系平正位置处的前置变换矩阵为RP1。前置变换矩阵RP1为@@8@@。",
        "按定义声明把机体系坐标轴与风轴系统一后，风轴系向体轴系平正位置处的前置变换矩阵为RP1。前置变换矩阵RP1为@@8@@。",
    ),
    (
        BODY,
        "机体系的绝对法向正方向（上方）、绝对轴向正方向（前方）、绝对侧向正方向（右侧）分别对应@@11@@、@@12@@、@@13@@。"
        "所以有天平系向体轴系的前置矩阵为RTP_TZ1。前置矩阵RTP_TZ1是@@14@@。",
        "按定义声明把机体系坐标轴与天平系统一后，天平系向体轴系的前置矩阵为RTP_TZ1。前置矩阵RTP_TZ1是@@14@@。",
    ),
)

FACTORS = (
    {
        "key": "alpha",
        "title": "理论迎角正负",
        "change": "只把理论迎角改为空速投影在机体轴向上方时为正。机构迎角、坐标轴、安装和旋转顺序都不动。",
        "apply": _alpha,
        "standard_id": "standard-default",
        "anchor": "理论迎角 α：",
        "marker": MARK_ALPHA,
        "edits": _ALPHA_EDITS,
        "removed": {TZ: {"21"}, WIND: {"39"}},
        "kept_rule": ("35", "11"),
    },
    {
        "key": "roll",
        "title": "机构滚转方向",
        "change": "只把机构滚转角 γ_M 改为与绕模型绝对轴向正方向（前方）的右手定则相反。理论滚转角仍与右手定则相同。",
        "apply": _roll,
        "standard_id": "standard-default",
        "anchor": "机构滚转角 γ_M：",
        "marker": MARK_ROLL,
        "edits": _ROLL_EDITS,
        "removed": {TZ: {"35"}, BODY: {"3"}},
        "kept_rule": ("21", "37"),
    },
    {
        "key": "order",
        "title": "机构角顺序",
        "change": "反常侧沿用当前功角机构：顺序由支撑杆结构决定，为迎角、侧滑、滚转。标准侧改为三自由度动轴球腕，顺序规定为偏航、俯仰、滚转。",
        "apply": None,
        "standard_apply": _wrist,
        "standard_id": "standard-order",
        "anchor": "机构迎角 α_M：",
        "marker": MARK_ORDER,
        "edits": _ORDER_EDITS,
        "after": _close_order_gap,
        "removed": {TZ: {"16", "17", "18"}},
        "kept_rule": ("11", "35"),
    },
    {
        "key": "mount",
        "title": "正装反装",
        "change": "只把模型安装改为反装。坐标系手性、角度正方向和旋转顺序都不动。",
        "apply": _mount,
        "standard_id": "standard-default",
        "anchor": "- 模型安装：",
        "marker": MARK_MOUNT,
        "edits": _MOUNT_EDITS,
        "removed": {TZ: {"11", "19", "27"}, BODY: {"15"}, WIND: {"35"}},
        "kept_rule": ("21", "35"),
    },
    {
        "key": "body",
        "title": "体轴系手性",
        "change": "只把体轴系改为左手系。XTZ 仍沿机体轴向指向前方，YTZ 仍沿机体法向指向上方，ZTZ 由手性推断。",
        "apply": _body,
        "standard_id": "standard-default",
        "anchor": "体轴系（",
        "marker": MARK_BODY,
        "edits": _BODY_EDITS,
        "removed": {TZ: {"5", "6", "7"}, BODY: {"11", "12", "13"}},
        "kept_rule": ("2", "11", "35"),
    },
)


def _merge_removed(*parts: dict[str, set[str]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = {}
    for part in parts:
        for process, ids in part.items():
            merged.setdefault(process, set()).update(ids)
    return merged


# Unmarked and marked drop this set on every definition, including both standards.
# It is the union of the five definitions' rule questions.
EMPHASIS_EDITS = _ALPHA_EDITS + _ROLL_EDITS + _MOUNT_EDITS + _BODY_EDITS
EMPHASIS_REMOVED = _merge_removed(*(factor["removed"] for factor in FACTORS))


def _payloads(spec: dict) -> dict[str, list[dict]]:
    key = json.dumps(spec, ensure_ascii=False, sort_keys=True)
    if key not in _CACHE:
        found = {}
        for process, generate in PROCESSES:
            found[process] = generate(deepcopy(spec))["items"]
        _CACHE[key] = found
    return _CACHE[key]


def _raw(spec: dict, process: str, local_id: str) -> str:
    for item in _payloads(spec)[process]:
        if item["manuscript_id"] == local_id:
            return item["answer"]
    raise KeyError(f"{process} 没有原稿编号 {local_id}")


def _emphasis_public_ids() -> set[str]:
    found = set()
    for process, local_ids in EMPHASIS_REMOVED.items():
        offset = PROCESS_OFFSET[process]
        found.update(str(int(local_id) + offset) for local_id in local_ids)
    return found


def _strip_order_paragraph(template: str) -> str:
    """Delete the angle-order questions, whichever mechanism paragraph is present."""
    for paragraph in (ROD_ORDER_PARAGRAPH, WRIST_ORDER_PARAGRAPH):
        if paragraph not in template:
            continue
        if template.count(paragraph) != 1:
            raise RuntimeError("机构顺序段出现多次。")
        return template.replace(paragraph, "", 1)
    raise RuntimeError("没有可删除的机构顺序段。")


def _apply_edits(template: str, edits: tuple, process: str) -> str:
    for owner, old, new in edits:
        if owner != process:
            continue
        found = template.count(old)
        if found != 1:
            raise RuntimeError(f"{process} 中替换目标出现 {found} 次：{old[:48]}")
        template = template.replace(old, new, 1)
    return template


def _public_template(process: str, template: str) -> str:
    offset = PROCESS_OFFSET[process]
    return SLOT_MARK.sub(lambda match: f"@@{int(match.group(1)) + offset}@@", template)


def _check_sentences(process: str, template: str, phrases: dict[str, str], by_local: dict[str, dict]) -> None:
    for sentence in _sentence_groups(process, phrases, template):
        kinds = [by_local[local_id]["label"] for local_id in sentence]
        families = {kind[0] for kind in kinds}
        if "A" in families and "C" in families:
            raise RuntimeError(f"{process} 把 A 与 C 放在同一句：{list(zip(sentence, kinds))}")
        if any(kind.startswith("A") for kind in kinds) and len(sentence) != 1:
            raise RuntimeError(f"{process} 的 A 类空没有单独成句：{list(zip(sentence, kinds))}")


def _insert_marker(declaration: str, anchor: str, marker: str) -> str:
    if declaration.count(anchor) != 1:
        raise RuntimeError(f"标记锚点出现 {declaration.count(anchor)} 次：{anchor}")
    return declaration.replace(anchor, marker + "\n" + anchor, 1)


def _note(title: str, condition: str, excluded: list[str]) -> str:
    if condition == "standard":
        return (
            f"审阅说明：{title}。这一句和文件标题不进入模型输入。"
            "模型只看到下面的定义声明、物理常识和过程。本条是配对的标准侧，规则询问全部保留。"
        )
    names = {"unmarked": "无标记", "marked": "有标记", "rule": "先问规则"}
    text = (
        f"审阅说明：{title}，条件为{names[condition]}。这一句和文件标题不进入模型输入。"
        "模型只看到下面的定义声明、物理常识和过程。"
    )
    if condition == "rule":
        return text + "本条保留全部定义的规则询问，定义声明与无标记稿相同，没有额外提醒。"
    if excluded:
        return text + "本条去掉全部定义的规则询问：" + "、".join(excluded) + "。"
    return text


def _make(
    spec: dict,
    sample_id: str,
    title: str,
    change: str,
    factor: str,
    condition: str,
    standard_id: str,
    edits: tuple,
    removed: dict[str, set[str]],
    marker: str | None,
    anchor: str | None,
    after,
) -> dict:
    model = BodyFrameModel().declare(deepcopy(spec))
    phrases = _mechanism_phrases(model)
    declaration = _declaration(model)
    if marker:
        if not anchor:
            raise RuntimeError(f"{sample_id} 有标记但没有锚点。")
        declaration = _insert_marker(declaration, anchor, marker)
    _check_declaration(declaration)
    questions = []
    processes = {}
    process_text = {}
    excluded: list[str] = []
    for process, _generate in PROCESSES:
        template = _process_template(model, process)
        before = SLOT_MARK.findall(template)
        template = _apply_edits(template, edits, process)
        if "16" in removed.get(TZ, ()) and process == TZ:
            template = _strip_order_paragraph(template)
            template = _close_order_gap(template)
            found = HEADING.findall(template)
            if found != [str(number) for number in range(1, 15)]:
                raise RuntimeError(f"{sample_id} 删节后的小节编号是 {found}，应为 1 到 14。")
        if after is not None and any(owner == process for owner, _old, _new in edits):
            template = after(template)
        after_slots = SLOT_MARK.findall(template)
        expected = removed.get(process, set())
        dropped = set(before) - set(after_slots)
        if dropped != expected:
            raise RuntimeError(f"{sample_id} {process} 去掉的空是 {sorted(dropped)}，应为 {sorted(expected)}。")
        if after_slots != [local_id for local_id in before if local_id not in expected]:
            raise RuntimeError(f"{sample_id} {process} 去掉规则问句后打乱了其余空的顺序。")
        items = [_question(process, item) for item in _payloads(spec)[process]]
        kept = [item for item in items if item["local_id"] not in expected]
        if sorted((item["local_id"] for item in kept), key=int) != sorted(after_slots, key=int):
            raise RuntimeError(f"{sample_id} {process} 的题目和模板槽不一致。")
        by_local = {item["local_id"]: item for item in kept}
        _check_sentences(process, template, phrases, by_local)
        processes[process] = _render_process(process, by_local, phrases=phrases, template=template)
        process_text[process] = _public_template(process, template)
        questions.extend(kept)
        excluded.extend(item["id"] for item in items if item["local_id"] in expected)
    record = {
        "id": sample_id,
        "title": title,
        "factor": factor,
        "condition": condition,
        "paired_standard_id": standard_id,
        "change": change,
        "spec": spec,
        "declaration": declaration,
        "prior": PRIOR,
        "process_text": process_text,
        "questions": questions,
        "excluded_ids": excluded,
        "tier": 0,
        "group": "pair",
    }
    record["processes"] = processes
    record["review_note"] = _note(title, condition, excluded)
    return record


def _answers(record: dict) -> dict[str, str]:
    return {item["id"]: item["answer"] for item in record["questions"]}


def _diff_ids(left: dict, right: dict) -> list[str]:
    shared = set(left) & set(right)
    return sorted((item_id for item_id in shared if left[item_id] != right[item_id]), key=int)


def _spec_keys(left: dict, right: dict) -> list[str]:
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _check_physics(default: dict, specs: dict[str, dict]) -> None:
    def raw(name: str, process: str, local_id: str) -> str:
        return _raw(specs[name], process, local_id)

    if raw("alpha", TZ, "20") != "下方" or raw("default", TZ, "20") != "下方":
        raise RuntimeError("理论迎角配对改动了机构迎角下的空速方向。")
    if raw("alpha", TZ, "21") != "上方" or raw("alpha", TZ, "22") != "负":
        raise RuntimeError("理论迎角反常后的符号不是负。")
    if raw("default", TZ, "22") != "正":
        raise RuntimeError("默认理论迎角符号不是正。")
    if raw("alpha", TZ, "26") != raw("default", TZ, "26"):
        raise RuntimeError("理论迎角配对改动了机构迎角矩阵。")
    if not raw("alpha", TZ, "80").startswith("-arctan"):
        raise RuntimeError("理论迎角反常后的 α_t 没有把负号写在外面。")

    if raw("roll", TZ, "35") != "相反" or raw("roll", TZ, "36") != "上方":
        raise RuntimeError("机构滚转反常没有改成与右手定则相反、右翼向上。")
    if raw("roll", TZ, "37") != "相同" or raw("default", TZ, "37") != "相同":
        raise RuntimeError("机构滚转配对改动了理论滚转角。")
    if raw("roll", TZ, "39") != "负":
        raise RuntimeError("γ_M>0 时理论滚转角的符号没有变成负。")
    if raw("roll", TZ, "43") == raw("default", TZ, "43"):
        raise RuntimeError("机构滚转反常没有改变滚转矩阵。")

    if raw("mount", TZ, "11") != "反装" or raw("mount", TZ, "12") != "相反":
        raise RuntimeError("反装配对没有写成反装、法向侧向相反。")
    if raw("mount", TZ, "13") != "[[1, 0, 0], [0, -1, 0], [0, 0, -1]]":
        raise RuntimeError(f"反装 RP2 不是预期矩阵：{raw('mount', TZ, '13')}")
    if raw("default", TZ, "13") != "[[1, 0, 0], [0, 1, 0], [0, 0, 1]]":
        raise RuntimeError("默认 RP2 不是单位矩阵。")

    if raw("body", TZ, "5") != "YTZ" or raw("body", TZ, "6") != "XTZ" or raw("body", TZ, "7") != "-ZTZ":
        raise RuntimeError("左手体轴系改动了已声明的 X、Y，或 Z 没有反向。")
    if raw("default", TZ, "7") != "ZTZ":
        raise RuntimeError("默认体轴侧向不是 ZTZ。")
    if raw("body", TZ, "8") == raw("default", TZ, "8"):
        raise RuntimeError("左手体轴系没有改变 RP1。")

    if [raw("wrist", TZ, "16"), raw("wrist", TZ, "17"), raw("wrist", TZ, "18")] != ["侧滑角", "迎角", "滚转角"]:
        raise RuntimeError("球腕顺序不是偏航、俯仰、滚转。")
    if [raw("default", TZ, "16"), raw("default", TZ, "17"), raw("default", TZ, "18")] != ["迎角", "侧滑角", "滚转角"]:
        raise RuntimeError("功角机构顺序不是迎角、侧滑、滚转。")
    if [raw("wrist", TZ, "47"), raw("wrist", TZ, "48"), raw("wrist", TZ, "49")] != ["R_γ_M", "R_α_M", "R_β_M"]:
        raise RuntimeError("球腕的 R1 R2 R3 不是滚转、迎角、侧滑。")
    if [raw("default", TZ, "47"), raw("default", TZ, "48"), raw("default", TZ, "49")] != ["R_γ_M", "R_β_M", "R_α_M"]:
        raise RuntimeError("功角机构的 R1 R2 R3 不是滚转、侧滑、迎角。")
    if raw("default", TZ, "23") != "中间体轴系" or raw("default", TZ, "24") != "ZTZ":
        raise RuntimeError("默认迎角没有写在中间体轴系的侧向轴上。")
    if raw("mount", TZ, "23") != "中间体轴系" or raw("mount", TZ, "24") != "ZTZ":
        raise RuntimeError("反装把姿态转轴改出了中间体轴系。")
    if raw("body", TZ, "34") == raw("default", TZ, "34") or raw("body", TZ, "43") == raw("default", TZ, "43"):
        raise RuntimeError("左手体轴系没有改写中间体轴系中的姿态矩阵。")
    if raw("wrist", TZ, "55") == raw("default", TZ, "55"):
        raise RuntimeError("两种机构顺序的总矩阵相同，不能做配对。")
    if raw("wrist", TZ, "11") != "正装" or raw("wrist", TZ, "22") != "正" or raw("wrist", TZ, "35") != "相同":
        raise RuntimeError("球腕顺序改动了安装、迎角符号或滚转方向。")


def _check_factor(factor: dict, rule: dict, unmarked: dict, marked: dict, standard: dict) -> tuple[list[str], list[str]]:
    if rule["spec"] != unmarked["spec"] or rule["spec"] != marked["spec"]:
        raise RuntimeError(f"{factor['key']} 三种条件的定义规格不一致。")
    if rule["declaration"] != unmarked["declaration"]:
        raise RuntimeError(f"{factor['key']} 的先问规则稿带上了标记，或定义声明与无标记稿不同。")
    if marked["declaration"] != _insert_marker(unmarked["declaration"], factor["anchor"], factor["marker"]):
        raise RuntimeError(f"{factor['key']} 的有标记稿除提醒外还有别的差异。")
    if factor["marker"] in unmarked["declaration"] or factor["marker"] in rule["declaration"]:
        raise RuntimeError(f"{factor['key']} 的无标记稿或先问规则稿含有提醒。")
    if unmarked["process_text"] != marked["process_text"] or unmarked["processes"] != marked["processes"]:
        raise RuntimeError(f"{factor['key']} 的无标记稿和有标记稿过程不同。")
    if unmarked["questions"] != marked["questions"]:
        raise RuntimeError(f"{factor['key']} 的无标记稿和有标记稿答案不同。")
    rule_ids = {item["id"] for item in rule["questions"]}
    hidden_ids = {item["id"] for item in unmarked["questions"]}
    excluded = set(unmarked["excluded_ids"])
    if not excluded:
        raise RuntimeError(f"{factor['key']} 没有去掉任何规则询问。")
    if excluded & hidden_ids:
        raise RuntimeError(f"{factor['key']} 的无标记稿仍含目标规则询问。")
    if not excluded <= rule_ids:
        raise RuntimeError(f"{factor['key']} 的先问规则稿缺少应保留的规则询问。")
    if hidden_ids != rule_ids - excluded:
        raise RuntimeError(f"{factor['key']} 的无标记稿除规则询问外还有增减。")
    if excluded != _emphasis_public_ids():
        missing = sorted(_emphasis_public_ids() - excluded, key=int)
        extra = sorted(excluded - _emphasis_public_ids(), key=int)
        raise RuntimeError(f"{factor['key']} 的无标记稿没有去掉全部定义的规则询问，缺 {missing}，多 {extra}。")
    for item_id in excluded:
        token = f"{{ID:{item_id},"
        blob_u = "\n".join(unmarked["processes"].values())
        blob_r = "\n".join(rule["processes"].values())
        if token in blob_u or token not in blob_r:
            raise RuntimeError(f"{factor['key']} 的规则询问 {item_id} 没有按条件出现。")
    kept_diff = _diff_ids(_answers(unmarked), _answers(standard))
    rule_diff = _diff_ids(_answers(rule), _answers(standard))
    if not kept_diff:
        raise RuntimeError(f"{factor['key']} 去掉规则询问后，剩余答案与标准完全相同。")
    if not (set(rule_diff) & excluded):
        raise RuntimeError(f"{factor['key']} 的规则询问答案没有离开标准。")
    changed = _spec_keys(rule["spec"], standard["spec"])
    if factor["key"] == "order":
        if changed != ["body_frame"]:
            raise RuntimeError(f"顺序配对改动了 {changed}。")
        left = deepcopy(rule["spec"]["body_frame"])
        right = deepcopy(standard["spec"]["body_frame"])
        if left.pop("euler_order") != ["alpha", "beta", "gamma"] or right.pop("euler_order") != ["beta", "alpha", "gamma"]:
            raise RuntimeError("顺序配对的欧拉顺序不是功角机构对球腕。")
        if left != right:
            raise RuntimeError("顺序配对还改了体轴坐标轴。")
    elif factor["key"] == "body":
        if changed != ["body_frame"]:
            raise RuntimeError(f"体轴配对改动了 {changed}。")
        if rule["spec"]["body_frame"]["chirality"] != "left":
            raise RuntimeError("体轴配对没有改成左手系。")
        if rule["spec"]["body_frame"]["XTZ"] != standard["spec"]["body_frame"]["XTZ"]:
            raise RuntimeError("体轴配对改动了 XTZ。")
        if rule["spec"]["body_frame"]["YTZ"] != standard["spec"]["body_frame"]["YTZ"]:
            raise RuntimeError("体轴配对改动了 YTZ。")
    else:
        expected = {"alpha": ["alpha"], "roll": ["gamma_m"], "mount": ["placement"]}[factor["key"]]
        if changed != expected:
            raise RuntimeError(f"{factor['key']} 改动了 {changed}，应为 {expected}。")
    return kept_diff, rule_diff


def _index(rows: list[dict], diffs: dict[str, tuple[list[str], list[str]]]) -> str:
    lines = [
        "# 单因素工程契约配对",
        "",
        "五组反常定义，每组三条：无标记、有标记、先问规则。",
        "标准定义也各有无标记和有标记：统一默认是 standard-default-unmarked、standard-default-marked；",
        "球腕顺序是 standard-order-unmarked、standard-order-marked。",
        "standard-default 与 standard-order 仍是完整对照稿，规则询问全部保留，供答案对照和先问规则使用。",
        "理论迎角、机构滚转、正装反装、体轴系手性这四组的标准侧是统一默认。",
        "机构角顺序的标准侧是三自由度动轴球腕，执行顺序为偏航、俯仰、滚转；反常侧沿用当前功角机构。",
        "",
        "无标记：定义仍写在定义声明里，不额外提醒，并去掉全部定义的规则询问，不是只去掉正在改的那一条。",
        "有标记：与无标记的过程和答案相同，只在被改定义前加一句“必须按本题定义执行”。",
        "标准定义的有标记没有单独被改的定义，提醒放在定义声明最前，要求后续步骤按全部定义执行。",
        "先问规则：定义声明与无标记相同，没有提醒，规则询问全部保留。",
        "空的编号不重排，便于和完整标准稿对照。",
        "",
        "模型输入不要带上文件标题和审阅说明。评测用 samples.jsonl 里的定义声明和 process_text。",
        "",
        "| 配对 | 标准 | 无标记 | 有标记 | 先问规则 | 去掉的规则询问 | 无标记中与标准不同的空 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for factor in FACTORS:
        kept, _rule = diffs[factor["key"]]
        unmarked = next(row for row in rows if row["id"] == f"{factor['key']}-unmarked")
        lines.append(
            "| {title} | {standard} | {key}-unmarked | {key}-marked | {key}-rule | {excluded} | {count} 个 |".format(
                title=factor["title"],
                standard=factor["standard_id"],
                key=factor["key"],
                excluded="、".join(unmarked["excluded_ids"]),
                count=len(kept),
            )
        )
    lines.extend(["", "## 与标准答案不同的空", ""])
    for factor in FACTORS:
        kept, rule = diffs[factor["key"]]
        lines.append(f"### {factor['title']}")
        lines.append("")
        lines.append(factor["change"])
        lines.append("")
        lines.append("无标记、有标记中与标准不同：" + "、".join(kept))
        lines.append("")
        lines.append("先问规则中与标准不同：" + "、".join(rule))
        lines.append("")
    return "\n".join(lines)


def _markdown(record: dict) -> str:
    text = _render_markdown(record)
    return text.replace("\n\n", "\n\n" + record["review_note"] + "\n\n", 1)


def _check_standards(default_row: dict, wrist_row: dict) -> None:
    default_spec = deepcopy(STANDARD_SPEC)
    built = build_record({
        "id": "01",
        "title": "统一默认",
        "change": "全部取统一默认定义。",
        "spec": default_spec,
    })
    if built["declaration"] != default_row["declaration"] or built["processes"] != default_row["processes"]:
        raise RuntimeError("统一默认配对稿和现有生成器不一致。")
    wrist_spec = deepcopy(STANDARD_SPEC)
    _wrist(wrist_spec)
    wrist_built = build_record({
        "id": "wrist",
        "title": "常规机构角顺序",
        "change": "三自由度动轴球腕。",
        "spec": wrist_spec,
    })
    if wrist_built["declaration"] != wrist_row["declaration"] or wrist_built["processes"] != wrist_row["processes"]:
        raise RuntimeError("球腕标准稿和生成器不一致。")
    if "三自由度动轴球腕" not in wrist_row["declaration"] or "偏航、俯仰、滚转" not in wrist_row["declaration"]:
        raise RuntimeError("球腕定义声明没有写明顺序。")
    if "功角支撑杆" in wrist_row["declaration"]:
        raise RuntimeError("球腕定义声明仍在使用功角支撑杆。")
    if "本题规定机构姿态角按偏航、俯仰、滚转的顺序执行" not in wrist_row["processes"][TZ]:
        raise RuntimeError("球腕过程仍把顺序写成由机构结构推断。")
    if "根据机构结构，推断第一个旋转角" in wrist_row["processes"][TZ]:
        raise RuntimeError("球腕过程仍保留推断句。")
    if "三自由度动轴球腕" in default_row["declaration"] or "功角支撑杆" not in default_row["declaration"]:
        raise RuntimeError("统一默认的机构描述被球腕替换了。")


def main() -> None:
    default_spec = deepcopy(STANDARD_SPEC)
    wrist_spec = deepcopy(STANDARD_SPEC)
    _wrist(wrist_spec)
    specs = {
        "default": default_spec,
        "wrist": wrist_spec,
        "alpha": deepcopy(STANDARD_SPEC),
        "roll": deepcopy(STANDARD_SPEC),
        "mount": deepcopy(STANDARD_SPEC),
        "body": deepcopy(STANDARD_SPEC),
        "order": deepcopy(STANDARD_SPEC),
    }
    _alpha(specs["alpha"])
    _roll(specs["roll"])
    _mount(specs["mount"])
    _body(specs["body"])
    _check_physics(default_spec, specs)

    standards = {
        "standard-default": _make(
            default_spec, "standard-default", "统一默认",
            "全部取统一默认定义。理论迎角、机构滚转、正装反装、体轴系手性四组以此为标准侧。",
            "default", "standard", "standard-default", (), {}, None, None, None,
        ),
        "standard-order": _make(
            wrist_spec, "standard-order", "常规机构角顺序",
            "攻角机构改为三自由度动轴球腕，执行顺序规定为偏航、俯仰、滚转。其余定义保持统一默认。",
            "order", "standard", "standard-order", (), {}, None, None, None,
        ),
    }
    _check_standards(standards["standard-default"], standards["standard-order"])

    rows = [standards["standard-default"], standards["standard-order"]]
    for spec, stem, title, change, factor_name, standard_id in (
        (
            default_spec, "standard-default", "统一默认",
            "全部取统一默认定义。过程和有标记稿相同，只是没有“必须按定义执行”的提醒。",
            "default", "standard-default",
        ),
        (
            wrist_spec, "standard-order", "常规机构角顺序",
            "攻角机构改为三自由度动轴球腕，执行顺序规定为偏航、俯仰、滚转。其余定义保持统一默认。",
            "order", "standard-order",
        ),
    ):
        shared_standard = {
            "spec": spec,
            "factor": factor_name,
            "change": change,
            "standard_id": standard_id,
            "anchor": "固定约定",
            "edits": EMPHASIS_EDITS,
            "removed": EMPHASIS_REMOVED,
            "after": None,
        }
        rows.append(_make(
            sample_id=f"{stem}-unmarked",
            title=f"{title}·无标记",
            condition="unmarked",
            marker=None,
            **shared_standard,
        ))
        rows.append(_make(
            sample_id=f"{stem}-marked",
            title=f"{title}·有标记",
            condition="marked",
            marker=MARK_ALL,
            **shared_standard,
        ))
    diffs = {}
    for factor in FACTORS:
        spec = specs[factor["key"]]
        standard = standards[factor["standard_id"]]
        shared = {
            "spec": spec,
            "factor": factor["key"],
            "change": factor["change"],
            "standard_id": factor["standard_id"],
            "anchor": factor["anchor"],
        }
        rule = _make(
            sample_id=f"{factor['key']}-rule",
            title=f"{factor['title']}·先问规则",
            condition="rule",
            edits=(),
            removed={},
            marker=None,
            after=None,
            **shared,
        )
        unmarked = _make(
            sample_id=f"{factor['key']}-unmarked",
            title=f"{factor['title']}·无标记",
            condition="unmarked",
            edits=EMPHASIS_EDITS,
            removed=EMPHASIS_REMOVED,
            marker=None,
            after=None,
            **shared,
        )
        marked = _make(
            sample_id=f"{factor['key']}-marked",
            title=f"{factor['title']}·有标记",
            condition="marked",
            edits=EMPHASIS_EDITS,
            removed=EMPHASIS_REMOVED,
            marker=factor["marker"],
            after=None,
            **shared,
        )
        if factor["key"] != "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError(f"{factor['key']} 的先问规则稿改写了与目标定义无关的过程句。")
        if factor["key"] == "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError("顺序反常稿没有沿用当前机构描述。")
        diffs[factor["key"]] = _check_factor(factor, rule, unmarked, marked, standard)
        rows.extend([unmarked, marked, rule])

    reference = next(row for row in rows if row["id"] == "standard-default-unmarked")
    for row in rows:
        if row["condition"] not in {"unmarked", "marked"}:
            continue
        if row["process_text"] != reference["process_text"]:
            raise RuntimeError(f"{row['id']} 的过程文本和标准无标记稿不同。")
        if set(row["excluded_ids"]) != _emphasis_public_ids():
            raise RuntimeError(f"{row['id']} 没有去掉全部定义的规则询问。")
    for stem in ("standard-default", "standard-order"):
        plain = next(row for row in rows if row["id"] == f"{stem}-unmarked")
        warned = next(row for row in rows if row["id"] == f"{stem}-marked")
        if MARK_ALL not in warned["declaration"] or MARK_ALL in plain["declaration"]:
            raise RuntimeError(f"{stem} 的有标记提醒没有只出现在有标记稿。")
        if plain["questions"] != warned["questions"]:
            raise RuntimeError(f"{stem} 的无标记和有标记答案不同。")

    conditions = [row for row in rows if row["condition"] != "standard"]
    if len(conditions) != 19:
        raise RuntimeError(f"无强调、有强调和先问规则共 {len(conditions)} 条，应为 19 条。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stored = []
    for row in rows:
        (OUT_DIR / f"{row['id']}.md").write_text(_markdown(row), encoding="utf-8")
        stored.append({key: value for key, value in row.items() if key not in {"processes", "review_note"}})
        print(
            f"{row['id']}: {len(row['questions'])} questions, "
            f"excluded {len(row['excluded_ids'])}",
            flush=True,
        )
    (OUT_DIR / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stored),
        encoding="utf-8",
    )
    (OUT_DIR / "00_索引.md").write_text(_index(rows, diffs) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} manuscripts to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
