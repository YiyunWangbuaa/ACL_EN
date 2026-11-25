# -*- coding: utf-8 -*-
"""Single-factor contract pairs.

Five abnormal definitions, each written once: the declaration states the
contract, and every definition's rule question stays on the page. Two
standards use the same full question set. Manuscripts that drop those
questions, or that add an extra warning in front of a changed rule, are not
generated.

The spherical-wrist order is the standard side of the order pair. The current
angle-rod description is the abnormal side. The other four abnormal
definitions are paired with the unified default.

Run::

    python -m domains.wind_tunnel.pair_generate
"""

from __future__ import annotations

import json
from copy import deepcopy

from body_frame_model import BodyFrameModel
from domain_layout import DOMAINS
from domains.wind_tunnel.dataset_generate import (
    PRIOR,
    PROCESSES,
    SLOT_MARK,
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
_CACHE: dict[str, dict[str, list[dict]]] = {}
RULE_IDS = (
    "standard-default",
    "standard-order",
    "alpha-rule",
    "roll-rule",
    "order-rule",
    "mount-rule",
    "body-rule",
)


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


FACTORS = (
    {
        "key": "alpha",
        "title": "理论迎角正负",
        "change": "只把理论迎角改为空速投影在机体轴向上方时为正。机构迎角、坐标轴、安装和旋转顺序都不动。",
        "apply": _alpha,
        "standard_id": "standard-default",
    },
    {
        "key": "roll",
        "title": "机构滚转方向",
        "change": "只把机构滚转角 γ_M 改为与绕模型绝对轴向正方向（前方）的右手定则相反。理论滚转角仍与右手定则相同。",
        "apply": _roll,
        "standard_id": "standard-default",
    },
    {
        "key": "order",
        "title": "机构角顺序",
        "change": "反常侧沿用当前功角机构：顺序由支撑杆结构决定，为迎角、侧滑、滚转。标准侧改为三自由度动轴球腕，顺序规定为偏航、俯仰、滚转。",
        "apply": None,
        "standard_id": "standard-order",
    },
    {
        "key": "mount",
        "title": "正装反装",
        "change": "只把模型安装改为反装。坐标系手性、角度正方向和旋转顺序都不动。",
        "apply": _mount,
        "standard_id": "standard-default",
    },
    {
        "key": "body",
        "title": "体轴系手性",
        "change": "只把体轴系改为左手系。XTZ 仍沿机体轴向指向前方，YTZ 仍沿机体法向指向上方，ZTZ 由手性推断。",
        "apply": _body,
        "standard_id": "standard-default",
    },
)


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


def _public_template(process: str, template: str) -> str:
    from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET

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


def _note(title: str, condition: str) -> str:
    if condition == "standard":
        return (
            f"审阅说明：{title}。这一句和文件标题不进入模型输入。"
            "模型只看到下面的定义声明、物理常识和过程。本条是配对的标准侧，规则询问全部保留。"
        )
    if condition != "rule":
        raise RuntimeError(f"只生成测点完备的正常契约稿，不能写 {condition}。")
    return (
        f"审阅说明：{title}，条件为先问规则。这一句和文件标题不进入模型输入。"
        "模型只看到下面的定义声明、物理常识和过程。本条保留全部定义的规则询问，定义声明没有额外提醒。"
    )


def _make(
    spec: dict,
    sample_id: str,
    title: str,
    change: str,
    factor: str,
    condition: str,
    standard_id: str,
) -> dict:
    model = BodyFrameModel().declare(deepcopy(spec))
    phrases = _mechanism_phrases(model)
    declaration = _declaration(model)
    if "本题特别约定" in declaration:
        raise RuntimeError(f"{sample_id} 的定义声明带了额外提醒。")
    _check_declaration(declaration)
    questions = []
    processes = {}
    process_text = {}
    for process, _generate in PROCESSES:
        template = _process_template(model, process)
        items = [_question(process, item) for item in _payloads(spec)[process]]
        slots = SLOT_MARK.findall(template)
        if sorted((item["local_id"] for item in items), key=int) != sorted(slots, key=int):
            raise RuntimeError(f"{sample_id} {process} 的题目和模板槽不一致。")
        by_local = {item["local_id"]: item for item in items}
        _check_sentences(process, template, phrases, by_local)
        processes[process] = _render_process(process, by_local, phrases=phrases, template=template)
        process_text[process] = _public_template(process, template)
        questions.extend(items)
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
        "excluded_ids": [],
        "tier": 0,
        "group": "pair",
    }
    record["processes"] = processes
    record["review_note"] = _note(title, condition)
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


def _check_factor(factor: dict, rule: dict, standard: dict) -> list[str]:
    if rule["excluded_ids"] or standard["excluded_ids"]:
        raise RuntimeError(f"{factor['key']} 去掉了规则询问。")
    if "本题特别约定" in rule["declaration"] or "本题特别约定" in standard["declaration"]:
        raise RuntimeError(f"{factor['key']} 的定义声明带了额外提醒。")
    rule_ids = {item["id"] for item in rule["questions"]}
    standard_ids = {item["id"] for item in standard["questions"]}
    if rule_ids != standard_ids:
        raise RuntimeError(f"{factor['key']} 的测点和标准稿不一致。")
    diff = _diff_ids(_answers(rule), _answers(standard))
    if not diff:
        raise RuntimeError(f"{factor['key']} 的答案与标准完全相同。")
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
    return diff


def _index(rows: list[dict], diffs: dict[str, list[str]]) -> str:
    by_id = {row["id"]: row for row in rows}
    lines = [
        "# 单因素工程契约配对",
        "",
        "五组反常定义，每组一份测点完备的稿面：定义写在声明里，规则询问全部保留，没有额外提醒。",
        "标准侧同样保留全部规则询问。统一默认是 standard-default，球腕顺序是 standard-order。",
        "理论迎角、机构滚转、正装反装、体轴系手性这四组的标准侧是统一默认。",
        "机构角顺序的标准侧是三自由度动轴球腕，执行顺序为偏航、俯仰、滚转；反常侧沿用当前功角机构。",
        "",
        "模型输入不要带上文件标题和审阅说明。评测用 samples.jsonl 里的定义声明和 process_text。",
        "",
        "| 配对 | 标准 | 稿面 | 题数 | 与标准不同的空 |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for factor in FACTORS:
        rule = by_id[f"{factor['key']}-rule"]
        lines.append(
            "| {title} | {standard} | {key}-rule | {count} | {diff} 个 |".format(
                title=factor["title"],
                standard=factor["standard_id"],
                key=factor["key"],
                count=len(rule["questions"]),
                diff=len(diffs[factor["key"]]),
            )
        )
    lines.extend(["", "## 与标准答案不同的空", ""])
    for factor in FACTORS:
        lines.append(f"### {factor['title']}")
        lines.append("")
        lines.append(factor["change"])
        lines.append("")
        lines.append("与标准不同：" + "、".join(diffs[factor["key"]]))
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
            "default", "standard", "standard-default",
        ),
        "standard-order": _make(
            wrist_spec, "standard-order", "常规机构角顺序",
            "攻角机构改为三自由度动轴球腕，执行顺序规定为偏航、俯仰、滚转。其余定义保持统一默认。",
            "order", "standard", "standard-order",
        ),
    }
    _check_standards(standards["standard-default"], standards["standard-order"])

    rows = [standards["standard-default"], standards["standard-order"]]
    diffs = {}
    for factor in FACTORS:
        spec = specs[factor["key"]]
        standard = standards[factor["standard_id"]]
        rule = _make(
            spec,
            f"{factor['key']}-rule",
            f"{factor['title']}·先问规则",
            factor["change"],
            factor["key"],
            "rule",
            factor["standard_id"],
        )
        if factor["key"] != "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError(f"{factor['key']} 的稿面改写了与目标定义无关的过程句。")
        if factor["key"] == "order" and rule["process_text"] != standards["standard-default"]["process_text"]:
            raise RuntimeError("顺序反常稿没有沿用当前机构描述。")
        diffs[factor["key"]] = _check_factor(factor, rule, standard)
        rows.append(rule)

    if [row["id"] for row in rows] != list(RULE_IDS):
        raise RuntimeError(f"稿面应为 {RULE_IDS}，实际是 {[row['id'] for row in rows]}。")
    if any(row["condition"] not in {"standard", "rule"} for row in rows):
        raise RuntimeError("生成结果含有非 rule 稿面。")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {f"{row['id']}.md" for row in rows} | {"00_索引.md"}
    for path in OUT_DIR.glob("*.md"):
        if path.name not in keep:
            path.unlink()
    stored = []
    for row in rows:
        (OUT_DIR / f"{row['id']}.md").write_text(_markdown(row), encoding="utf-8")
        stored.append({key: value for key, value in row.items() if key not in {"processes", "review_note"}})
        print(f"{row['id']}: {len(row['questions'])} questions", flush=True)
    (OUT_DIR / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in stored),
        encoding="utf-8",
    )
    (OUT_DIR / "00_索引.md").write_text(_index(rows, diffs) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} manuscripts to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
