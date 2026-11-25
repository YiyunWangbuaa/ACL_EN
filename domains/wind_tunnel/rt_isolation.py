# -*- coding: utf-8 -*-
"""Keep probe answers out of R_t, and keep the two R_t conditions apart.

R_t-original keeps the component matrices already answered in the flow.
R_t-rederive hides those matrices and still does not receive any probe answer.
The write-back prompt is a control, not a main result.
"""
from __future__ import annotations

import re

from cloze_eval import prompt_for
from domains.wind_tunnel.contract_probes import filled_prefix
from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET, PROCESS_TEXT, SLOT_MARK

PROCESS = "风轴系到体轴系"
RT_ID = "55"
DEEP_ANCHOR = "51"
PLACEHOLDER = "（此分矩阵答案已隐去）"
PROBE_MARK = "《探针记录》"
# The name is stated before the matrix blank. Shallow probes stop at that sentence.
MATRICES = (
    {"key": "rp1", "name": "RP1", "blank": "8", "shallow_anchor": "4", "named": "前置变换矩阵为RP1"},
    {"key": "rp2", "name": "RP2", "blank": "13", "shallow_anchor": "12", "named": "前置变换矩阵为RP2"},
    {"key": "alpha", "name": "R_α_M", "blank": "26", "shallow_anchor": "25", "named": "对应的旋转矩阵为R_α_M"},
    {"key": "beta", "name": "R_β_M", "blank": "34", "shallow_anchor": "33", "named": "对应的旋转矩阵为R_β_M"},
    {"key": "gamma", "name": "R_γ_M", "blank": "43", "shallow_anchor": "42", "named": "对应的旋转矩阵为R_γ_M"},
)
COMPONENT_IDS = tuple(item["blank"] for item in MATRICES)
CONTROL_LEAD = "控制组把探针答案显式写入前缀。R_γ_M 的探针记录为"
# 与 data/wind_tunnel/review/00_注入格式.md 的矩阵格式相同。
MATRIX_FORMAT = (
    "输出格式：矩阵。输出一行3×3矩阵，形如[[a, b, c], [d, e, f], [g, h, i]]。"
    "逗号后恰好一个空格。元素为整数，或由cos、sin和本题角度符号组成的表达式。"
    "乘号用*。不要换行，不要使用LaTeX。"
    "示例：[[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]。示例不是本题答案。"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _copy(memory: dict[str, str]) -> dict[str, str]:
    return {str(key): value for key, value in memory.items()}


def hide_components(memory: dict[str, str]) -> dict[str, str]:
    """Question sentences stay. Their matrix answers become a placeholder."""
    hidden = _copy(memory)
    for blank_id in COMPONENT_IDS:
        if blank_id not in hidden:
            raise KeyError(f"冻结前缀缺少分矩阵 ID{blank_id}")
        hidden[blank_id] = PLACEHOLDER
    return hidden


def _question(name: str) -> str:
    return f"写出{name}的展开式。{MATRIX_FORMAT}"


def _template(chain: dict) -> str:
    """The subprocess text this chain was asked from, with mechanism phrases filled in.

    Single-factor rows do not store ``process_text``. The stock template numbers
    slots inside each subprocess, while probe anchors use the public blank ids.
    """
    template = chain.get("process_text")
    if not template:
        offset = PROCESS_OFFSET[chain["process"]]
        template = SLOT_MARK.sub(
            lambda match: f"@@{int(match.group(1)) + offset}@@",
            PROCESS_TEXT[chain["process"]],
        )
    for token, text in (chain.get("phrases") or {}).items():
        template = template.replace(token, text)
    return template


def _next_slotfree(chain: dict, anchor: str) -> str:
    """Naming sentences after the anchor, while those sentences have no blank."""
    template = _template(chain)
    found = re.search(rf"@@{re.escape(anchor)}@@", template)
    if found is None:
        raise KeyError(f"题面没有 @@{anchor}@@")
    tail = template[found.end():]
    period = tail.find("。")
    if period < 0:
        raise RuntimeError(f"ID{anchor} 后面没有句号")
    rest = tail[period + 1:].lstrip()
    extra: list[str] = []
    while rest:
        end = rest.find("。")
        if end < 0:
            break
        sentence = rest[: end + 1].strip()
        if SLOT_MARK.search(sentence):
            break
        extra.append(sentence)
        rest = rest[end + 1:].lstrip()
    return "".join(extra)


def _probe_prefix(chain: dict, anchor: str, memory: dict[str, str]) -> str:
    prefix = filled_prefix(chain, anchor, memory)
    extra = _next_slotfree(chain, anchor)
    if extra:
        prefix = f"{prefix}{extra}"
    return prefix


def _probe_prompt(chain: dict, anchor: str, memory: dict[str, str], question: str) -> str:
    prefix = _probe_prefix(chain, anchor, memory)
    if question in prefix:
        raise RuntimeError("探针题干写进了过程前缀")
    if PROBE_MARK in prefix or PROBE_MARK in question:
        raise RuntimeError("探针前缀里已经有探针记录")
    return "\n".join([
        "这是一次旁路探针。只回答下面这一个问题。",
        "不要填写过程中的空，不要继续后面的句子。",
        "这个答案只作评分记录，不会写回原流程，也不会进入 R_t。",
        "",
        "【定义声明】",
        chain["declaration"],
        "",
        "【过程】",
        prefix,
        "",
        "【探针】",
        question,
    ])


def _rt_prompt(chain: dict, memory: dict[str, str]) -> str:
    return prompt_for(chain, [RT_ID], "self-trunc", _copy(memory))


def _inject_control(prompt: str, probe_answer: str) -> str:
    sentence = f"{CONTROL_LEAD}{probe_answer}。"
    needle = "写出R1*R2*R3符号化简后的总矩阵"
    if needle not in prompt:
        raise RuntimeError("R_t 题面里没有总矩阵提问，无法放入控制句")
    return prompt.replace(needle, sentence + needle, 1)


def build_conditions(chain: dict, frozen: dict[str, str], held: dict[str, dict[str, str]]) -> dict[str, str]:
    """frozen is fixed before any probe. held answers are never copied into the main prompts."""
    if chain["process"] != PROCESS:
        raise RuntimeError(f"这一组条件目前只用于{PROCESS}")
    required = set(COMPONENT_IDS) | {item["shallow_anchor"] for item in MATRICES} | {DEEP_ANCHOR, RT_ID}
    missing = sorted(required - set(map(str, frozen)))
    if missing:
        raise KeyError("冻结前缀缺少 " + "、".join(missing))
    answers = []
    for item in MATRICES:
        pair = held[item["key"]]
        if PROBE_MARK not in pair["shallow"] or PROBE_MARK not in pair["deep"]:
            raise RuntimeError(f"{item['name']} 的探针答案必须带记录标记")
        answers.extend((pair["shallow"], pair["deep"]))
    original = _copy(frozen)
    masked = hide_components(original)
    prompts = {
        "rt_original": _rt_prompt(chain, original),
        "rt_rederive": _rt_prompt(chain, masked),
    }
    for item in MATRICES:
        question = _question(item["name"])
        prompts[f"shallow_{item['key']}"] = _probe_prompt(
            chain, item["shallow_anchor"], original, question,
        )
        prompts[f"deep_original_{item['key']}"] = _probe_prompt(
            chain, DEEP_ANCHOR, original, question,
        )
        prompts[f"deep_rederive_{item['key']}"] = _probe_prompt(
            chain, DEEP_ANCHOR, masked, question,
        )
    prompts["rt_writeback_control"] = _inject_control(prompts["rt_original"], held["gamma"]["deep"])
    _assert_isolated(prompts, original, answers)
    return prompts


def _assert_isolated(prompts: dict[str, str], frozen: dict[str, str], answers: list[str]) -> None:
    gold = {item["blank"]: _compact(frozen[item["blank"]]) for item in MATRICES}
    if any(not text or text == _compact(PLACEHOLDER) for text in gold.values()):
        raise RuntimeError("分矩阵金标是空的，遮蔽检查没有意义")
    rt_answer = _compact(frozen[RT_ID])

    def compact_has(name: str, part: str) -> bool:
        return bool(part) and part in _compact(prompts[name])

    probe_names = [
        key for key in prompts
        if key.startswith("shallow_") or key.startswith("deep_")
    ]
    for name in probe_names:
        if any(answer in prompts[name] for answer in answers):
            raise RuntimeError(f"{name} 看见了探针答案")
        if rt_answer and rt_answer in _compact(prompts[name]):
            raise RuntimeError(f"{name} 看见了 R_t 的答案")
    for item in MATRICES:
        shallow = f"shallow_{item['key']}"
        deep_original = f"deep_original_{item['key']}"
        deep_rederive = f"deep_rederive_{item['key']}"
        question = _question(item["name"])
        if not question.startswith(f"写出{item['name']}的展开式。") or MATRIX_FORMAT not in question:
            raise RuntimeError(f"{item['name']} 的探针没有要求写出展开式，或没有附上矩阵格式")
        if "重新推导" in question or "重新写出" in question:
            raise RuntimeError(f"{item['name']} 的探针仍在要求重新推导矩阵")
        if item["named"] not in prompts[shallow]:
            raise RuntimeError(f"{item['name']} 的浅层探针还没看见矩阵名称")
        if compact_has(shallow, gold[item["blank"]]):
            raise RuntimeError(f"{item['name']} 的浅层探针看见了自己的原答案")
        if not compact_has(deep_original, gold[item["blank"]]):
            raise RuntimeError(f"{item['name']} 的原流程深层探针应当看见原答案")
        if compact_has(deep_rederive, gold[item["blank"]]):
            raise RuntimeError(f"{item['name']} 的重推导探针仍然看见原答案")
        if item["named"] not in prompts[deep_rederive]:
            raise RuntimeError(f"{item['name']} 的重推导探针连名称都没有")

    for name in ("rt_original", "rt_rederive"):
        if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
            raise RuntimeError(f"{name} 看见了探针答案")
    if not all(compact_has("rt_original", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("R_t-original 应当保留原流程里的分矩阵")
    if any(compact_has("rt_rederive", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("R_t-rederive 仍然看见分矩阵答案")
    if prompts["rt_original"] == prompts["rt_rederive"]:
        raise RuntimeError("两个 R_t 条件的提示相同，不能分开记分")

    gamma_deep = next(answer for answer in answers if answer.endswith("深层R_γ_M"))
    if gamma_deep not in prompts["rt_writeback_control"]:
        raise RuntimeError("写回控制没有放入 R_γ_M 的深层探针答案")
    if any(answer in prompts["rt_writeback_control"] for answer in answers if answer != gamma_deep):
        raise RuntimeError("写回控制放入了不该写入的探针答案")
    if not all(compact_has("rt_writeback_control", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("写回控制应当在原始前缀上追加探针答案，而不是改成重推导")


STAGES = (
    {
        "process": "天平系到体轴系",
        "deep_anchor": "100",
        "total_id": "104",
        "matrices": (
            {"key": "rtp_tz1", "name": "RTP_TZ1", "blank": "95", "shallow_anchor": "91", "named": "前置矩阵为RTP_TZ1"},
            {"key": "roll", "name": "R_(γ_M-γ_tp)", "blank": "88", "shallow_anchor": "87", "named": "对应的旋转矩阵为R_(γ_M-γ_tp)"},
            {"key": "rtp_tz2", "name": "RTP_TZ2", "blank": "97", "shallow_anchor": "88", "named": "前置矩阵为RTP_TZ2"},
        ),
    },
    {
        "process": "天平系到风轴系",
        "deep_anchor": "134",
        "total_id": "",
        "matrices": (
            {"key": "rp_tp_fd", "name": "RP_TP_FD", "blank": "126", "shallow_anchor": "125", "named": "对应的变换矩阵为RP_TP_FD"},
            {"key": "gamma_tp", "name": "R_γ_tp", "blank": "132", "shallow_anchor": "131", "named": "对应的旋转矩阵为R_γ_tp"},
        ),
    },
)


def isolate_stage(chain: dict, frozen: dict[str, str], stage: dict) -> dict[str, str]:
    """Shallow and deep probes for one later stage. Probe answers stay out of the total, when there is one."""
    if chain["process"] != stage["process"]:
        raise RuntimeError(f"探针阶段与过程不符：{chain['process']}")
    matrices = stage["matrices"]
    hide_ids = tuple(item["blank"] for item in matrices)
    required = set(hide_ids) | {item["shallow_anchor"] for item in matrices} | {stage["deep_anchor"]}
    total_id = stage.get("total_id") or ""
    if total_id:
        required.add(total_id)
    missing = sorted(required - set(map(str, frozen)))
    if missing:
        raise KeyError("冻结前缀缺少 " + "、".join(missing))
    answers = [f"{PROBE_MARK}浅层{item['name']}" for item in matrices]
    answers += [f"{PROBE_MARK}深层{item['name']}" for item in matrices]
    original = _copy(frozen)
    masked = _copy(frozen)
    for blank_id in hide_ids:
        masked[blank_id] = PLACEHOLDER
    prompts: dict[str, str] = {}
    for item in matrices:
        question = _question(item["name"])
        prompts[f"shallow_{item['key']}"] = _probe_prompt(chain, item["shallow_anchor"], original, question)
        prompts[f"deep_original_{item['key']}"] = _probe_prompt(chain, stage["deep_anchor"], original, question)
        prompts[f"deep_rederive_{item['key']}"] = _probe_prompt(chain, stage["deep_anchor"], masked, question)
    if total_id:
        prompts["total_original"] = prompt_for(chain, [total_id], "self-trunc", _copy(original))
        prompts["total_rederive"] = prompt_for(chain, [total_id], "self-trunc", _copy(masked))
    _assert_stage(prompts, original, answers, matrices, total_id)
    return prompts


def _assert_stage(prompts, frozen, answers, matrices, total_id: str) -> None:
    gold = {item["blank"]: _compact(frozen[item["blank"]]) for item in matrices}
    if any(not text or text == _compact(PLACEHOLDER) for text in gold.values()):
        raise RuntimeError("分矩阵金标是空的，遮蔽检查没有意义")
    total_answer = _compact(frozen[total_id]) if total_id else ""

    def compact_has(name: str, part: str) -> bool:
        return bool(part) and part in _compact(prompts[name])

    probe_names = [key for key in prompts if key.startswith("shallow_") or key.startswith("deep_")]
    for name in probe_names:
        if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
            raise RuntimeError(f"{name} 看见了探针答案")
        if total_answer and total_answer in _compact(prompts[name]):
            raise RuntimeError(f"{name} 看见了总矩阵答案")
    for item in matrices:
        shallow = f"shallow_{item['key']}"
        question = _question(item["name"])
        if not question.startswith(f"写出{item['name']}的展开式。") or MATRIX_FORMAT not in prompts[shallow]:
            raise RuntimeError(f"{item['name']} 的探针没有要求写出展开式，或没有附上矩阵格式")
        if item["named"] not in prompts[shallow]:
            raise RuntimeError(f"{item['name']} 的浅层探针还没看见矩阵名称")
        if compact_has(shallow, gold[item["blank"]]):
            raise RuntimeError(f"{item['name']} 的浅层探针看见了自己的原答案")
        if not compact_has(f"deep_original_{item['key']}", gold[item["blank"]]):
            raise RuntimeError(f"{item['name']} 的原流程深层探针应当看见原答案")
        if any(compact_has(f"deep_rederive_{item['key']}", gold[other["blank"]]) for other in matrices):
            raise RuntimeError(f"{item['name']} 的重推导探针仍然看见分矩阵答案")
    if total_id:
        for name in ("total_original", "total_rederive"):
            if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
                raise RuntimeError(f"{name} 看见了探针答案")
        if not all(compact_has("total_original", gold[item["blank"]]) for item in matrices):
            raise RuntimeError("总矩阵原流程应当保留分矩阵")
        if any(compact_has("total_rederive", gold[item["blank"]]) for item in matrices):
            raise RuntimeError("总矩阵重推导仍然看见分矩阵答案")
        if prompts["total_original"] == prompts["total_rederive"]:
            raise RuntimeError("两个总矩阵条件的提示相同，不能分开记分")

