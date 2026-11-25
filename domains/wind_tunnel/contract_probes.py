# -*- coding: utf-8 -*-
"""Side-channel contract probes for the wind-tunnel cloze.

A probe asks one local contract question on a frozen prefix. The answer is
recorded and is never written into a later sentence or into the target call.
The probe does not ask for the target matrix or formula. The target execution
stays the answer already produced by the original run.

Current public ids, not the schematic numbers in the design note:
    roll direction ID87 then matrix ID88; left-handed cells before ID34 and ID43;
    moment lever ID184, MCN_CZ sign ID192, formula ID198.
"""
from __future__ import annotations

import hashlib
import re

from cloze_score import normalize, sympy_equal
from domains.wind_tunnel.dataset_generate import PRIOR, SLOT_MARK, _sentence_groups

# These items are not primary rollback probes. A two-way sign collides with the
# other definition, ID80 is an ordinary formula miss, and ID104 can be produced
# by two inverted matrices cancelling.
EXCLUDED_TARGETS = frozenset({"22", "68", "80", "104"})

PROBES = (
    {
        "probe_id": "roll_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "机构滚转",
        "group": "roll",
        "sample_id": "roll-rule",
        "process": "天平系到体轴系",
        "after": "87",
        "target": "88",
        "kind": "cells",
        "cells": ((2, 3), (3, 2)),
        "repair_parent": ("85", "86"),
        "repair_other": ("87",),
        "question": (
            "根据前文已经确定的滚转方向，写入该旋转矩阵时，"
            "第2行第3列和第3行第2列的 sin 项应分别取什么符号？"
            "按 (2,3)=正或负,(3,2)=正或负 回答。不要写出完整矩阵。"
        ),
        "hidden_stem": "对应的旋转矩阵R_(γ_M-γ_tp)为",
    },
    {
        "probe_id": "beta_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "左手系",
        "group": "body",
        "sample_id": "body-rule",
        "process": "风轴系到体轴系",
        "after": "33",
        "target": "34",
        "kind": "cells",
        "cells": ((1, 3), (3, 1)),
        "repair_parent": ("32", "33"),
        "repair_other": ("31",),
        "question": (
            "根据前文已经确定的中间体轴系和旋转方向，R_β_M 中第1行第3列和第3行第1列的 sin(β_M) 应分别取什么符号？"
            "按 (1,3)=正或负,(3,1)=正或负 回答。不要写出完整矩阵。"
        ),
        "hidden_stem": "对应的旋转矩阵R_β_M为",
    },
    {
        "probe_id": "gamma_cells",
        "node_role": "probe",
        "new_label": "A-LOCAL-EXEC",
        "subtype": "matrix_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "左手系",
        "group": "body",
        "sample_id": "body-rule",
        "process": "风轴系到体轴系",
        "after": "42",
        "target": "43",
        "kind": "cells",
        "cells": ((2, 3), (3, 2)),
        "repair_parent": ("41", "42"),
        "repair_other": ("40",),
        "question": (
            "根据前文已经确定的中间体轴系和旋转方向，R_γ_M 中第2行第3列和第3行第2列的 sin(γ_M) 应分别取什么符号？"
            "按 (2,3)=正或负,(3,2)=正或负 回答。不要写出完整矩阵。"
        ),
        "hidden_stem": "对应的旋转矩阵R_γ_M为",
    },
    {
        "probe_id": "mcn_cz_sign",
        "node_role": "probe",
        "new_label": "A-SPATIAL",
        "subtype": "moment_sign",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "力矩公式",
        "group": "body",
        "sample_id": "body-rule",
        "process": "力矩参考点",
        "after": "191",
        "target": "198",
        "kind": "sign",
        "source_id": "192",
        "repair_parent": ("184",),
        "repair_other": ("190",),
        "question": (
            "根据前文已经确定的体轴系和力臂，等效力矩系数 MCN_CZ 的整体符号应取正号还是负号？"
            "只回答正或负。不要写出完整公式。"
        ),
        "hidden_stem": "等效力矩系数MCN_CZ的符号为",
    },
    {
        "probe_id": "mcn_cz_place",
        "node_role": "probe",
        "new_label": "A-MATH-FORMAT",
        "subtype": "sign_placement",
        "context_depth": "target_local",
        "writes_back": False,
        "case": "力矩公式",
        "group": "body",
        "sample_id": "body-rule",
        "process": "力矩参考点",
        "after": "188",
        "target": "198",
        "kind": "placement",
        "repair_parent": ("184",),
        "repair_other": ("185",),
        "question": (
            "根据前文已经确定的力臂，若 MCN_CZ 需要负号，这个负号应写在整个式子最外侧，还是写进力臂内部？"
            "只回答外侧或内部。不要写出完整公式。"
        ),
        "hidden_stem": "力臂内部必须为正",
    },
)

CONDITIONS = (
    "deep_b",
    "deep_gold",
    "replay",
    "matched",
    "repair_parent",
    "repair_other",
)
DEEP_CONDITIONS = frozenset({"deep_b", "deep_gold", "replay", "matched", "repair_parent", "repair_other"})

# ID58 only checks whether an earlier chirality choice reaches the component.
CONSEQUENCE = {
    "group": "body",
    "sample_id": "body-rule",
    "target": "58",
    "role": "组合后果节点",
}


def _prepared(chain: dict) -> str:
    template = chain["process_text"]
    for token, text in (chain.get("phrases") or {}).items():
        template = template.replace(token, text)
    return template


def _groups(chain: dict) -> list[list[str]]:
    return _sentence_groups(chain["process"], chain.get("phrases"), chain.get("process_text"))


def filled_prefix(chain: dict, through_id: str, memory: dict[str, str]) -> str:
    """Process text through the sentence that ends at through_id, with blanks filled."""
    through = str(through_id)
    groups = _groups(chain)
    index = next((i for i, group in enumerate(groups) if through in group), None)
    if index is None:
        raise KeyError(f"{chain['process']} 没有题号 {through}")
    if groups[index][-1] != through:
        raise RuntimeError(
            f"探针锚点 {through} 不是所在句的最后一个空，所在句是 {groups[index]}。"
            "切在半句会把同句后续空留在前缀里。"
        )
    filled = [blank_id for group in groups[: index + 1] for blank_id in group]
    missing = [blank_id for blank_id in filled if blank_id not in memory]
    if missing:
        raise KeyError(f"前缀缺少这些空的答案：{missing}")
    template = _prepared(chain)
    pieces: list[str] = []
    cursor = 0
    for match in SLOT_MARK.finditer(template):
        blank_id = match.group(1)
        pieces.append(template[cursor:match.start()])
        if blank_id not in set(filled):
            raise RuntimeError(f"探针前缀碰到了锚点 {through} 之后的空 {blank_id}")
        pieces.append(memory[blank_id])
        cursor = match.end()
        if blank_id == through:
            tail = template[match.end():]
            period = tail.find("。")
            if period < 0:
                raise RuntimeError(f"{chain['process']} 的 {through} 所在句没有句号。")
            if SLOT_MARK.search(tail[:period]):
                raise RuntimeError(f"{through} 的句子里还有未填的空。")
            pieces.append(tail[: period + 1])
            return "".join(pieces).strip()
    raise RuntimeError(f"没有在题面里截到 {through}")


def _question(probe: dict, condition: str, phrases: dict[str, str]) -> str:
    text = probe["question"]
    if "{" in text or "}" in text:
        raise RuntimeError(f"{probe['probe_id']} 的题干不能填入流程答案。")
    if phrases:
        return text
    return text


def build_target_prompt(chain: dict, probe: dict, memory: dict[str, str]) -> str:
    """Ask the real target sentence on a repaired prefix, then stop.

    Earlier blanks use the branch memory. The target sentence stays open.
    Text after that sentence is omitted, and this answer is not a probe.
    """
    from domains.wind_tunnel.dataset_generate import _injection

    target = str(probe["target"])
    groups = _groups(chain)
    index = next((i for i, group in enumerate(groups) if target in group), None)
    if index is None:
        raise KeyError(f"{chain['process']} 没有目标题号 {target}")
    needed = [blank_id for group in groups[:index] for blank_id in group]
    missing = [blank_id for blank_id in needed if blank_id not in memory]
    if missing:
        raise KeyError(f"{probe['probe_id']} 的目标分支缺少这些空的答案：{missing}")
    prompt = _injection(
        chain["declaration"],
        chain["process"],
        chain["by_id"],
        target,
        phrases=chain.get("phrases"),
        memory=memory,
        keep_suffix=False,
        template=chain.get("process_text"),
    )
    if f"{{ID:{target}," not in prompt:
        raise RuntimeError(f"{probe['probe_id']} 的目标分支没有留下 ID{target} 的待填槽。")
    if "【探针】" in prompt:
        raise RuntimeError(f"{probe['probe_id']} 的目标分支混入了探针题。")
    process = prompt.split("【过程】", 1)[-1].split("【输出】", 1)[0]
    if index + 1 < len(groups) and f"{{ID:{groups[index + 1][0]}," in process:
        raise RuntimeError(f"{probe['probe_id']} 的目标分支没有在目标句结束。")
    return prompt


def build_prompt(chain: dict, probe: dict, condition: str, memory: dict[str, str], phrases: dict[str, str] | None = None) -> str:
    """One isolated probe call. The returned text is the whole model input."""
    if probe["target"] in EXCLUDED_TARGETS:
        raise RuntimeError(f"ID{probe['target']} 不作为这组契约探针的目标。")
    anchor = probe["after"]
    prefix = filled_prefix(chain, anchor, memory)
    question = _question(probe, condition, phrases or {})
    if question in prefix:
        raise RuntimeError(f"{probe['probe_id']} 的探针题干写进了流程前缀。")
    hidden = probe.get("hidden_stem") or ""
    if hidden and hidden in prefix:
        raise RuntimeError(f"{probe['probe_id']} 的前缀含有应隐藏的题干：{hidden}")
    prior = chain.get("prior") or PRIOR
    return "\n".join([
        "这是一次旁路探针。只回答下面这一个问题。",
        "不要填写过程中的空，不要继续后面的句子。",
        "这个答案只作记录，不会写回原流程。",
        '只输出一个 JSON 对象 {"probe":"答案"}。',
        "",
        "【定义声明】",
        chain["declaration"],
        "",
        "【物理常识】",
        prior,
        "",
        "【子过程】",
        chain["process"],
        "",
        "【过程】",
        prefix,
        "",
        "【探针】",
        question,
        "",
        "【输出】",
        '{"probe":"答案"}',
    ])


def prompt_id(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _cell_sign(matrix_text: str, row: int, col: int) -> str:
    import sympy

    from cloze_score import _parse_expr, _sympy_modules

    _sympy, locals_map = _sympy_modules()
    parsed = _parse_expr(matrix_text, sympy, locals_map)
    expr = sympy.simplify(parsed[row - 1, col - 1])
    sins = list(expr.atoms(sympy.sin))
    if len(sins) != 1:
        raise RuntimeError(f"矩阵第{row}行第{col}列不是单个 sin 项：{expr}")
    coeff = sympy.simplify(expr / sins[0])
    if coeff == 1:
        return "正"
    if coeff == -1:
        return "负"
    raise RuntimeError(f"sin 项系数不是 ±1：{expr}")


def _parsed_expr(text: str):
    import sympy

    from cloze_score import _parse_expr, _sympy_modules

    _sympy, locals_map = _sympy_modules()
    return sympy.simplify(_parse_expr(text, sympy, locals_map))


def _entry_sign(matrix_text: str, row: int, col: int) -> str:
    expr = _parsed_expr(matrix_text)[row - 1, col - 1]
    if expr == 1:
        return "正"
    if expr == -1:
        return "负"
    raise RuntimeError(f"矩阵第{row}行第{col}列不是 ±1：{expr}")


def _leading_sign(text: str) -> str:
    expr = _parsed_expr(text)
    return "负" if expr.could_extract_minus_sign() else "正"


def contract_key(probe: dict, gold_text: str) -> dict:
    """The local contract value a probe is scored against. It comes from the manuscript."""
    if probe["kind"] == "direction":
        return {"text": gold_text, "label": _direction_label(gold_text)}
    if probe["kind"] == "cells":
        signs = {f"{row},{col}": _cell_sign(gold_text, row, col) for row, col in probe["cells"]}
        return {"text": gold_text, "signs": signs}
    if probe["kind"] == "entry":
        row, col = probe["entry"]
        return {"text": gold_text, "label": _entry_sign(gold_text, row, col)}
    if probe["kind"] == "leading":
        return {"text": gold_text, "label": _leading_sign(gold_text)}
    if probe["kind"] == "sign":
        label = "负" if "负" in normalize(gold_text) else "正" if "正" in normalize(gold_text) else ""
        return {"text": gold_text, "label": label}
    if probe["kind"] == "placement":
        return {"text": "外侧", "label": "外侧"}
    if probe["kind"] == "expression":
        return {"text": gold_text}
    raise KeyError(probe["kind"])


def recognition_text(probe: dict, phrase_for) -> str:
    """Text the probe is judged against. Cell probes use the target matrix."""
    if probe["kind"] == "placement":
        return "外侧"
    if probe["kind"] in {"sign", "direction"}:
        return phrase_for(probe["source_id"])
    return phrase_for(probe["target"])


def _direction_label(text: str) -> str:
    raw = normalize(text)
    if any(token in raw for token in ("逆", "反向", "负向")):
        return "反向"
    if any(token in raw for token in ("正向", "顺")) or raw in {"正", "+"}:
        return "正向"
    return ""


def parse_probe(probe: dict, response: str) -> dict:
    body = response.strip()
    match = re.search(r'"probe"\s*:\s*"(.*?)"', body, flags=re.S)
    if match:
        body = match.group(1).strip()
    if probe["kind"] == "direction":
        label = _direction_label(body)
        return {"text": body, "label": label, "parsed": bool(label)}
    if probe["kind"] == "cells":
        signs = {}
        for row, col in probe["cells"]:
            found = re.search(
                rf"(?:\({row}\s*,\s*{col}\)|第{row}行第{col}列|{row}\s*行\s*{col}\s*列)"
                r"\s*[=:：]?\s*(正号|负号|正向|反向|正|负|\+|−|-)",
                body,
            )
            if not found:
                signs[f"{row},{col}"] = ""
                continue
            token = found.group(1)
            signs[f"{row},{col}"] = "负" if token in {"负", "负号", "-", "−", "反向"} else "正"
        if not any(signs.values()):
            parts = re.findall(r"负号|正号|负|正", body)
            if len(parts) == len(probe["cells"]):
                for (row, col), token in zip(probe["cells"], parts):
                    signs[f"{row},{col}"] = "负" if "负" in token else "正"
        return {"text": body, "signs": signs, "parsed": all(signs.values())}
    if probe["kind"] in {"sign", "entry", "leading"}:
        negative = "负" in body
        positive = "正" in body
        label = "负" if negative and not positive else "正" if positive and not negative else ""
        return {"text": body, "label": label, "parsed": bool(label)}
    if probe["kind"] == "placement":
        outside = "外侧" in body
        inside = "内部" in body
        label = "外侧" if outside and not inside else "内部" if inside and not outside else ""
        return {"text": body, "label": label, "parsed": bool(label)}
    return {"text": body, "parsed": bool(body)}


def same_contract(probe: dict, parsed: dict, contract: dict) -> bool:
    if not parsed.get("parsed"):
        return False
    if probe["kind"] in {"direction", "sign", "entry", "leading", "placement"}:
        return parsed.get("label") == contract.get("label") and bool(parsed.get("label"))
    if probe["kind"] == "cells":
        return parsed.get("signs") == contract.get("signs")
    return sympy_equal(parsed.get("text") or "", contract.get("text") or "") or normalize(parsed.get("text") or "") == normalize(contract.get("text") or "")


def execution_class(prediction: str, contract_text: str, standard_text: str) -> str:
    """contract, standard, both, or neither. Algebraic equality counts."""
    hit_contract = sympy_equal(prediction, contract_text) or normalize(prediction) == normalize(contract_text)
    hit_standard = sympy_equal(prediction, standard_text) or normalize(prediction) == normalize(standard_text)
    if hit_contract and hit_standard:
        return "both"
    if hit_contract:
        return "contract"
    if hit_standard:
        return "standard"
    return "neither"


def classify(facts: dict) -> list[str]:
    """Labels from the side-probe protocol. A row may carry more than one."""
    labels = []
    deep_b = facts["deep_b_hit"]
    deep_gold = facts["deep_gold_hit"]
    target_b = facts["target_b"]
    target_gold = facts["target_gold_run"]
    deep_b_standard = facts["deep_b_standard"]
    if deep_b and target_b in {"contract", "both"}:
        labels.append("契约识别能够转化为契约一致执行")
    if deep_b and target_b == "standard":
        labels.append("契约识别与执行不一致")
    if deep_b_standard and target_b == "standard":
        labels.append("探针表现为标准定义一致，尚不能区分契约丢失、模板和普通错误")
    if deep_b and target_b == "neither":
        labels.append("能识别契约，但执行计算失败")
    if (
        deep_b
        and deep_gold
        and target_b not in {"contract", "both"}
        and target_gold not in {"contract", "both"}
    ):
        labels.append("错误更可能位于执行或公式组合，而不是契约识别")
    if facts["deep_b_parsed"] and not deep_b and not deep_b_standard:
        labels.append("探针答案既不是契约也不是标准")
    if not facts["parents_correct"] and target_b not in {"contract", "both"}:
        labels.append("直接父节点存在错误，执行结果不能单独解释")
    return labels or ["未落入既定解释"]


_LEAKS = ("左手", "右手", "重新推导", "重新写出", "被定义为反", "{", "}")


def _chains() -> tuple[dict[str, dict], dict[str, dict]]:
    from cloze_eval import contexts_for, load_samples
    from cloze_eval import ROOT as root

    wanted = {item["sample_id"] for item in PROBES} | {"standard-default"}
    rows = {row["id"]: row for row in load_samples(root / "data" / "wind_tunnel" / "pair" / "samples.jsonl", wanted)}
    chains = {}
    standard = {}
    for probe in PROBES:
        key = (probe["sample_id"], probe["process"])
        if key not in chains:
            row = rows[probe["sample_id"]]
            chains[key] = next(item for item in contexts_for(row) if item["process"] == probe["process"])
        if probe["process"] not in standard:
            row = rows["standard-default"]
            standard[probe["process"]] = next(item for item in contexts_for(row) if item["process"] == probe["process"])
    return chains, standard


def _blank_map(chain: dict) -> dict[str, dict]:
    found = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            found[str(blank["blank_id"])] = blank
    return found


def _gold_memory(chain: dict) -> dict[str, str]:
    from cloze_eval import _repair_phrase

    memory = {}
    for sentence in chain["sentences"]:
        for blank in sentence["blanks"]:
            memory[str(blank["blank_id"])] = _repair_phrase(blank)
    return memory


def check_probes() -> None:
    """Build every probe on the manuscript and reject leaked or full-answer questions."""
    chains, standard_chains = _chains()
    for probe in PROBES:
        chain = chains[(probe["sample_id"], probe["process"])]
        blanks = _blank_map(chain)
        standard_blanks = _blank_map(standard_chains[probe["process"]])
        memory = _gold_memory(chain)
        for condition in ("deep_b",):
            question = _question(probe, condition, {})
            for leak in _LEAKS:
                if leak in question:
                    raise RuntimeError(f"{probe['probe_id']} 的题干含有 {leak}")
            if "不要写出完整" not in question:
                raise RuntimeError(f"{probe['probe_id']} 没有禁止写出完整目标")
            prompt = build_prompt(chain, probe, condition, memory, {})
            target_prompt = build_target_prompt(chain, probe, memory)
            if f"{{ID:{probe['target']}," not in target_prompt:
                raise RuntimeError(f"{probe['probe_id']} 的目标分支没有待填的目标空")
            if "不会写回原流程" not in prompt:
                raise RuntimeError(f"{probe['probe_id']} 没有声明答案不写回")
            if prompt.index("【探针】") < prompt.index("【过程】"):
                raise RuntimeError(f"{probe['probe_id']} 的探针不在过程之后")
            prefix = prompt.split("【探针】", 1)[0]
            if f"@@{probe['target']}@@" in prefix:
                raise RuntimeError(f"{probe['probe_id']} 的前缀留着目标槽")
        contract = contract_key(probe, recognition_text(probe, lambda blank_id: memory[blank_id]))
        standard_memory = _gold_memory(standard_chains[probe["process"]])
        standard = contract_key(
            probe,
            recognition_text(probe, lambda blank_id, bank=standard_memory: bank[blank_id]),
        )
        if probe["kind"] in {"cells", "sign", "entry", "leading"} and contract == standard:
            raise RuntimeError(f"{probe['probe_id']} 的契约和标准定义没有可区分的局部答案")
        if probe["kind"] == "placement" and contract.get("label") != "外侧":
            raise RuntimeError(f"{probe['probe_id']} 的位置规则不是外侧")
        deep_ids = []
        for group in _groups(chain):
            deep_ids.extend(group)
            if probe["after"] in group:
                break
        for parent in probe["repair_parent"] + probe["repair_other"]:
            if parent not in deep_ids:
                raise RuntimeError(f"{probe['probe_id']} 的修复点 ID{parent} 不在探针前缀里")
        if probe["kind"] == "sign" and probe["source_id"] in deep_ids:
            raise RuntimeError(f"{probe['probe_id']} 的探针前缀已经包含要测量的符号空")
        if probe["target"] not in blanks or probe["target"] not in standard_blanks:
            raise RuntimeError(f"{probe['probe_id']} 缺少目标空")
    labels = " ".join(classify({
        "deep_b_hit": True,
        "deep_gold_hit": True,
        "deep_b_standard": False,
        "deep_b_parsed": True,
        "target_b": "standard",
        "target_gold_run": "contract",
        "parents_correct": True,
    }))
    if "契约识别与执行不一致" not in labels or "回退" in labels:
        raise RuntimeError(f"探针解释标签不对：{labels}")
    print(f"probe check ok: {len(PROBES)} side probes")


if __name__ == "__main__":
    check_probes()

