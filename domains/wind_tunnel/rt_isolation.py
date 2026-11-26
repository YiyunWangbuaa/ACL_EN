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
from domains.wind_tunnel.dataset_generate import PROCESS_OFFSET, PROCESS_TEXT, SLOT_MARK, _period_index

PROCESS = "wind frame to body-axis frame"
RT_ID = "55"
DEEP_ANCHOR = "51"
PLACEHOLDER = "(this component-matrix answer is hidden)"
PROBE_MARK = "Probe record "
# The name is stated before the matrix blank. Shallow probes stop at that sentence.
MATRICES = (
    {"key": "rp1", "name": "RP1", "blank": "8", "shallow_anchor": "7", "named": "is therefore RP1"},
    {"key": "rp2", "name": "RP2", "blank": "13", "shallow_anchor": "12", "named": "pre-transform is RP2"},
    {"key": "alpha", "name": "R_α_M", "blank": "26", "shallow_anchor": "25", "named": "rotation matrix is R_α_M"},
    {"key": "beta", "name": "R_β_M", "blank": "34", "shallow_anchor": "33", "named": "rotation matrix is R_β_M"},
    {"key": "gamma", "name": "R_γ_M", "blank": "43", "shallow_anchor": "42", "named": "rotation matrix is R_γ_M"},
)
COMPONENT_IDS = tuple(item["blank"] for item in MATRICES)
CONTROL_LEAD = "The control writes the probe answer into the prefix. The probe record for R_γ_M is "
# Same matrix wording as dataset_generate.MATRIX_FORMAT, plus the worked example.
MATRIX_FORMAT = (
    "Write one 3×3 matrix on a single line, as [[a, b, c], [d, e, f], [g, h, i]]. "
    "Put exactly one space after each comma. Each entry is an integer or an expression in cos, sin, and the angle symbols of this item. Use * for multiplication. Do not break the line and do not use LaTeX. "
    "Example: [[cos(θ), 0, -sin(θ)], [0, 1, 0], [sin(θ), 0, cos(θ)]]. The example is not the answer to this item."
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
            raise KeyError(f"The frozen prefix is missing component matrix ID{blank_id}")
        hidden[blank_id] = PLACEHOLDER
    return hidden


def _question(name: str) -> str:
    return f"Write the expansion of {name}. {MATRIX_FORMAT}"


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
        raise KeyError(f"the problem text does not contain @@{anchor}@@")
    tail = template[found.end():]
    period = _period_index(tail)
    if period < 0:
        raise RuntimeError(f"There is no period after ID{anchor}")
    rest = tail[period + 1:].lstrip()
    extra: list[str] = []
    while rest:
        end = _period_index(rest)
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
        raise RuntimeError("The probe stem was written into the process prefix")
    if PROBE_MARK in prefix or PROBE_MARK in question:
        raise RuntimeError("The probe prefix already contains a probe record")
    return "\n".join([
        "This is a side probe. Answer only the question below.",
        "Do not fill blanks in the process, and do not continue the later sentences.",
        "This answer is only a score record. It is not written back to the original flow, and it does not enter R_t.",
        "",
        "[Declaration]",
        chain["declaration"],
        "",
        "[Process]",
        prefix,
        "",
        "[Probe]",
        question,
    ])


def _rt_prompt(chain: dict, memory: dict[str, str]) -> str:
    return prompt_for(chain, [RT_ID], "self-trunc", _copy(memory))


def _inject_control(prompt: str, probe_answer: str) -> str:
    sentence = f"{CONTROL_LEAD}{probe_answer}. "
    needle = "Write the symbolically simplified product R_t = R1*R2*R3"
    if needle not in prompt:
        raise RuntimeError("The R_t prompt has no total-matrix question, so the control sentence cannot be inserted")
    return prompt.replace(needle, sentence + needle, 1)


def build_conditions(chain: dict, frozen: dict[str, str], held: dict[str, dict[str, str]]) -> dict[str, str]:
    """frozen is fixed before any probe. held answers are never copied into the main prompts."""
    if chain["process"] != PROCESS:
        raise RuntimeError(f"This set of conditions is only used for {PROCESS}")
    required = set(COMPONENT_IDS) | {item["shallow_anchor"] for item in MATRICES} | {DEEP_ANCHOR, RT_ID}
    missing = sorted(required - set(map(str, frozen)))
    if missing:
        raise KeyError("The frozen prefix is missing " + ", ".join(missing))
    answers = []
    for item in MATRICES:
        pair = held[item["key"]]
        if PROBE_MARK not in pair["shallow"] or PROBE_MARK not in pair["deep"]:
            raise RuntimeError(f"The probe answer for {item['name']} must carry the record mark")
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
        raise RuntimeError("The component-matrix gold is empty, so the masking check has nothing to test")
    rt_answer = _compact(frozen[RT_ID])

    def compact_has(name: str, part: str) -> bool:
        return bool(part) and part in _compact(prompts[name])

    probe_names = [
        key for key in prompts
        if key.startswith("shallow_") or key.startswith("deep_")
    ]
    for name in probe_names:
        if any(answer in prompts[name] for answer in answers):
            raise RuntimeError(f"{name} saw a probe answer")
        if rt_answer and rt_answer in _compact(prompts[name]):
            raise RuntimeError(f"{name} saw the R_t answer")
    for item in MATRICES:
        shallow = f"shallow_{item['key']}"
        deep_original = f"deep_original_{item['key']}"
        deep_rederive = f"deep_rederive_{item['key']}"
        question = _question(item["name"])
        if not question.startswith(f"Write the expansion of {item['name']}.") or MATRIX_FORMAT not in question:
            raise RuntimeError(f"The {item['name']} probe does not ask for the expansion, or it omits the matrix format")
        if "re-derive" in question or "write it again" in question:
            raise RuntimeError(f"The {item['name']} probe still asks to re-derive the matrix")
        if item["named"] not in prompts[shallow]:
            raise RuntimeError(f"The {item['name']} shallow probe has not seen the matrix name")
        if compact_has(shallow, gold[item["blank"]]):
            raise RuntimeError(f"The {item['name']} shallow probe saw its own original answer")
        if not compact_has(deep_original, gold[item["blank"]]):
            raise RuntimeError(f"The {item['name']} original-flow deep probe should see the original answer")
        if compact_has(deep_rederive, gold[item["blank"]]):
            raise RuntimeError(f"The {item['name']} rederive probe still sees the original answer")
        if item["named"] not in prompts[deep_rederive]:
            raise RuntimeError(f"The {item['name']} rederive probe does not even contain the name")

    for name in ("rt_original", "rt_rederive"):
        if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
            raise RuntimeError(f"{name} saw a probe answer")
    if not all(compact_has("rt_original", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("R_t-original should keep the component matrices from the original flow")
    if any(compact_has("rt_rederive", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("R_t-rederive still sees a component-matrix answer")
    if prompts["rt_original"] == prompts["rt_rederive"]:
        raise RuntimeError("The two R_t conditions have the same prompt and cannot be scored apart")

    gamma_deep = next(answer for answer in answers if answer.endswith("deep R_γ_M"))
    if gamma_deep not in prompts["rt_writeback_control"]:
        raise RuntimeError("The write-back control did not include the deep R_γ_M probe answer")
    if any(answer in prompts["rt_writeback_control"] for answer in answers if answer != gamma_deep):
        raise RuntimeError("The write-back control included a probe answer that should not be written")
    if not all(compact_has("rt_writeback_control", gold[item["blank"]]) for item in MATRICES):
        raise RuntimeError("The write-back control should append the probe answer to the original prefix, not switch to rederive")


STAGES = (
    {
        "process": "balance frame to body-axis frame",
        "deep_anchor": "100",
        "total_id": "104",
        "matrices": (
            {"key": "rtp_tz1", "name": "RTP_TZ1", "blank": "95", "shallow_anchor": "94", "named": "is therefore RTP_TZ1"},
            {"key": "roll", "name": "R_(γ_M-γ_tp)", "blank": "88", "shallow_anchor": "87", "named": "rotation matrix is R_(γ_M-γ_tp)"},
            {"key": "rtp_tz2", "name": "RTP_TZ2", "blank": "97", "shallow_anchor": "88", "named": "RTP_TZ2 is also required"},
        ),
    },
    {
        "process": "balance frame to wind frame",
        "deep_anchor": "134",
        "total_id": "",
        "matrices": (
            {"key": "rp_tp_fd", "name": "RP_TP_FD", "blank": "126", "shallow_anchor": "125", "named": "transform matrix is RP_TP_FD"},
            {"key": "gamma_tp", "name": "R_γ_tp", "blank": "132", "shallow_anchor": "131", "named": "rotation matrix is R_γ_tp"},
        ),
    },
)


def isolate_stage(chain: dict, frozen: dict[str, str], stage: dict) -> dict[str, str]:
    """Shallow and deep probes for one later stage. Probe answers stay out of the total, when there is one."""
    if chain["process"] != stage["process"]:
        raise RuntimeError(f"The probe stage does not match the process: {chain['process']}")
    matrices = stage["matrices"]
    hide_ids = tuple(item["blank"] for item in matrices)
    required = set(hide_ids) | {item["shallow_anchor"] for item in matrices} | {stage["deep_anchor"]}
    total_id = stage.get("total_id") or ""
    if total_id:
        required.add(total_id)
    missing = sorted(required - set(map(str, frozen)))
    if missing:
        raise KeyError("The frozen prefix is missing " + ", ".join(missing))
    answers = [f"{PROBE_MARK}shallow {item['name']}" for item in matrices]
    answers += [f"{PROBE_MARK}deep {item['name']}" for item in matrices]
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
        raise RuntimeError("The component-matrix gold is empty, so the masking check has nothing to test")
    total_answer = _compact(frozen[total_id]) if total_id else ""

    def compact_has(name: str, part: str) -> bool:
        return bool(part) and part in _compact(prompts[name])

    probe_names = [key for key in prompts if key.startswith("shallow_") or key.startswith("deep_")]
    for name in probe_names:
        if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
            raise RuntimeError(f"{name} saw a probe answer")
        if total_answer and total_answer in _compact(prompts[name]):
            raise RuntimeError(f"{name} saw the total-matrix answer")
    for item in matrices:
        shallow = f"shallow_{item['key']}"
        question = _question(item["name"])
        if not question.startswith(f"Write the expansion of {item['name']}.") or MATRIX_FORMAT not in prompts[shallow]:
            raise RuntimeError(f"The {item['name']} probe does not ask for the expansion, or it omits the matrix format")
        if item["named"] not in prompts[shallow]:
            raise RuntimeError(f"The {item['name']} shallow probe has not seen the matrix name")
        if compact_has(shallow, gold[item["blank"]]):
            raise RuntimeError(f"The {item['name']} shallow probe saw its own original answer")
        if not compact_has(f"deep_original_{item['key']}", gold[item["blank"]]):
            raise RuntimeError(f"The {item['name']} original-flow deep probe should see the original answer")
        if any(compact_has(f"deep_rederive_{item['key']}", gold[other["blank"]]) for other in matrices):
            raise RuntimeError(f"The {item['name']} rederive probe still sees a component-matrix answer")
    if total_id:
        for name in ("total_original", "total_rederive"):
            if any(answer in prompts[name] for answer in answers) or PROBE_MARK in prompts[name]:
                raise RuntimeError(f"{name} saw a probe answer")
        if not all(compact_has("total_original", gold[item["blank"]]) for item in matrices):
            raise RuntimeError("The original-flow total matrix should keep the component matrices")
        if any(compact_has("total_rederive", gold[item["blank"]]) for item in matrices):
            raise RuntimeError("The rederive total matrix still sees a component-matrix answer")
        if prompts["total_original"] == prompts["total_rederive"]:
            raise RuntimeError("The two total-matrix conditions have the same prompt and cannot be scored apart")

