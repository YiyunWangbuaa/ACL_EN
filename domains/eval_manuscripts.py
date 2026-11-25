# -*- coding: utf-8 -*-
"""Load the four-domain full manuscripts into cloze chains.

Wind tunnel stays on its batch jsonl. Flight, vision and robotics are the
review manuscripts: each ### block is one subprocess, sent on its own.
Sentences are the stretches between periods. A sentence may hold more than
one slot. Slots keep their manuscript id inside a subprocess stage.
"""
from __future__ import annotations

import re
from pathlib import Path

from domain_layout import EVAL
from domains.aerodynamics.pipeline_scope import LAST_ID as FLIGHT_LAST
from domains.computer_vision.pipeline_scope import LAST_ID as THEORY_LAST
from domains.computer_vision.scene_pipeline_scope import LAST_ID as SCENE_LAST
from domains.robotics.kin_pipeline_scope import FORWARD_IDS, INVERSE_IDS

SLOT = re.compile(
    r"\{ID:(\d+), 选项:(.*?), 输出格式:(.*?), 标签:(.*?), 答案:(.*?)\}",
    re.S,
)
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
_SAMPLE_ID = re.compile(
    r"^# 数据[ \t\u3000]+(\d+)(?:[ \t\u3000]+(\d{3}-\d{3}))?",
    re.M,
)


def load_rows(domain: str, path: Path, wanted: list[str] | None) -> list[dict]:
    """Every sample under the domain entry, or the ids in `wanted`."""
    if domain not in {"flight", "cv", "robotics"}:
        raise RuntimeError(f"{domain} 不从审阅稿读取。")
    files = _files(path)
    rows = [_load_file(domain, item) for item in files]
    found = {row["id"] for row in rows}
    if len(found) != len(rows):
        raise RuntimeError(f"{path} 里有重复的样本编号。")
    if wanted is None:
        rows.sort(key=lambda row: row["id"])
        return rows
    missing = [item for item in wanted if item not in found]
    if missing:
        raise RuntimeError("样本里没有这些编号：" + "、".join(missing))
    order = {item: index for index, item in enumerate(wanted)}
    picked = [row for row in rows if row["id"] in order]
    picked.sort(key=lambda row: order[row["id"]])
    return picked


def manuscript_standard(domain: str) -> dict[tuple[str, str], str]:
    """Unperturbed answers keyed by (subprocess, blank id)."""
    if domain == "flight":
        return _answers([_load_file("flight", _file_with_id("flight", "01"))])
    if domain == "robotics":
        return _answers([_load_file("robotics", _file_with_id("robotics", "01"))])
    if domain == "cv":
        return _answers([_load_file("cv", _file_with_id("cv", "01"))])
    raise RuntimeError(f"{domain} 没有审阅稿标准答案。")


def render_sentence(sentence: dict, fill: bool, memory: dict[str, str] | None = None) -> str:
    """Open the current slots, or write an earlier slot as its surface phrase."""
    from domains.wind_tunnel.dataset_generate import _gold_phrase, _open_slot

    text = sentence["source"]
    spans = list(SLOT.finditer(text))
    if len(spans) != len(sentence["blanks"]):
        raise RuntimeError("句子里的槽和解析结果对不上。")
    parts: list[str] = []
    cursor = 0
    for match, blank in zip(spans, sentence["blanks"]):
        parts.append(text[cursor:match.start()])
        item = {
            "id": str(blank["blank_id"]),
            "answer": blank.get("answer") or "",
            "options": list(blank.get("options") or []),
            "format_code": blank.get("format_code") or "",
        }
        if fill:
            parts.append(_phrase(item, memory, _gold_phrase))
        else:
            parts.append(_open_slot(item))
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts).strip()


def process_text(
    chain: dict,
    blank_id: str,
    keep_suffix: bool,
    memory: dict[str, str] | None,
) -> str:
    """Subprocess text through the current sentence. Earlier slots are filled."""
    target = str(blank_id)
    chunks: list[str] = []
    reached = False
    for sentence in chain["sentences"]:
        hit = target in {str(item) for item in sentence["blank_ids"]}
        if reached:
            if not keep_suffix:
                break
            chunks.append(render_sentence(sentence, fill=False))
            continue
        chunks.append(render_sentence(sentence, fill=not hit, memory=memory))
        if hit:
            reached = True
    if not reached:
        raise RuntimeError(f"{chain['process']} 没有空 {target}。")
    return "\n".join(chunks)


def _phrase(item: dict, memory: dict[str, str] | None, gold) -> str:
    blank_id = str(item["id"])
    if memory is not None and memory.get(blank_id, "").strip():
        return memory[blank_id]
    return gold(item)


def _files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    found = sorted(path.glob("sample_*.md"))
    if not found:
        raise RuntimeError(f"没有审阅稿：{path}")
    return found


def _load_file(domain: str, path: Path) -> dict:
    text = _cut(path.read_text(encoding="utf-8"))
    sample_id, tier = _identity(text)
    format_note = _section(text, "输出格式规范")
    if domain == "flight":
        chains = _flight_chains(text, sample_id, tier, format_note)
        _expect(chains, "", FLIGHT_LAST, path)
    elif domain == "cv":
        chains = _vision_chains(text, sample_id, tier, format_note)
        if any(chain["process"].startswith("基础理论/") for chain in chains):
            _expect(chains, "基础理论/", THEORY_LAST, path)
        _expect(chains, "物体与双目/", SCENE_LAST, path)
    elif domain == "robotics":
        chains = _robotics_chains(text, sample_id, tier, format_note)
        _expect(chains, "正运动学/", len(FORWARD_IDS), path)
        _expect(chains, "逆运动学/", len(INVERSE_IDS), path)
    else:
        raise RuntimeError(domain)
    return {
        "id": sample_id,
        "tier": tier,
        "manuscript": True,
        "domain": domain,
        "chains": chains,
        "questions": _questions(chains),
    }


def _flight_chains(text: str, sample_id: str, tier: int, format_note: str) -> list[dict]:
    declaration = _section(text, "定义声明")
    prior = _section(text, "物理常识")
    aero = _section(text, "气动力与参考点定义")
    if not declaration or not aero:
        raise RuntimeError(f"{sample_id} 缺少定义声明或气动力定义。")
    chains = []
    in_aero = False
    for block in _blocks(text):
        if block["level"] == 1 and block["title"].startswith("生成注记"):
            break
        if block["level"] == 2 and block["title"] == "气动力与参考点定义":
            in_aero = True
            continue
        if block["level"] != 3:
            continue
        if in_aero:
            chains.append(_chain(sample_id, tier, block["title"], aero, "", format_note, block["body"]))
        else:
            chains.append(_chain(sample_id, tier, block["title"], declaration, prior, format_note, block["body"]))
    if len(chains) != 6:
        raise RuntimeError(f"{sample_id} 应有 6 个子过程，实际 {len(chains)}。")
    return chains


def _vision_chains(text: str, sample_id: str, tier: int, format_note: str) -> list[dict]:
    if "# 物体与双目" not in text:
        raise RuntimeError(f"{sample_id} 缺少物体与双目。")
    scene = _stage_chains(_between(text, "# 物体与双目", None), "物体与双目", sample_id, tier, format_note)
    if "# 前置环节：基础理论" not in text:
        return scene
    theory = _stage_chains(
        _between(text, "# 前置环节：基础理论", "# 物体与双目"),
        "基础理论",
        sample_id,
        tier,
        format_note,
    )
    return theory + scene


def _robotics_chains(text: str, sample_id: str, tier: int, format_note: str) -> list[dict]:
    forward = _between(text, "# 正运动学样本", "# 逆运动学样本")
    inverse = _between(text, "# 逆运动学样本", "# 层间对应")
    return (
        _stage_chains(forward, "正运动学", sample_id, tier, format_note)
        + _stage_chains(inverse, "逆运动学", sample_id, tier, format_note)
    )


def _stage_chains(
    text: str,
    stage: str,
    sample_id: str,
    tier: int,
    format_note: str,
) -> list[dict]:
    declaration = _section(text, "定义声明")
    prior = _section(text, "物理常识")
    if not declaration:
        raise RuntimeError(f"{sample_id} {stage} 没有定义声明。")
    chains = []
    for block in _blocks(text):
        if block["level"] == 1:
            break
        if block["level"] == 3:
            chains.append(_chain(
                sample_id, tier, f"{stage}/{block['title']}",
                declaration, prior, format_note, block["body"],
            ))
    if not chains:
        raise RuntimeError(f"{sample_id} {stage} 没有子过程。")
    return chains


def _chain(
    sample_id: str,
    tier: int,
    process: str,
    declaration: str,
    prior: str,
    format_note: str,
    body: str,
) -> dict:
    sentences = _sentences(body)
    if not sentences:
        raise RuntimeError(f"{sample_id} {process} 没有待填槽。")
    return {
        "sample_id": sample_id,
        "tier": tier,
        "process": process,
        "domain": "manuscript",
        "declaration": declaration.strip(),
        "prior": prior.strip(),
        "format_note": format_note.strip(),
        "sentences": sentences,
    }


def _sentences(body: str) -> list[dict]:
    """Group slots that share a sentence. A period inside a slot is not a boundary."""
    slots = list(SLOT.finditer(body))
    if not slots:
        return []
    groups: list[list[re.Match[str]]] = []
    current: list[re.Match[str]] = []
    cursor = 0
    for match in slots:
        if current and "。" in body[cursor:match.start()]:
            groups.append(current)
            current = []
        current.append(match)
        cursor = match.end()
    if current:
        groups.append(current)
    sentences = []
    prev_end = 0
    for group in groups:
        start, end = _span(body, group, prev_end)
        prev_end = end
        source = body[start:end].strip()
        blanks = []
        blank_ids = []
        for match in group:
            blank_id = match.group(1)
            blanks.append({
                "blank_id": blank_id,
                "label": match.group(4).strip(),
                "format_code": match.group(3).strip(),
                "answer": match.group(5).strip(),
                "options": _options(match.group(2)),
            })
            blank_ids.append(blank_id)
        sentences.append({"blank_ids": blank_ids, "blanks": blanks, "source": source})
    return sentences


def _span(body: str, group: list[re.Match[str]], prev_end: int) -> tuple[int, int]:
    first = group[0].start()
    last_period = body.rfind("。", prev_end, first)
    start = last_period + 1 if last_period != -1 else prev_end
    period = body.find("。", group[-1].end())
    end = period + 1 if period != -1 else len(body)
    return start, end


def _options(raw: str) -> list[str]:
    text = raw.strip()
    if not text or text == "无":
        return []
    found = []
    for part in text.split("；"):
        piece = part.strip()
        if not piece:
            continue
        if "：" in piece:
            found.append(piece.split("：", 1)[1].strip())
        elif ":" in piece:
            found.append(piece.split(":", 1)[1].strip())
        else:
            found.append(piece)
    return found


def _expect(chains: list[dict], prefix: str, last: int, path: Path) -> None:
    found = []
    for chain in chains:
        if prefix and not chain["process"].startswith(prefix):
            continue
        for sentence in chain["sentences"]:
            found.extend(str(item) for item in sentence["blank_ids"])
    expected = [str(number) for number in range(1, last + 1)]
    if found != expected:
        raise RuntimeError(
            f"{path.name} {prefix or '全文'} 空编号不是 1–{last}。"
            f"解析到 {len(found)} 个。"
        )


def _questions(chains: list[dict]) -> list[dict]:
    rows = []
    for chain in chains:
        for sentence in chain["sentences"]:
            for blank in sentence["blanks"]:
                rows.append({
                    "process": chain["process"],
                    "id": blank["blank_id"],
                    "local_id": blank["blank_id"],
                    "label": blank["label"],
                    "format_code": blank["format_code"],
                    "options": list(blank["options"]),
                    "answer": blank["answer"],
                })
    return rows


def _answers(rows: list[dict]) -> dict[tuple[str, str], str]:
    return _answers_from_chains([chain for row in rows for chain in row["chains"]])


def _answers_from_chains(chains: list[dict]) -> dict[tuple[str, str], str]:
    found = {}
    for chain in chains:
        for sentence in chain["sentences"]:
            for blank in sentence["blanks"]:
                found[(chain["process"], str(blank["blank_id"]))] = str(blank["answer"])
    return found


def _identity(text: str) -> tuple[str, int]:
    match = _SAMPLE_ID.search(text)
    if match is None:
        raise RuntimeError("审阅稿缺少“# 数据”标题。")
    if match.group(2):
        public = match.group(2)
        return public, int(public.split("-")[0])
    return match.group(1), 0


def _section(text: str, title: str) -> str:
    for block in _blocks(text):
        if block["level"] == 2 and block["title"] == title:
            return block["body"].strip()
    return ""


def _between(text: str, start: str, end: str | None) -> str:
    at = text.find(start)
    if at < 0:
        raise RuntimeError(f"审阅稿缺少 {start}")
    body = text[at + len(start):]
    if end is None:
        return body
    stop = body.find(end)
    if stop < 0:
        raise RuntimeError(f"审阅稿缺少 {end}")
    return body[:stop]


def _file_with_id(domain: str, sample_id: str) -> Path:
    for path in _files(EVAL[domain]["data"]):
        if _identity(path.read_text(encoding="utf-8"))[0] == sample_id:
            return path
    raise RuntimeError(f"{domain} 没有样本 {sample_id}。")


def _cut(text: str) -> str:
    at = text.find("\n# 生成注记")
    if at >= 0:
        text = text[:at]
    return text


def _blocks(text: str) -> list[dict]:
    matches = list(_HEADING.finditer(text))
    blocks = []
    for index, match in enumerate(matches):
        start = match.end()
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append({
            "level": len(match.group(1)),
            "title": match.group(2).strip(),
            "body": text[start:stop].strip(),
        })
    return blocks
