# -*- coding: utf-8 -*-
"""Draw the engineering-contract graph and open it in a browser.

Question text comes from data/wind_tunnel/pair/samples.jsonl.
The graph comes from data/wind_tunnel/dependency_graph.json.

    python analysis/contract_view.py
"""

from __future__ import annotations

import json
import sys
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cloze_score import choice_letter, resolve_choice
GRAPH_PATH = ROOT / "data" / "wind_tunnel" / "dependency_graph.json"
SAMPLE_PATH = ROOT / "data" / "wind_tunnel" / "pair" / "samples.jsonl"
OUT_PATH = ROOT / "results" / "contract_view" / "index.html"

_SLOT = __import__("re").compile(r"@@(\d+)@@")


def _relation_cards(records: list[dict]) -> list[dict]:
    """Keep every stored path in the graph file. The page shows a short sample."""
    cards = []
    current = None
    for item in records:
        key = (item["contract"], item["relation"])
        if current is None or current["key"] != key:
            current = {
                "key": key,
                "contract": item["contract"],
                "relation": item["relation"],
                "effect": item.get("effect", ""),
                "paths": [],
            }
            cards.append(current)
        if item["relation"] == "indirect":
            current["paths"].append({
                "via": [part.removeprefix("ID") for part in item["via"]],
                "types": list(item["contribution_types"]),
            })
    shown = []
    for card in cards:
        paths = card["paths"]
        shown.append({
            "contract": card["contract"],
            "relation": card["relation"],
            "effect": card["effect"],
            "count": len(paths) if card["relation"] == "indirect" else 1,
            "groups": _shortest_groups(paths),
        })
    return shown


def _shortest_groups(paths: list[dict]) -> list[dict]:
    """One group per direct target. All of its shortest paths stay visible."""
    grouped: dict[str, list[dict]] = {}
    for path in paths:
        grouped.setdefault(path["via"][0], []).append(path)
    groups = []
    for direct in sorted(grouped, key=int):
        routes = grouped[direct]
        shortest = min(len(path["via"]) for path in routes)
        short_paths = [path for path in routes if len(path["via"]) == shortest]
        short_paths.sort(key=lambda path: path["via"])
        groups.append({
            "direct": direct,
            "paths": short_paths,
            "longer": len(routes) - len(short_paths),
        })
    return groups


def _samples() -> list[dict]:
    rows = []
    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sentences(process_text: dict) -> dict[str, str]:
    found = {}
    for text in process_text.values():
        for block in __import__("re").split(r"\n\s*\n", str(text)):
            body = " ".join(block.split())
            if not body:
                continue
            for blank_id in _SLOT.findall(body):
                found[blank_id] = body
    return found


def _questions(record: dict) -> dict[str, dict]:
    sentences = _sentences(record.get("process_text") or {})
    packed = {}
    for item in record["questions"]:
        options = [str(option) for option in (item.get("options") or []) if option and str(option) != "无"]
        if options:
            letter, text = resolve_choice(str(item["answer"]), options)
        else:
            letter, text = "", str(item["answer"])
        blank_id = str(item["id"])
        packed[blank_id] = {
            "prompt": item["prompt"],
            "options": [{"letter": choice_letter(index), "text": option} for index, option in enumerate(options)],
            "letter": letter,
            "text": text,
            "format": item.get("format_code") or "无",
            "sentence": sentences.get(blank_id, ""),
        }
    return packed


def _blank_index(graph: dict) -> dict[str, dict]:
    found = {}
    for process in graph["processes"].values():
        for blank_id, item in process["blanks"].items():
            found[str(blank_id)] = item
    return found


def build_payload() -> dict:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    layer = graph["contract_layer"]
    blanks = _blank_index(graph)
    records = _samples()
    default = next(item for item in records if item["id"] == "standard-default")
    missing = [blank_id for blank_id in blanks if blank_id not in _questions(default)]
    if missing:
        raise RuntimeError(f"standard-default 缺少题目：{', '.join(missing)}")
    nodes = []
    for item in layer["instances"]:
        siblings = []
        for row in layer["globals"]:
            if row["id"] == item["global_id"]:
                siblings = [other for other in row["instances"] if other != item["id"]]
        nodes.append({
            "nid": item["id"],
            "kind": "atom",
            "process": item["process"],
            "title": item["name"],
            "subtitle": item["slot"] or item["global_id"],
            "tag": item["kind"],
            "status": item["status"],
            "globalId": item["global_id"],
            "standard": item["standard"],
            "perturbed": item["perturbed"],
            "slot": item["slot"],
            "direct": [value.removeprefix("ID") for value in item["direct_targets"]],
            "downstream": [value.removeprefix("ID") for value in item["downstream_targets"]],
            "indirectCount": item.get("indirect_path_count", 0),
            "siblings": siblings,
        })
    for blank_id, item in blanks.items():
        binding = layer["answer_bindings"].get(f"ID{blank_id}", {})
        nodes.append({
            "nid": f"ID{blank_id}",
            "kind": "answer",
            "process": item["process"],
            "title": f"ID{blank_id}",
            "subtitle": item["role"],
            "tag": item["label"],
            "order": int(blank_id),
            "label": item["label"],
            "role": item["role"],
            "dependsOn": list(item["depends_on"]),
            "children": list(item["direct_dependents"]),
            "bindings": list(binding.get("binding_rule_ids") or []),
            "atoms": list(binding.get("contract_targets") or []),
            "fields": list(binding.get("sensitive_fields") or []),
            "relations": _relation_cards(list(binding.get("contract_relations") or [])),
        })
    edges = []
    for item in layer["direct_edges"]:
        edges.append({
            "source": item["source"],
            "target": item["target"],
            "kind": "joint" if item["combination"] == "joint" else "single",
            "relation": item["relation"],
            "binding": item["binding"],
        })
    for blank_id, item in blanks.items():
        for parent in item["depends_on"]:
            edges.append({
                "source": f"ID{parent}",
                "target": f"ID{blank_id}",
                "kind": "answer",
                "relation": "depends_on",
                "binding": "",
            })
    return {
        "meaning": layer["meaning"],
        "notes": layer["id_notes"],
        "relations": layer["relation_types"],
        "samples": [
            {
                "id": item["id"],
                "title": item["title"],
                "condition": item["condition"],
                "declaration": item["declaration"],
                "excluded": [str(value) for value in item.get("excluded_ids") or []],
                "questions": _questions(item),
            }
            for item in records
        ],
        "nodes": nodes,
        "edges": edges,
        "bindings": [
            {
                "id": item["rule_id"],
                "process": item["process"],
                "chain": item["chain"],
                "relation": item["relation"],
                "parents": list(item["contract_parents"]),
                "context": list(item["fixed_context"]),
                "targets": [value.removeprefix("ID") for value in item["targets"]],
                "fields": list(item["sensitive_fields"]),
                "standard": item["standard_sensitive"],
                "note": item["note"],
            }
            for item in layer["contract_bindings"]
        ],
    }


_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>工程契约依赖图</title>
<style>
:root {
  --bg: #f3efe6;
  --ink: #1e1a16;
  --muted: #6d645b;
  --line: #ddd4c6;
  --panel: #fffdf8;
  --mark: #f3d3a1;
  --conv: #1d4e89;
  --axis: #0f6e56;
  --sign: #8a4b08;
  --install: #6b3fa0;
  --method: #1f4b99;
  --force: #8c2f39;
  --geom: #3f6212;
  --scope: #3f5c6e;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font: 14px/1.45 "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
}
header {
  padding: 14px 18px 10px;
  border-bottom: 1px solid var(--line);
  background: #faf7f1;
}
h1 { margin: 0 0 4px; font-size: 20px; font-weight: 650; }
.sub { margin: 0; color: var(--muted); font-size: 13px; }
.tabs, .tools { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }
button, select, input { font: inherit; color: inherit; }
.tab, .tool, button.node, .chip, .jump {
  border: 1px solid var(--line);
  background: #fff;
  border-radius: 999px;
  padding: 4px 10px;
  cursor: pointer;
}
.tab.on { background: var(--ink); color: #fff; border-color: var(--ink); }
.tools input[type="search"] { width: 220px; border-radius: 8px; padding: 5px 8px; border: 1px solid var(--line); background: #fff; }
.main { display: grid; grid-template-columns: minmax(0, 1fr) minmax(300px, 380px); height: calc(100vh - 118px); }
.board-wrap { overflow: auto; position: relative; }
#board { position: relative; }
svg.edges { position: absolute; inset: 0; width: 100%; height: 100%; }
button.node {
  position: absolute;
  display: flex; flex-direction: column; justify-content: center;
  text-align: left; border-radius: 8px; padding: 3px 8px;
  overflow: hidden; white-space: nowrap;
}
button.node .t { font-weight: 650; font-size: 12px; }
button.node .s { color: var(--muted); font-size: 11px; overflow: hidden; text-overflow: ellipsis; }
button.node.atom { background: #fff; border-left: 4px solid var(--method); }
button.node.atom.conv { border-left-color: var(--conv); }
button.node.atom.axis { border-left-color: var(--axis); }
button.node.atom.sign { border-left-color: var(--sign); }
button.node.atom.install { border-left-color: var(--install); }
button.node.atom.method { border-left-color: var(--method); }
button.node.atom.force { border-left-color: var(--force); }
button.node.atom.geom { border-left-color: var(--geom); }
button.node.atom.scope { border-left-color: var(--scope); }
button.node.factor { background: #fff7ed; border-color: #e7c8a2; }
button.node.answer { background: #f7f4ee; }
button.node.answer.C1, button.node.answer.C2 { background: #e7eef6; }
button.node.answer.A2 { background: #f8efe4; }
button.node.on { outline: 2px solid var(--ink); }
button.node.dim, svg .dim { opacity: 0.16; }
.colhead {
  position: absolute; top: 8px; font-size: 12px; color: var(--muted); font-weight: 650;
}
aside {
  border-left: 1px solid var(--line);
  background: var(--panel);
  overflow: auto;
  padding: 16px 16px 32px;
}
aside h2 { margin: 0 0 6px; font-size: 18px; }
aside h3 { margin: 16px 0 6px; font-size: 13px; color: var(--muted); font-weight: 650; }
.meta { color: var(--muted); font-size: 12px; }
.sentence { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; }
mark { background: var(--mark); padding: 0 2px; }
pre {
  white-space: pre-wrap; word-break: break-word;
  background: #f6f3ec; border-radius: 8px; padding: 8px; margin: 6px 0;
  font-family: Consolas, "Sarasa Mono SC", monospace; font-size: 12px;
}
.hit { display: block; width: 100%; text-align: left; margin: 0 0 6px; border-radius: 8px; }
.gold { font-weight: 650; }
.miss { color: var(--muted); }
.path { margin: 2px 0 2px 12px; font-size: 13px; }
ul.notes { margin: 8px 0 0; padding-left: 18px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>工程契约依赖图</h1>
  <p class="sub">契约原子直接连到答案空。联合条件写在绑定规则里，规则不是节点。点节点后，右侧显示当前稿件里的题目。</p>
  <div class="tabs" id="tabs"></div>
  <div class="tools">
    <label>稿件 <select id="sample"></select></label>
    <label>链 <select id="chain"></select></label>
    <label class="tool"><input id="answers" type="checkbox"> 答案依赖边</label>
    <input id="q" type="search" placeholder="搜索 ID、题干或契约名">
    <button id="zoom-out" type="button">缩小</button>
    <button id="zoom-in" type="button">放大</button>
  </div>
</header>
<div class="main">
  <div class="board-wrap" id="wrap"><div id="board"></div></div>
  <aside id="panel"></aside>
</div>
<script id="payload" type="application/json">/*__DATA__*/</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const byId = Object.fromEntries(DATA.nodes.map(node => [node.nid, node]));
const state = {
  process: "风轴系到体轴系",
  sample: "standard-default",
  chain: "",
  showAnswers: false,
  zoom: 1,
  selected: null,
  query: "",
};

function sample() {
  return DATA.samples.find(item => item.id === state.sample) || DATA.samples[0];
}
function question(blankId) {
  return (sample().questions || {})[String(blankId)] || null;
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[ch]);
}
function avg(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function spread(list, gap) {
  const items = [...list].sort((a, b) => a.y - b.y || (a.order || 0) - (b.order || 0));
  let last = -1e9;
  for (const node of items) {
    if (node.y < last + gap) node.y = last + gap;
    last = node.y;
  }
}
function processes() {
  const found = [];
  for (const node of DATA.nodes) {
    if (!found.includes(node.process)) found.push(node.process);
  }
  return found;
}
function visibleSet() {
  const nodes = DATA.nodes.filter(node => node.process === state.process);
  if (!state.chain) return new Set(nodes.map(node => node.nid));
  const rules = DATA.bindings.filter(rule => rule.process === state.process && rule.chain === state.chain);
  const keep = new Set();
  for (const rule of rules) {
    rule.parents.forEach(id => keep.add(id));
    rule.targets.forEach(id => keep.add("ID" + id));
  }
  return keep;
}
function layout() {
  const zoom = state.zoom;
  const pitch = 40 * zoom;
  const height = 32 * zoom;
  const wrapWidth = document.getElementById("wrap").clientWidth || 900;
  const usable = Math.max(240, wrapWidth - 8);
  const gap = 48;
  const col = (usable - gap) / 2;
  const x = { atom: 8, answer: 8 + col + gap };
  const w = { atom: col, answer: col };
  const nodes = DATA.nodes.filter(node => node.process === state.process);
  const answers = nodes.filter(node => node.kind === "answer").sort((a, b) => a.order - b.order);
  const atoms = nodes.filter(node => node.kind === "atom");
  const y0 = 42 * zoom;
  answers.forEach((node, index) => {
    node.x = x.answer; node.y = y0 + index * pitch; node.w = w.answer; node.h = height;
  });
  const answerY = Object.fromEntries(answers.map(node => [node.nid, node.y]));
  atoms.forEach(node => {
    const ys = [];
    for (const edge of DATA.edges) {
      if (edge.source === node.nid && edge.kind !== "answer" && answerY[edge.target] !== undefined) {
        ys.push(answerY[edge.target]);
      }
    }
    node.y = ys.length ? avg(ys) : y0;
    node.x = x.atom; node.w = w.atom; node.h = height;
  });
  spread(atoms, pitch);
  const bottom = Math.max(...nodes.map(node => node.y + node.h), 480);
  return {
    nodes,
    width: state.showAnswers ? usable + 120 : usable,
    height: bottom + 28,
    heads: [["契约原子", x.atom], ["答案空", x.answer]],
  };
}
function edgeEnds(edge, placed) {
  const source = placed.get(edge.source);
  const target = placed.get(edge.target);
  if (!source || !target) return null;
  if (edge.kind === "answer") {
    return [source.x + source.w, source.y + source.h / 2, target.x + target.w, target.y + target.h / 2];
  }
  return [source.x + source.w, source.y + source.h / 2, target.x, target.y + target.h / 2];
}
function edgePath(edge, ends) {
  const [x1, y1, x2, y2] = ends;
  if (edge.kind === "answer") {
    const bow = Math.max(x1, x2) + 28 + Math.min(90, Math.abs(y2 - y1) / 5);
    return `M ${x1} ${y1} C ${bow} ${y1}, ${bow} ${y2}, ${x2} ${y2}`;
  }
  const bend = Math.max(36, (x2 - x1) / 2);
  return `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`;
}
function matches(node) {
  const query = state.query.trim().toLowerCase();
  if (!query) return false;
  const bag = [node.nid, node.title, node.subtitle, node.role, node.globalId].filter(Boolean).join(" ").toLowerCase();
  if (bag.includes(query)) return true;
  if (node.kind === "answer") {
    const item = question(node.order);
    const text = item ? `${item.prompt} ${item.sentence} ${item.text}` : "";
    return text.toLowerCase().includes(query);
  }
  return false;
}
function renderBoard() {
  const board = document.getElementById("board");
  const drawn = layout();
  const keep = visibleSet();
  const placed = new Map(drawn.nodes.map(node => [node.nid, node]));
  const selected = state.selected;
  const incident = new Set();
  if (selected) {
    incident.add(selected);
    for (const edge of DATA.edges) {
      if (edge.kind === "answer" && !state.showAnswers) continue;
      if (edge.source === selected || edge.target === selected) {
        incident.add(edge.source);
        incident.add(edge.target);
      }
    }
  }
  const paths = [];
  for (const edge of DATA.edges) {
    if (edge.kind === "answer" && !state.showAnswers) continue;
    if (!keep.has(edge.source) || !keep.has(edge.target)) continue;
    const ends = edgeEnds(edge, placed);
    if (!ends) continue;
    const hot = selected && (edge.source === selected || edge.target === selected);
    const dim = selected && !hot;
    paths.push(`<path d="${edgePath(edge, ends)}" fill="none" stroke="${colorOf(edge)}" stroke-width="${hot ? 2.2 : 1.1}" class="${dim ? "dim" : ""}"/>`);
  }
  const heads = drawn.heads.map(([label, left]) => `<div class="colhead" style="left:${left}px">${label}</div>`).join("");
  const buttons = drawn.nodes.filter(node => keep.has(node.nid)).map(node => {
    const classes = ["node", node.kind, node.tag];
    if (node.nid === selected) classes.push("on");
    else if (selected && !incident.has(node.nid)) classes.push("dim");
    else if (matches(node)) classes.push("on");
    return `<button class="${classes.join(" ")}" data-nid="${escapeHtml(node.nid)}" style="left:${node.x}px;top:${node.y}px;width:${node.w}px;height:${node.h}px" title="${escapeHtml(node.subtitle || node.title)}"><span class="t">${escapeHtml(node.title)}</span><span class="s">${escapeHtml(node.subtitle || "")}</span></button>`;
  }).join("");
  board.style.width = drawn.width + "px";
  board.style.height = drawn.height + "px";
  board.innerHTML = `<svg class="edges" viewBox="0 0 ${drawn.width} ${drawn.height}">${paths.join("")}</svg>${heads}${buttons}`;
  board.querySelectorAll("button.node").forEach(button => {
    button.addEventListener("click", event => {
      event.stopPropagation();
      choose(button.dataset.nid);
    });
  });
}
function colorOf(edge) {
  if (edge.kind === "answer") return "#b7b1a6";
  if (edge.kind === "joint") return "#9a6b2f";
  if (edge.relation === "scope_constraint") return "#5c6b73";
  if (edge.relation === "direct_method") return "#1d4e89";
  return "#0f6e56";
}
function chip(nid, label) {
  return `<button class="chip" data-jump="${escapeHtml(nid)}" type="button">${escapeHtml(label || nid)}</button> `;
}
function sentenceBlock(blankId) {
  const item = question(blankId);
  if (!item || !item.sentence) return `<p class="miss">这份稿没有 ID${escapeHtml(blankId)}。</p>`;
  const html = escapeHtml(item.sentence).replaceAll(`@@${blankId}@@`, `<mark>@@${blankId}@@</mark>`);
  return `<div class="sentence">${html}</div>`;
}
function answerBody(blankId) {
  const item = question(blankId);
  const node = byId["ID" + blankId];
  if (!node) return "";
  const options = item && item.options.length
    ? item.options.map(option => `<div>${option.letter === item.letter ? "<b>" : ""}${escapeHtml(option.letter)}. ${escapeHtml(option.text)}${option.letter === item.letter ? "</b>" : ""}</div>`).join("")
    : "<p class='miss'>这道题没有选项，答案按输出格式书写。</p>";
  const gold = item ? (item.letter ? `${item.letter}. ${item.text}` : item.text) : "";
  return `
    <p class="meta">${escapeHtml(node.label)} · ${escapeHtml(node.role)} · 输出格式 ${escapeHtml(item ? item.format : "无")}</p>
    <h3>所在段落</h3>
    ${sentenceBlock(blankId)}
    <h3>题干</h3>
    <p>${item ? escapeHtml(item.prompt) : "这份稿没有这道题。"}</p>
    <h3>选项</h3>
    ${options}
    <h3>标准答案</h3>
    <pre class="gold">${escapeHtml(gold)}</pre>
    <h3>绑定规则</h3>
    <p>${(node.bindings || []).map(id => `<span class="chip">${escapeHtml(id)}</span>`).join(" ") || "<span class='miss'>这道题没有联合绑定规则。</span>"}</p>
    <p>${(node.fields || []).map(field => `<span class="chip">${escapeHtml(field)}</span>`).join(" ")}</p>
    <h3>直接作用到这道题的契约</h3>
    <p>${node.atoms.map(id => chip(id, byId[id] ? byId[id].title : id)).join("") || "<span class='miss'>没有契约直接作用到这道题。</span>"}</p>
    <h3>潜在契约依赖</h3>
    ${relationBlock(node)}
    <h3>答案图</h3>
    <p>父空 ${node.dependsOn.map(id => chip("ID" + id, "ID" + id)).join("") || "无"}</p>
    <p>子空 ${node.children.map(id => chip("ID" + id, "ID" + id)).join("") || "无"}</p>`;
}
function relationBlock(node) {
  const rows = node.relations || [];
  if (!rows.length) return `<p class="miss">没有契约路径到达这道题。</p>`;
  return `<p class="miss">这是图上的潜在路径。实际错误路径要等路径上的答案都错才成立，正确的中间节点会阻断串联。下面按直接作用点列出全部最短路径；更长的简单路径保存在数据文件里。</p>` + rows.map(row => {
    const title = byId[row.contract] ? byId[row.contract].title : row.contract;
    if (row.relation === "direct") {
      return `<p>${chip(row.contract, title)} <span class="chip">直接</span> ${escapeHtml(row.effect || "")}</p>`;
    }
    const lines = (row.groups || []).map(group => {
      const body = group.paths.map(path => {
        const hops = path.via.map(id => "ID" + id).concat(["ID" + node.order]);
        return `<div class="path">${escapeHtml(hops.join(" → "))} <span class="miss">${escapeHtml(path.types.join(" · "))}</span></div>`;
      }).join("");
      const more = group.longer
        ? `<div class="path miss">从 ID${group.direct} 另有 ${group.longer} 条更长路径。</div>`
        : "";
      return body + more;
    }).join("");
    return `<p>${chip(row.contract, title)} <span class="chip">间接 ${row.count}</span></p>${lines}`;
  }).join("");
}
function renderPanel() {
  const panel = document.getElementById("panel");
  const current = sample();
  const node = state.selected ? byId[state.selected] : null;
  if (!node) {
    panel.innerHTML = `<h2>点一个节点</h2>
      <p>左侧是契约原子，右侧是答案空。棕色边是同一条绑定规则里的联合作用，不是独立决定。答案来自稿件 <b>${escapeHtml(current.id)}</b>。</p>
      <p class="meta">${escapeHtml(DATA.meaning)}</p>
      <h3>编号</h3>
      <ul class="notes">${DATA.notes.map(note => `<li>${escapeHtml(note)}</li>`).join("")}</ul>
      <h3>本条定义声明</h3>
      <pre>${escapeHtml(current.declaration)}</pre>`;
    return;
  }
  let body = "";
  if (node.kind === "answer") body = answerBody(node.order);
  if (node.kind === "atom") {
    const rules = DATA.bindings.filter(rule => rule.parents.includes(node.nid));
    const statusName = node.status === "derived" ? "没有问答，字段可核对" : "有直接问答";
    body = `<p class="meta">${escapeHtml(node.nid)} · ${statusName} · 标准 ${escapeHtml(node.standard) || "固定"} · 扰动 ${escapeHtml(node.perturbed) || "当前不单独扰动"}</p>
      <h3>同一定义的其他读取</h3>
      <p>${node.siblings.map(id => chip(id, byId[id] ? `${byId[id].title} · ${byId[id].process}` : id)).join("") || "<span class='miss'>只在这个子过程进入评测图。</span>"}</p>
      <h3>参与的绑定规则</h3>
      ${rules.map(rule => `<p><b>${escapeHtml(rule.id)}</b> → ${rule.targets.map(id => "ID" + id).join("、")}<br><span class="miss">${escapeHtml(rule.note || "")} 字段：${escapeHtml(rule.fields.join("、"))}</span></p>`).join("") || "<p class='miss'>没有联合绑定，只有单独问答。</p>"}
      <h3>直接作用的题目</h3>
      ${node.direct.map(id => `<button class="hit" data-jump="ID${id}" type="button"><b>ID${id}</b> ${escapeHtml(byId["ID" + id] ? byId["ID" + id].role : "")}</button>${sentenceBlock(id)}`).join("")}
      <h3>下游答案空</h3>
      <p class="miss">共 ${node.indirectCount} 条潜在间接路径。点开答案空可看具体 via。这些不是某次运行的错误传播。</p>
      <p>${node.downstream.map(id => chip("ID" + id, "ID" + id)).join("") || "<span class='miss'>没有更后面的答案空。</span>"}</p>`;
  }
  const kindName = { atom: "契约原子", answer: "答案空" }[node.kind];
  panel.innerHTML = `<p class="meta">${kindName} · ${escapeHtml(node.process)} · 稿件 ${escapeHtml(current.id)}</p><h2>${escapeHtml(node.title)}</h2>${body}`;
  panel.querySelectorAll("[data-jump]").forEach(button => {
    button.addEventListener("click", () => choose(button.dataset.jump));
  });
}
function choose(nid) {
  const node = byId[nid];
  if (!node) return;
  state.selected = nid;
  if (node.process !== state.process) {
    state.process = node.process;
    state.chain = "";
  }
  if (state.chain && !visibleSet().has(nid)) state.chain = "";
  render();
  const button = document.querySelector(`button.node[data-nid="${CSS.escape(nid)}"]`);
  if (button) button.scrollIntoView({block: "nearest", inline: "nearest"});
}
function chainsForProcess() {
  const found = [];
  for (const rule of DATA.bindings) {
    if (rule.process === state.process && !found.includes(rule.chain)) found.push(rule.chain);
  }
  return found;
}
function renderControls() {
  document.getElementById("tabs").innerHTML = processes().map(name =>
    `<button class="tab${name === state.process ? " on" : ""}" data-process="${escapeHtml(name)}" type="button">${escapeHtml(name)}</button>`
  ).join("");
  document.querySelectorAll(".tab").forEach(button => {
    button.addEventListener("click", () => {
      state.process = button.dataset.process;
      state.chain = "";
      state.selected = null;
      render();
    });
  });
  const sampleBox = document.getElementById("sample");
  sampleBox.innerHTML = DATA.samples.map(item =>
    `<option value="${escapeHtml(item.id)}">${escapeHtml(item.id)} · ${escapeHtml(item.condition)}</option>`
  ).join("");
  sampleBox.value = state.sample;
  const chainBox = document.getElementById("chain");
  chainBox.innerHTML = `<option value="">全部</option>` + chainsForProcess().map(chain =>
    `<option value="${escapeHtml(chain)}">${escapeHtml(chain)}</option>`
  ).join("");
  chainBox.value = state.chain;
}
function render() {
  renderControls();
  renderBoard();
  renderPanel();
}
document.getElementById("sample").addEventListener("change", event => {
  state.sample = event.target.value;
  renderPanel();
});
document.getElementById("chain").addEventListener("change", event => {
  state.chain = event.target.value;
  render();
});
document.getElementById("answers").addEventListener("change", event => {
  state.showAnswers = event.target.checked;
  render();
});
document.getElementById("q").addEventListener("input", event => {
  state.query = event.target.value;
  const hit = event.target.value.trim().match(/^(?:ID)?(\d+)$/i);
  if (hit && byId["ID" + hit[1]]) choose("ID" + hit[1]);
  else renderBoard();
});
document.getElementById("zoom-in").addEventListener("click", () => {
  state.zoom = Math.min(1.6, state.zoom + 0.1);
  render();
});
document.getElementById("zoom-out").addEventListener("click", () => {
  state.zoom = Math.max(0.7, state.zoom - 0.1);
  render();
});
document.getElementById("wrap").addEventListener("click", () => {
  if (state.selected) {
    state.selected = null;
    render();
  }
});
window.addEventListener("resize", () => render());
render();
</script>
</body>
</html>
"""


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="生成工程契约依赖图的可点击页面。")
    parser.add_argument("--no-open", action="store_true", help="只写 HTML，不打开浏览器。")
    args = parser.parse_args()
    payload = build_payload()
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(_PAGE.replace("/*__DATA__*/", data), encoding="utf-8")
    print(
        f"Wrote {OUT_PATH} · samples {len(payload['samples'])}, "
        f"nodes {len(payload['nodes'])}, edges {len(payload['edges'])}"
    )
    if not args.no_open:
        webbrowser.open(OUT_PATH.resolve().as_uri())


if __name__ == "__main__":
    main()
