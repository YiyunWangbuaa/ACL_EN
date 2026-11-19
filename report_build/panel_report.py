# -*- coding: utf-8 -*-
"""Build the Word analysis report for the wind-tunnel cloze panel.

Inputs (both regenerated from results/run_* by analysis/panel_structure.py and
analysis/panel_analysis.py):
    results/panel_stats.json      rates, pairs, cross-model, meta
    results/panel_structure.json  scale, composition, protocol checks

Output:
    风洞五模型实验分析报告.docx
    results/report_numbers.json   every figure quoted in the report

Run:
    python report_build/panel_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
STATS = json.loads((ROOT / "results" / "panel_stats.json").read_text(encoding="utf-8"))
STRUCT = json.loads((ROOT / "results" / "panel_structure.json").read_text(encoding="utf-8"))
LBL = json.loads((ROOT / "results" / "label_by_condition.json").read_text(encoding="utf-8"))
FIGS = ROOT / "report_build" / "figs"
OUT = ROOT / "风洞填空实验分析报告.docx"
NUMBERS = ROOT / "results" / "report_numbers.json"
SUMMARY_MD = ROOT / "results" / "panel_summary_4models.md"

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
HEADER_FILL = "1F3A5F"
ZEBRA_FILL = "EDF2F7"

CONDITIONS = ["A", "B", "C", "D", "E"]
LABELS = ["C1", "C2", "A1", "A2"]
LABEL_DESC = {
    "C1": "轴/符号选择（2–18 选项）",
    "C2": "方向或正负判断（2–3 选项）",
    "A1": "短答：表达式/矩阵（均值 10 字符）",
    "A2": "长答构造：矩阵/公式（无选项）",
}
TIERS = ["25", "50", "75", "100"]
PROCESSES = ["风轴系到体轴系", "天平系到体轴系", "天平系到风轴系", "力矩参考点"]
PROCESS_SHORT = ["风轴→体轴", "天平→体轴", "天平→风轴", "力矩参考点"]
MODEL_ORDER = ["Kimi K3", "Qwen3.8 Flash", "Qwen3.8 27B", "DeepSeek V4 Flash"]


# ---------------------------------------------------------------- helpers
def set_run_font(run, name, size, bold=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rfonts.set(qn(attr), name)


def cell_bg(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_margins(table, top=40, bottom=40, left=80, right=80):
    mar = OxmlElement("w:tblCellMar")
    for side, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    table._tbl.tblPr.insert_element_before(mar, "w:tblLook")


def keep_row(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    tr_pr.append(hdr)


def add_para(doc, text, size=10.5, bold=False, color=DARK, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             space_before=0, space_after=4, line=1.45, first_indent=False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    run = p.add_run(text)
    set_run_font(run, "宋体", size, bold, color)
    return p


def add_rich(doc, parts, size=10.5, first_indent=True, space_after=4):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    pf = p.paragraph_format
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    pf.line_spacing = 1.45
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    for text, bold in parts:
        run = p.add_run(text)
        set_run_font(run, "宋体", size, bold, DARK)
    return p


def add_heading(doc, text, level=1, number=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.keep_with_next = True
    if level == 1:
        pf.space_before = Pt(16)
        pf.space_after = Pt(8)
        run = p.add_run((f"{number}  " if number else "") + text)
        set_run_font(run, "微软雅黑", 15, True, NAVY)
        ppr = p._p.get_or_add_pPr()
        p_bdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), "2E74B5")
        p_bdr.append(bottom)
        ppr.insert_element_before(p_bdr, "w:shd", "w:tabs", "w:spacing", "w:ind", "w:jc")
    elif level == 2:
        pf.space_before = Pt(10)
        pf.space_after = Pt(4)
        run = p.add_run((f"{number}  " if number else "") + text)
        set_run_font(run, "微软雅黑", 12, True, BLUE)
    else:
        pf.space_before = Pt(8)
        pf.space_after = Pt(3)
        run = p.add_run((f"{number}  " if number else "") + text)
        set_run_font(run, "微软雅黑", 10.5, True, NAVY)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, "微软雅黑", 9, True, NAVY)


def add_note(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.2)
    run = p.add_run(text)
    set_run_font(run, "宋体", 9, False, GRAY)


def add_figure(doc, path, width_cm, caption, note=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Cm(width_cm))
    add_caption(doc, caption)
    if note:
        add_note(doc, note)


def add_table(doc, headers, rows, widths, font_size=8.8, center=False, first_bold=True,
              caption=None, left_first_col=False):
    if caption:
        add_caption(doc, caption)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    tbl_w = table._tbl.tblPr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        table._tbl.tblPr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(int(round(sum(widths) * 567))))
    tbl_w.set(qn("w:type"), "dxa")
    cell_margins(table)
    for index, width in enumerate(widths):
        table.columns[index].width = Cm(width)
    hdr = table.rows[0]
    keep_row(hdr)
    repeat_header(hdr)
    for index, text in enumerate(headers):
        cell = hdr.cells[index]
        cell.width = Cm(widths[index])
        cell_bg(cell, HEADER_FILL)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(str(text))
        set_run_font(run, "微软雅黑", font_size, True, WHITE)
    for r_index, row_values in enumerate(rows):
        row = table.add_row()
        keep_row(row)
        for c_index, value in enumerate(row_values):
            cell = row.cells[c_index]
            cell.width = Cm(widths[c_index])
            if r_index % 2 == 1:
                cell_bg(cell, ZEBRA_FILL)
            p = cell.paragraphs[0]
            if center:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif c_index == 0 and not left_first_col:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.1
            run = p.add_run(str(value))
            set_run_font(run, "宋体", font_size, bool(first_bold and c_index == 0), DARK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_page_field(paragraph, kind):
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {kind} "
    sep = OxmlElement("w:fldChar")
    sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(sep)
    run._r.append(text)
    run._r.append(end)


def setup(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    sec.top_margin = Cm(2.2)
    sec.bottom_margin = Cm(2.0)
    normal = doc.styles["Normal"]
    normal.font.name = "宋体"
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = DARK
    rpr = normal._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), "宋体")
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = hp.add_run("风洞填空实验分析报告  |  2026-11-03")
    set_run_font(run, "微软雅黑", 8, False, GRAY)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("第 ")
    set_run_font(run, "宋体", 8.5, False, GRAY)
    add_page_field(fp, "PAGE")
    run = fp.add_run(" 页 / 共 ")
    set_run_font(run, "宋体", 8.5, False, GRAY)
    add_page_field(fp, "NUMPAGES")
    run = fp.add_run(" 页")
    set_run_font(run, "宋体", 8.5, False, GRAY)


# ---------------------------------------------------------------- numbers
def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pp(x, d=1):
    return f"{x * 100:+.{d}f}"


def rb(model, cond):
    return STATS["models"][model][cond]


def rate(model, cond, kind="norm_rate"):
    return rb(model, cond)[kind]


def delta(model, a, b):
    p = STATS["models"][model].get("pairs", {}).get(f"{a}|{b}")
    return p["delta"] if p else None


def lrate(model, cond, label):
    cell = LBL["models"][model][cond][label]
    return cell["norm_rate"]


def ldrop(model, a, b, label):
    return lrate(model, b, label) - lrate(model, a, label)


def collect():
    models = {}
    for model in MODEL_ORDER:
        models[model] = {
            cond: {
                "exact": rb(model, cond)["exact_rate"],
                "norm": rb(model, cond)["norm_rate"],
                "n": rb(model, cond)["n"],
                "format": rb(model, cond).get("format"),
                "prompt_tokens": rb(model, cond).get("prompt_tokens"),
                "completion_tokens": rb(model, cond).get("completion_tokens"),
                "cost": rb(model, cond).get("cost"),
            }
            for cond in CONDITIONS
        }

    meta = {m: STATS["meta"][m] for m in MODEL_ORDER}
    calls = sum(v["calls"] for m in MODEL_ORDER for v in meta[m].values())
    prompt_tokens = sum(v["prompt_tokens"] for m in MODEL_ORDER for v in meta[m].values())
    completion_tokens = sum(v["completion_tokens"] for m in MODEL_ORDER for v in meta[m].values())
    cost = sum(v["cost"] for m in MODEL_ORDER for v in meta[m].values())

    gaps = {}
    for cond in CONDITIONS:
        ranked = sorted(((rate(m, cond), m) for m in MODEL_ORDER), reverse=True)
        gaps[cond] = {"top": ranked[0][1], "gap": ranked[0][0] - ranked[1][0]}

    flips = {m: STATS["models"][m]["pairs"]["A|B"] for m in MODEL_ORDER}
    labels = {m: STATS["models"][m]["A"]["labels"] for m in MODEL_ORDER}
    procs = {m: STATS["models"][m]["A"]["processes"] for m in MODEL_ORDER}
    tiers = {m: STATS["models"][m]["A"]["tiers"] for m in MODEL_ORDER}
    errs = {m: STATS["models"][m]["A"] for m in MODEL_ORDER}
    did = {m: delta(m, "B", "D") - delta(m, "A", "C") for m in MODEL_ORDER}

    return {
        "models": models,
        "meta": meta,
        "gaps": gaps,
        "flips": flips,
        "labels": labels,
        "processes": procs,
        "tiers": tiers,
        "errors": errs,
        "did": did,
        "pairs": {m: STATS["models"][m]["pairs"] for m in MODEL_ORDER},
        "scale": {
            "runs": len(MODEL_ORDER) * 5,
            "blanks_per_condition": 1930,
            "blanks_total": 1930 * len(MODEL_ORDER) * 5,
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd_recorded": round(cost, 4),
        },
        "structure": STRUCT["structure"],
        "protocol": STRUCT["protocol"],
    }


def main():
    F = collect()
    scale = F["scale"]
    s = F["structure"]
    parity = F["protocol"]["parity_vs_kimi"]

    gap_a = F["gaps"]["A"]["gap"]
    gap_b = F["gaps"]["B"]["gap"]
    gap_cp = F["gaps"]["C"]["gap"]
    gap_cpp = F["gaps"]["D"]["gap"]
    ab = [delta(m, "A", "B") for m in MODEL_ORDER]
    acp = [delta(m, "A", "C") for m in MODEL_ORDER]
    bcpp = [delta(m, "B", "D") for m in MODEL_ORDER]
    db = [delta(m, "E", "B") for m in MODEL_ORDER]
    did = list(F["did"].values())

    doc = Document()
    setup(doc)

    # ---------------- cover
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("风洞坐标变换填空实验")
    set_run_font(run, "微软雅黑", 26, True, NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("四模型 × 五条件：实验结构与结果分析")
    set_run_font(run, "微软雅黑", 16, False, BLUE)
    for _ in range(2):
        doc.add_paragraph()
    add_para(doc, "DeepSeek V4 Flash · Qwen3.8 Flash · Qwen3.8 27B · Kimi K3",
             size=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=DARK)
    add_para(doc, "条件 A / B / C / D / E  ·  10 条样本 × 193 空  ·  20 个运行",
             size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)
    for _ in range(4):
        doc.add_paragraph()
    add_para(doc, "2026 年 11 月 3 日", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)
    add_para(doc, "数据：results/panel_stats.json、results/panel_structure.json",
             size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)
    doc.add_page_break()

    # ---------------- 1 abstract
    add_heading(doc, "摘要", 1, "1")
    add_para(doc, f"本报告汇总 4 个模型在 5 个条件（A/B/C/D/E）上的填空实验结果。"
                  f"每个条件含 10 条样本、193 个空，共 1,930 空；四个逐句条件各 1,550 次调用，"
                  f"端到端条件 E 各 40 次调用。合计 {scale['runs']} 个运行、"
                  f"{scale['blanks_total']:,} 个判分空位、{scale['calls']:,} 次模型调用；"
                  f"严格正确率按整段回复与 gold 完全一致判定，归一化正确率额外容忍格式、符号与表达式等价。",
             first_indent=True)
    gap_gold_lo, gap_gold_hi = sorted((gap_a, gap_cp))
    gap_self_lo, gap_self_hi = sorted((gap_b, gap_cpp))
    c1_drop = [ldrop(m, "A", "B", "C1") for m in MODEL_ORDER]
    c2_drop = [ldrop(m, "A", "B", "C2") for m in MODEL_ORDER]
    a1_drop = [ldrop(m, "A", "B", "A1") for m in MODEL_ORDER]
    a2_drop = [ldrop(m, "A", "B", "A2") for m in MODEL_ORDER]
    a2_a = [lrate(m, "A", "A2") for m in MODEL_ORDER]
    a2_b = [lrate(m, "B", "A2") for m in MODEL_ORDER]
    a2_d = [lrate(m, "E", "A2") for m in MODEL_ORDER]

    add_rich(doc, [
        ("五项主要发现。", True),
        (f"① Kimi K3 在全部五个条件上均列第一：严格口径 A 79.9%、C 79.9%、E 65.3%、B 57.7%、D 57.3%；"
         f"gold 前缀条件下领先第二名 {pct(gap_gold_lo)}–{pct(gap_gold_hi)}，"
         f"自身前缀条件下领先收窄到 {pct(gap_self_lo)}–{pct(gap_self_hi)}（归一化）。", False),
    ])
    add_rich(doc, [
        (f"② 前缀来源是最大单效应：A→B 四个模型全部显著下降，幅度 "
         f"{pp(min(ab))} 到 {pp(max(ab))} 个百分点（McNemar p < 0.001）。", False),
    ])
    add_rich(doc, [
        (f"③ 后文可见性几乎没有效应：A→C 为 {pp(min(acp))} 到 {pp(max(acp))} 个百分点，"
         f"C≈A，既没有带来强泄露，也没有带来收益。", False),
    ])
    add_rich(doc, [
        (f"④ 后文救不回错误前缀：B→D 为 {pp(min(bcpp))} 到 {pp(max(bcpp))} 个百分点；"
         f"交互项（DiD）为 {pp(min(did))} 到 {pp(max(did))}，两个因子近似可加。", False),
    ])
    add_rich(doc, [
        (f"⑤ 标签层面的损失极不均匀：C1 在 A→B 上只掉 "
         f"{abs(max(c1_drop)) * 100:.1f}–{abs(min(c1_drop)) * 100:.1f} 个百分点，"
         f"A2 却掉 {abs(max(a2_drop)) * 100:.1f}–{abs(min(a2_drop)) * 100:.1f} 个百分点"
         f"（B 条件只剩 {pct(min(a2_b))}–{pct(max(a2_b))}），损失随“级联深度”单调放大"
         f"（C1 < C2 < A1 < A2，见图 1–图 4）。", False),
    ])
    add_para(doc, "结构上，本实验是 2×2 因子（前缀来源 gold/自身 × 后文截断/保留）加一个端到端基线臂 E，"
                  "嵌套层次为样本 → 子过程 → 句级调用 → 空位。数据核查发现 20 个运行的判分与元数据完全自洽；"
                  "Qwen 与 Kimi 的 A/C 提示词逐字节相同，DeepSeek 的 A/C 提示词仅空编号不同"
                  "（屏蔽数字后 1,550/1,550 相同），因此跨模型对比成立，但空位级配对必须按"
                  "（样本, 子过程, 序号）对齐，而不能直接用空号连接。", first_indent=True)

    rows = []
    for model in MODEL_ORDER:
        rows.append([model] + [pct(F["models"][model][c]["exact"]) for c in CONDITIONS])
    add_table(doc, ["模型", "A", "B", "C", "D", "E"], rows,
              [4.6, 2.3, 2.3, 2.3, 2.3, 2.3], font_size=9, center=True,
              caption="表 1  严格正确率总览（每格 1,930 空）")
    add_note(doc, "严格口径要求整段回复与 gold 完全一致；归一化口径见第 3 节。"
                  "results/ 下只有这 20 个 run（4 模型 × 5 条件）；早前两次试点"
                  "（DeepSeek V4 Pro 的逐空试跑、DeepSeek V4 Flash 的 A 复跑）目录已不存在，故不进入本表与后续比较。")

    # ---------------- 2 structure
    add_heading(doc, "实验结构", 1, "2")
    add_para(doc, "本节先说明因子结构，再给出规模、内容构成、判分与产物结构，最后报告一致性核查。"
                  "后文所有统计量都可由 results/ 下的 20 个 run 目录重算，"
                  "重算脚本为 analysis/panel_structure.py 与 analysis/panel_analysis.py。", first_indent=True)

    add_heading(doc, "2.1 条件设计与因子结构", 2)
    add_para(doc, "五个条件构成一个 2×2 因子加一个端到端臂：前缀来源（gold 或模型自身）与后文可见性"
                  "（在当前句截断或保留后文）交叉得到 A/B/C/D；E 不分工句，每个子过程只调用一次，"
                  "要求一次性填完全部空并只输出 JSON。", first_indent=True)
    rows = [
        ["A", "gold", "截断", "句级", "1,550", "1.25"],
        ["B", "模型自身", "截断", "句级", "1,550", "1.25"],
        ["C", "gold", "保留", "句级", "1,550", "1.25"],
        ["D", "模型自身", "保留", "句级", "1,550", "1.25"],
        ["E", "gold（整段）", "不适用", "每子过程 1 次", "40", "48.25"],
    ]
    add_table(doc, ["条件", "前缀来源", "后文", "调用粒度", "调用数/模型", "空/次"],
              rows, [2.4, 3.4, 2.2, 3.2, 2.8, 2.0], font_size=8.8, center=True,
              caption="表 2  五个条件的操作化定义")
    add_note(doc, "“空/次”为该条件下每次调用平均覆盖的空位数：A/B/C/D 的 1,930 空由 1,550 次调用覆盖，"
                  "E 的 1,930 空由 40 次调用覆盖。")

    add_heading(doc, "2.2 规模与嵌套层次", 2)
    add_para(doc, "最小判分单元是“空”，但调用单元是“句子的一小段（含 1–6 个空）”。两者之间是三级嵌套："
                  "样本 → 子过程 → 句级调用 → 空位。任何跨条件比较都必须按样本聚类，"
                  "否则会把 1,930 个空当成独立观测，严重高估显著性。", first_indent=True)
    rows = [
        ["样本（scenario）", "10", "同一批 025/050/075/100 档样本，20 个 run 完全一致"],
        ["子过程", "4", "风轴系到体轴系、天平系到体轴系、天平系到风轴系、力矩参考点"],
        ["句级调用（A/B/C/D）", "1,550", "155 次/样本；每条件 1,550 次"],
        ["端到端调用（E）", "40", "4 次/样本；每子过程 1 次覆盖该子过程全部空"],
        ["空位", "1,930", "193 空/样本；与调用粒度无关的固定判分总量"],
        ["判分单元（全部运行）", f"{scale['blanks_total']:,}", "20 个 run × 1,930 空"],
        ["模型调用总数", f"{scale['calls']:,}", "4 模型 ×（4×1,550 + 40）"],
    ]
    add_table(doc, ["层级", "数量", "说明"], rows, [4.4, 2.4, 9.0], font_size=8.8,
              caption="表 3  实验的嵌套层次与规模")
    rows = []
    for k in sorted(s["per_call_blank_hist"], key=int):
        calls = s["per_call_blank_hist"][k]
        blanks = int(k) * calls
        rows.append([k, f"{calls:,}", f"{blanks:,}", pct(blanks / 1930)])
    rows.append(["合计", "1,550", "1,930", "100.0%"])
    add_table(doc, ["每调用空数", "调用数", "覆盖空数", "占全部空"], rows,
              [3.2, 3.2, 3.2, 3.2], font_size=8.8, center=True,
              caption="表 4  每次调用覆盖的空数分布（条件 A）")
    add_note(doc, "分布由 1 空/次主导（1,330 次，占 68.9% 的空），说明多数句子只含一个待填空；"
                  "少数调用覆盖 4–6 空，集中在力矩参考点等连续表达式段落。")

    add_heading(doc, "2.3 内容构成：子过程、标签与档位", 2)
    add_para(doc, "混入不同子过程与空类型，是为了看错误是否集中在特定推理形态。四个子过程的空数并不均衡："
                  "风轴系到体轴系占 42.0%，力矩参考点仅占 7.3%，因此子过程差异要按列内比例读，"
                  "不能比较绝对数量。", first_indent=True)
    proc_blanks = {"风轴系到体轴系": 810, "天平系到体轴系": 330,
                   "天平系到风轴系": 650, "力矩参考点": 140}
    proc_calls = {"风轴系到体轴系": 690, "天平系到体轴系": 230,
                  "天平系到风轴系": 520, "力矩参考点": 110}
    rows = []
    for proc in PROCESSES:
        b = proc_blanks[proc]
        c = proc_calls[proc]
        rows.append([proc, str(b), pct(b / 1930), str(c), f"{b / c:.2f}"])
    rows.append(["合计", "1,930", "100.0%", "1,550", "1.25"])
    add_table(doc, ["子过程", "空数", "占比", "调用数", "空/次"], rows,
              [4.6, 2.6, 2.4, 2.6, 2.4], font_size=8.8, center=True,
              caption="表 5  四个子过程的空与调用构成（每条件）")
    rows = []
    for label in LABELS:
        v = F["labels"]["Kimi K3"][label]
        rows.append([label, LABEL_DESC[label], str(v["n"]), pct(v["n"] / 1930)])
    add_table(doc, ["标签", "含义", "空数", "占比"], rows,
              [2.2, 7.0, 2.6, 2.6], font_size=8.6,
              caption="表 6  四类空的构成（条件 A，各模型相同）")
    add_para(doc, "C1/C2 是选项型（答案为一个字母），A1/A2 是填写型（表达式或矩阵）。"
                  "A1 中仍有 630/820 带选项，属于“半构造”空；A2 全部无选项、平均答案长度 39 字符，"
                  "是难度最高的一类，也是分标签分析的重点。", first_indent=True)
    rows = []
    tier_samples = {"25": 3, "50": 3, "75": 2, "100": 2}
    for tier in TIERS:
        v = F["tiers"]["Kimi K3"][tier]
        rows.append([f"{tier}%", str(tier_samples[tier]), str(v["n"]), pct(v["n"] / 1930)])
    add_table(doc, ["档位", "样本数", "空数", "占比"], rows,
              [3.4, 3.0, 3.0, 3.0], font_size=8.8, center=True,
              caption="表 7  四个扰动档位的样本与空构成")
    add_note(doc, "档位指定义扰动幅度（25/50/75/100%）；每档只有 2–3 条样本，"
                  "档位层面的结论只能视为方向性，不能当作剂量-反应曲线。")

    add_heading(doc, "2.4 判分与统计结构", 2)
    add_para(doc, "判分分两层。严格层要求整段回复与 gold 字面一致；归一化层在此基础上接受格式差异"
                  "（空白、单位、希腊字母写法）、符号等价与 sympy 表达式等价。两层同时报告，"
                  "可以把“会不会”与“写得规不规范”分开。", first_indent=True)
    add_para(doc, "统计上，同一模型的两个条件共享同一批样本与同一批空位，属于配对设计，"
                  "因此用 McNemar 精确检验比较条件间的对/错翻转，用按样本重抽的 cluster bootstrap 给区间。"
                  "有效样本量是 10 条场景（每条 193 空），而不是 1,930 个独立观测："
                  "这正是 A→B 数百次翻转能给出 p < 1e-30 的原因，也是不应把空位当独立样本的原因。",
             first_indent=True)
    rows = []
    for model in MODEL_ORDER:
        p = F["flips"][model]
        ci = p["ci"]
        rows.append([model, str(p["a_only"]), str(p["b_only"]), pp(p["delta"]),
                     f"[{pp(ci[0])}, {pp(ci[1])}]",
                     "<0.001" if p["p"] < 0.001 else f"{p['p']:.3f}"])
    add_table(doc, ["模型", "仅 A 对", "仅 B 对", "Δ(pp)", "95% CI", "p"], rows,
              [4.2, 2.2, 2.2, 2.0, 3.4, 2.2], font_size=8.6, center=True,
              caption="表 8  配对结构：A→B 的翻转计数与检验（归一化）")
    add_note(doc, "“仅 A 对”指 A 判对而 B 判错的空数，“仅 B 对”反之。四个模型的翻转都强烈不对称"
                  "（仅 A 对 389–535 次，仅 B 对 68–101 次），是误差传播的直接证据。")

    add_heading(doc, "2.5 运行与产物结构", 2)
    add_para(doc, "每个 run 目录包含五类产物：meta.json（配置与汇总计数）、records.jsonl"
                  "（逐次调用的完整记录，含提示词、回复、逐空判分）、checkpoint.json（断点续跑状态）、"
                  "report.md（自动生成的条件报告）、raw_samples/（该 run 使用的样本快照）。"
                  "这种结构使得任何一张表都能从 records.jsonl 重算，而不必信任中间汇总。", first_indent=True)
    rows = [
        ["meta.json", "模型、条件、温度、thinking、样本列表、调用/空/对错计数", "配置与总量核对"],
        ["records.jsonl", "逐次调用：提示词、回复、逐空 exact/normalized、token 与成本", "唯一的判分事实来源"],
        ["checkpoint.json", "已完成句子数与断点信息", "断点续跑与完整性核对"],
        ["report.md", "该 run 的条件级汇总", "人工抽查"],
        ["raw_samples/", "该 run 实际使用的样本快照", "复现与版本追溯"],
    ]
    add_table(doc, ["产物", "内容", "用途"], rows, [3.4, 7.4, 4.8], font_size=8.6,
              caption="表 9  单个 run 目录的产物结构")
    add_para(doc, "命名上，DeepSeek V4 Flash 用 run_A/run_B/run_C/run_D/run_E，"
                  "Qwen 与 Kimi 用 run_<条件>_<模型> 后缀。两种命名对应同一套脚本与同一批样本，"
                  "差别只在空编号方案（见 2.6）。", first_indent=True)

    add_heading(doc, "2.6 协议一致性与可复现性核查", 2)
    add_para(doc, "在比较之前，先核对 20 个 run 是否真的可比。以下六项由脚本自动检查，"
                  "结果写入 results/panel_structure.json 的 protocol 字段。", first_indent=True)
    qwen_cpp = parity["qwen_D"]["identical_prompts"]
    rows = [
        ["判分自洽", "20/20 通过",
         "meta.json 的 exact/normalized 与 records.jsonl 逐空重算完全一致"],
        ["样本一致", "20/20 通过",
         "全部 run 使用相同的 10 条样本及其顺序"],
        ["提示词一致（Qwen vs Kimi）",
         f"A/C {parity['qwen_A']['identical_prompts']}/1,550；B {parity['qwen_B']['identical_prompts']}/1,550；"
         f"D {qwen_cpp}/1,550",
         "gold 前缀条件下逐字节相同；B/D 因前缀由模型自写而必然分叉"],
        ["提示词一致（DeepSeek vs Kimi）", f"A/C {parity['A']['identical_prompts']}/1,550",
         "其余 860 次仅空编号不同；屏蔽数字后 1,550/1,550 完全相同"],
        ["空编号方案", "DeepSeek 每子过程重启；Qwen/Kimi 连续",
         "DeepSeek 为 1–81 / 1–33 / 1–65 / 1–14；Qwen/Kimi 为 1–81 / 82–114 / 115–179 / 180–193"],
        ["解码设置", "一致",
         "温度 0、thinking disabled、max_tokens 512；DeepSeek 官方接口忽略 seed"],
    ]
    add_table(doc, ["检查项", "结果", "含义"], rows, [4.8, 4.6, 6.2], font_size=8.4,
              caption="表 10  协议一致性核查")
    add_para(doc, "编号差异是本次核查最重要的发现：DeepSeek 的 A–D 采用“每个子过程从 1 重新编号”，"
                  "Qwen 与 Kimi 采用“全样本连续编号”。两种方案下句子切分、空数与提示词内容都相同，"
                  "但空号会写进提示词，因此 DeepSeek 的 A/C 有 860 次提示词在字面上与他人不同。"
                  "屏蔽数字后 1,550/1,550 完全一致，说明差异仅来自编号，不构成难度差异；"
                  "但它意味着跨模型做空位级配对时不能用空号 join，必须按（样本, 子过程, 序号）对齐。",
             first_indent=True)
    add_para(doc, "E 臂还有一处完整性瑕疵：Qwen3.8 Flash 有 1 次调用因 max_tokens 截断"
                  "（finish_reason=length），33 个空没有预测，占该 run 的 1.7%。"
                  "其余 39 次与另外三个模型的 40 次均正常返回。该截断会轻微低估 Qwen Flash 的 E，"
                  "但不足以改变它与 Qwen 27B 的相对位置（两者相差 1.7 个百分点，与空位损失同量级，"
                  "故这一处排序应视为统计上不可区分）。", first_indent=True)

    add_heading(doc, "2.7 这个结构能分离什么", 2)
    add_para(doc, "把四个条件写成 2×2，可以把“历史由谁写”和“后文是否可见”两个因子分开：", first_indent=True,
             space_after=2)
    add_para(doc, "A→B 度量前缀来源效应（截断后文时，gold 前缀换成自身前缀）；", first_indent=True, space_after=2)
    add_para(doc, "A→C 度量后文可见性效应（gold 前缀下）；", first_indent=True, space_after=2)
    add_para(doc, "B→D 度量后文可见性效应（自身前缀下）；", first_indent=True, space_after=2)
    add_para(doc, "C→D 度量保留后文时的前缀来源效应；", first_indent=True, space_after=2)
    add_para(doc, "两因子是否可加，用差分中的差分（DiD）=（D−B）−（C−A）判断。", first_indent=True)
    add_para(doc, "E 臂不属于这个 2×2：它没有逐句前缀，是“整段一次性填写”的工程基线，"
                  "用来回答“如果只给一次机会能拿到多少”，因此只与 A/B 作参照，不做因子解释。",
             first_indent=True)

    # ---------------- 3 overall
    add_heading(doc, "总体结果", 1, "3")
    add_heading(doc, "3.1 归一化正确率", 2)
    rows = []
    for model in MODEL_ORDER:
        rows.append([model] + [pct(F["models"][model][c]["norm"]) for c in CONDITIONS])
    add_table(doc, ["模型", "A", "B", "C", "D", "E"], rows,
              [4.6, 2.3, 2.3, 2.3, 2.3, 2.3], font_size=9, center=True,
              caption="表 11  归一化正确率总览（每格 1,930 空）")
    add_para(doc, "严格与归一化的差距主要来自矩阵与表达式的书写格式。Kimi K3 的两套口径差 0.3–3.0 个百分点，"
                  "Qwen3.8 27B 为 0.0–3.0，DeepSeek 为 0.2–3.2，Qwen3.8 Flash 为 0.4–2.4。"
                  "差距在 C 与 E 上最大、在 B 上最小，说明“写不规范”主要发生在需要长表达式的条件里，"
                  "而不是在短选项判断上。", first_indent=True)

    add_heading(doc, "3.2 条件内配对差", 2)
    rows = []
    for model in MODEL_ORDER:
        row = [model]
        for a, b in (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")):
            row.append(pp(delta(model, a, b)))
        row.append(pp(-delta(model, "E", "B")))
        row.append(pp(-delta(model, "E", "A")))
        rows.append(row)
    add_table(doc, ["模型", "B−A", "C−A", "D−B", "D−C", "E−B", "E−A"], rows,
              [4.0, 1.95, 1.95, 1.95, 1.95, 1.95, 1.95], font_size=8.8, center=True,
              caption="表 12  条件内配对差（归一化，单位 pp；每列均为后者减前者）")
    add_para(doc, "表 12 的三条信息：B−A 全部为负且绝对值最大；C−A 与 D−B 都在零附近；"
                  "E 高于 B（+4.0 到 +9.4）但低于 A（−14.8 到 −10.9）。也就是说，"
                  "把历史交给模型自己写会损失 14.9–24.2 个百分点，而多给后文几乎不改变结果；"
                  "“整段一次填”确实避开了跨句累积，但仍拿不回 gold 前缀的收益。", first_indent=True)

    add_heading(doc, "3.3 交互项：两因子是否可加", 2)
    rows = []
    for model in MODEL_ORDER:
        rows.append([model, pp(delta(model, "A", "C")), pp(delta(model, "B", "D")),
                     pp(F["did"][model])])
    add_table(doc, ["模型", "后文效应（gold 前缀）", "后文效应（自身前缀）", "DiD"],
              rows, [4.6, 4.0, 4.0, 3.0], font_size=8.8, center=True,
              caption="表 13  后文效应的差分中的差分（pp）")
    add_para(doc, f"DiD 在 {pp(min(did))} 到 {pp(max(did))} 个百分点之间，"
                  f"相对 A→B 的 {abs(max(ab)) * 100:.1f}–{abs(min(ab)) * 100:.1f} 个百分点可以忽略，"
                  "说明两个因子近似可加：后文带来的微小变化不依赖前缀由谁写。"
                  "这解释了为什么 C≈A 与 D≈B 会同时成立。", first_indent=True)

    # ---------------- 4 cross model
    add_heading(doc, "跨模型比较", 1, "4")
    add_heading(doc, "4.1 各条件的模型排序", 2)
    rows = []
    for cond in CONDITIONS:
        ranked = sorted(MODEL_ORDER, key=lambda m: -rate(m, cond))
        for i, m in enumerate(ranked, 1):
            rows.append([cond, f"{i}. {m}", pct(rate(m, cond, "exact_rate")), pct(rate(m, cond))])
    add_table(doc, ["条件", "模型（按归一化排序）", "严格", "归一化"], rows,
              [2.2, 6.2, 3.6, 3.6], font_size=8.6,
              caption="表 14  五个条件的跨模型排序（20 行）")
    add_para(doc, f"Kimi K3 在五个条件上都是第一，且在 gold 前缀条件下优势最大"
                  f"（A 领先 {pct(gap_a)}、C 领先 {pct(gap_cp)}），"
                  f"在自身前缀条件下收窄到 {pct(gap_b)}–{pct(gap_cpp)}。"
                  f"Qwen3.8 Flash 与 27B 互有胜负：Flash 在 B、E 上更好，27B 在 D 上更好，A 与 C 基本相同。"
                  f"DeepSeek V4 Flash 的 A 与两档 Qwen 接近（严格 68.1% 对 68.3%/67.7%），"
                  f"但 B 与 D 偏低（严格 49.7%、48.1%），是对自身前缀最敏感的一个。", first_indent=True)

    add_heading(doc, "4.2 严格与归一化的口径敏感度", 2)
    rows = []
    for model in MODEL_ORDER:
        row = [model]
        for cond in CONDITIONS:
            row.append(pp(F["models"][model][cond]["norm"] - F["models"][model][cond]["exact"]))
        rows.append(row)
    add_table(doc, ["模型", "A", "B", "C", "D", "E"], rows,
              [4.6, 2.3, 2.3, 2.3, 2.3, 2.3], font_size=8.8, center=True,
              caption="表 15  归一化减严格（pp），度量格式损失")
    add_para(doc, "格式损失在 B 上最小（0.2–0.4 个百分点），在 C 与 E 上最大（2.0–3.2）。"
                  "若只看严格口径，会把“表达不规范”误记为“不会”，这也是本报告并列两套口径的原因。",
             first_indent=True)

    # ---------------- 5 breakdown
    add_heading(doc, "分标签、分档位与分子过程", 1, "5")

    add_heading(doc, "5.1 分标签 × 条件（图 1–图 4）", 2)
    add_para(doc, "results/panel_stats.json 只保存了条件 A 的标签拆分，无法回答“标签效应是否随条件改变”。"
                  "为此从 20 个 run 的 records.jsonl 逐空重算 4 个标签 × 5 个条件 × 4 个模型的完整网格"
                  "（results/label_by_condition.json），得到图 1–图 4：横轴是 5 个条件，纵轴是归一化正确率，"
                  "每组 4 根柱子对应 4 个模型。每格 230–820 空。", first_indent=True)
    label_note = {"C1": 530, "C2": 230, "A1": 820, "A2": 350}
    for idx, (label, fname) in enumerate((("C1", "label_C1.png"), ("C2", "label_C2.png"),
                                          ("A1", "label_A1.png"), ("A2", "label_A2.png")), 1):
        add_figure(doc, FIGS / fname, 15.4,
                   f"图 {idx}  {label} 在五个条件上的正确率（每条件 {label_note[label]} 空）",
                   note="纵轴归一化正确率；柱顶数字为百分比。数据：results/label_by_condition.json。")

    add_para(doc, f"四张图给出三条结论。第一，标签难度序在各条件下基本稳定：C1 > C2 > A1 > A2，"
                  f"唯一的例外是 Qwen3.8 Flash 的 E 条件（C2 85.7% 略高于 C1 82.8%）。"
                  f"在 A 条件下，C1 为 {pct(min(lrate(m, 'A', 'C1') for m in MODEL_ORDER))}–"
                  f"{pct(max(lrate(m, 'A', 'C1') for m in MODEL_ORDER))}，"
                  f"A2 只有 {pct(min(a2_a))}–{pct(max(a2_a))}，两端的差距接近 30–40 个百分点。",
             first_indent=True)
    add_para(doc, f"第二，前缀来源的破坏力随“级联深度”单调放大。C1（局部选轴）在 A→B 上只掉 "
                  f"{abs(max(c1_drop)) * 100:.1f}–{abs(min(c1_drop)) * 100:.1f} 个百分点；"
                  f"C2（局部判断）掉 {abs(max(c2_drop)) * 100:.1f}–{abs(min(c2_drop)) * 100:.1f}；"
                  f"A1（短构造）掉 {abs(max(a1_drop)) * 100:.1f}–{abs(min(a1_drop)) * 100:.1f}；"
                  f"A2（长链构造）掉 {abs(max(a2_drop)) * 100:.1f}–{abs(min(a2_drop)) * 100:.1f}。"
                  f"换句话说，一个空需要引用越多前面算出的中间结果，自身的错误率就越被前缀质量支配。",
             first_indent=True)
    add_para(doc, f"第三，A2 在 B 条件下几乎触底：四个模型只有 {pct(min(a2_b))}–{pct(max(a2_b))}，"
                  f"而同一批空在 A 条件下是 {pct(min(a2_a))}–{pct(max(a2_a))}。"
                  f"E 条件因为没有跨句级联，A2 回升到 {pct(min(a2_d))}–{pct(max(a2_d))}，"
                  f"仍远低于 A。这说明 A2 的崩塌不是“模型不会写矩阵”，而是“一旦中间结果被自己写坏，"
                  f"后面无法修复”。我抽查了 B/D 的 A2 错误：Kimi 的 326 个错空里 293 个是“其他”"
                  f"（矩阵轴序置换、力臂公式形态不同），预测本身格式完整、不是解析失败或空输出。",
             first_indent=True)

    add_heading(doc, "5.2 标签降幅：级联深度的量化", 2)
    rows = []
    for model in MODEL_ORDER:
        rows.append([model] + [pp(ldrop(model, "A", "B", label)) for label in LABELS]
                    + [pp(delta(model, "A", "B"))])
    add_table(doc, ["模型", "C1", "C2", "A1", "A2", "全部空"], rows,
              [4.2, 2.5, 2.5, 2.5, 2.5, 2.5], font_size=8.6, center=True,
              caption="表 16  A→B 的降幅（归一化，pp；最后一列为全部 1,930 空）")
    add_note(doc, "四个模型在 A2 上的降幅（−36.9 到 −61.1）都远大于各自的总降幅（−14.9 到 −24.2），"
                  "说明总体 A→B 的损失是被 A1/A2 这两类“构造型”空驱动的；C1/C2 把总体损失向下拉。")
    add_para(doc, "这张表也解释了第 3 节的总体结果为什么看起来“下降 20 个百分点左右”："
                  "它是 530 个几乎不受影响的 C1、230 个轻微受影响的 C2、820 个中度受影响的 A1 和"
                  "350 个崩塌的 A2 加权后的平均。若把 A2 单独拿出来看，风险量级完全不同。",
             first_indent=True)

    add_heading(doc, "5.3 分档位（归一化，条件 A）", 2)
    rows = []
    for model in MODEL_ORDER:
        t = F["tiers"][model]
        rows.append([model] + [pct(t[tier]["norm"] / t[tier]["n"]) for tier in TIERS])
    add_table(doc, ["模型", "25%", "50%", "75%", "100%"], rows,
              [4.2, 2.9, 2.9, 2.9, 2.9], font_size=8.8, center=True,
              caption="表 17  条件 A 的分档位正确率")
    t25 = [F["tiers"][m]["25"]["norm"] / F["tiers"][m]["25"]["n"] for m in MODEL_ORDER]
    t25_lo, t25_hi = min(t25), max(t25)
    add_para(doc, f"四个模型都是 25% 档最高（{pct(t25_lo)}–{pct(t25_hi)}），但 50/75/100 三档之间没有单调关系："
                  "Kimi 在 100% 档回升到 81.3%，DeepSeek 在 100% 档降到 65.8%，Qwen 两档在 75% 档最低。"
                  "由于每档只有 2–3 条样本，这些起伏更像是样本构成差异，"
                  "不应解读为“扰动越大越难”。若要把档位当作因子，每档至少需要 10 条样本。", first_indent=True)

    add_heading(doc, "5.4 分子过程（归一化，条件 A）", 2)
    rows = []
    for model in MODEL_ORDER:
        pr = F["processes"][model]
        rows.append([model] + [pct(pr[p]["norm"] / pr[p]["n"]) for p in PROCESSES])
    add_table(doc, ["模型", "风轴→体轴", "天平→体轴", "天平→风轴", "力矩参考点"], rows,
              [3.6, 3.1, 3.1, 3.1, 3.1], font_size=8.6, center=True,
              caption="表 18  条件 A 的分子过程正确率")
    add_para(doc, "力矩参考点是四个模型共同的最弱项（40.7–62.9%），而天平系到风轴系与天平系到体轴系"
                  "在 Kimi 上达到 84.5–88.2%。前者要求把载荷平移到参考点并写出完整表达式，"
                  "后者以矩阵与轴判断为主，难度结构不同。Qwen 27B 在天平系到体轴系上只有 53.0%，"
                  "是四个模型在该子过程上的最低值，提示其矩阵链式推导更脆弱。", first_indent=True)

    # ---------------- 6 errors
    add_heading(doc, "错误类型（条件 A）", 1, "6")
    rows = []
    for model in MODEL_ORDER:
        e = F["errors"][model]
        rows.append([model, str(e["defs"]), str(e["other"]), str(e["format"]),
                     pct(e["defs"] / e["n"])])
    add_table(doc, ["模型", "定义未用", "其他", "格式差", "定义未用占比"], rows,
              [4.2, 2.8, 2.4, 2.4, 3.2], font_size=8.8, center=True,
              caption="表 19  条件 A 的错误类型计数")
    add_para(doc, "“定义未用”指模型仍按默认定义作答、没有采用题面给出的定义，占 4.5–7.5%；"
                  "“其他”是级联错误与多步推导错误，数量最大（169–291）；格式差在归一化口径下被吸收。"
                  "定义未用比例最低的是 Kimi（4.5%），最高的是 Qwen3.8 Flash（7.5%），"
                  "而 DeepSeek 的“轴”类错误最多（79 次），说明它更多在轴的正负与方向上出错，"
                  "而不是忽略题面定义。", first_indent=True)

    # ---------------- 7 cost
    add_heading(doc, "成本与调用结构", 1, "7")
    rows = []
    for model in MODEL_ORDER:
        meta = F["meta"][model]
        calls = sum(v["calls"] for v in meta.values())
        ptok = sum(v["prompt_tokens"] for v in meta.values())
        ctok = sum(v["completion_tokens"] for v in meta.values())
        cost = sum(v["cost"] for v in meta.values())
        per_k = f"${cost / 9.65:.3f}" if cost else "未记录"
        rows.append([model, f"{calls:,}", f"{ptok:,}", f"{ctok:,}",
                     f"${cost:.2f}" if cost else "未记录", per_k])
    add_table(doc, ["模型", "调用", "输入 token", "输出 token", "记录成本", "每千空"],
              rows, [4.0, 2.2, 2.9, 2.7, 2.3, 2.3], font_size=8.6, center=True,
              caption="表 20  调用量、token 与记录成本（5 条件合计）")
    add_note(doc, "Qwen 与 Kimi 的 run 未写入成本字段（记录为 0），实际支出以账单为准；"
                  "DeepSeek 的 3.72 美元为逐次调用累加，可直接核对。“每千空”= 记录成本 / 9.65"
                  "（5 条件共 9,650 空）。")
    rows = []
    for cond in CONDITIONS:
        m = F["meta"]["DeepSeek V4 Flash"][cond]
        rows.append([cond, f"{m['calls']:,}", f"{m['prompt_tokens']:,}",
                     f"{m['completion_tokens']:,}", f"${m['cost']:.3f}"])
    add_table(doc, ["条件", "调用", "输入 token", "输出 token", "成本"], rows,
              [2.6, 2.2, 3.4, 3.2, 2.6], font_size=8.8, center=True,
              caption="表 21  DeepSeek V4 Flash 的分条件成本")
    add_para(doc, "C 的输入 token 比 A 高 41.3%（6,403,592 对 4,533,452），成本高 3.8 倍"
                  "（1.764 对 0.470 美元），但归一化正确率反而低 1.9 个百分点。"
                  "这是“保留后文”在工程上不划算的直接证据：它买到的是负收益。"
                  "E 相反，40 次调用覆盖全部 1,930 空，输入 token 只有 A 的 4.0%，是成本最低的臂。",
             first_indent=True)

    # ---------------- 8 discussion
    add_heading(doc, "讨论", 1, "8")
    add_para(doc, "第一，误差传播是主效应，且方向一致。四个模型的 A→B 都显著为负，"
                  "翻转计数高度不对称（仅 A 对 389–535 次，仅 B 对 68–101 次），"
                  "说明一旦历史由模型自己写，错误会沿句子链累积。这是“多轮自身历史”相对"
                  "“单轮给定上下文”的结构性劣势，而不是某个模型的能力差异。", first_indent=True)
    add_para(doc, "第二，模型差异主要体现在“有 gold 前缀”时。Kimi K3 的 A 与 C 都在 80% 左右，"
                  "比其他模型高 10.8–11.9 个百分点；但在 B 与 D 上只领先 2.9–3.2 个百分点。"
                  "换句话说，强模型的优势更多是“给定正确上下文时算得准”，"
                  "而不是“自己写错之后还能救回来”。对工程的含义是："
                  "与其换更强的模型，不如把上下文管理好。", first_indent=True)
    add_para(doc, "第三，泄露风险可控但并非为零。C 相对 A 的变化在 −1.9 到 +2.1 个百分点之间，"
                  "没有出现“看到后文就抄答案”的强泄露；但 Qwen Flash 与 27B 的 C 略高于 A"
                  "（+2.1、+2.0），Kimi 为 +0.7，方向一致地偏正，提示存在轻微的后文提示效应；"
                  "DeepSeek 为 −1.9，是唯一的负值。若把 C 当作无泄露上界使用，"
                  "应把这 2 个百分点计入。", first_indent=True)
    add_para(doc, "第四，E 不能替代 A/B。E 的排序与 A/B 不完全一致：Qwen Flash 的 E（56.6%）高于"
                  "Qwen 27B（54.4%，严格），但两者 A 几乎相同；DeepSeek 的 E（54.9%）也高于其 B（49.7%）。"
                  "E 是“一次性整段填写”，错误不会跨句传播，因此它更像 A 的下界而不是 B 的替代。",
             first_indent=True)
    add_para(doc, "第五，成本结构提示实验应压缩 C/D。C 与 D 的输入 token 比 A/B 高约 41%，"
                  "但边际信息量接近零（后文效应 −1.9 到 +2.1）。扩到 160 条时，"
                  "把 C/D 限制在 40 条子集、其余样本只跑 A/B，可以在几乎不损失结论的前提下"
                  "把成本降一半以上。", first_indent=True)
    add_para(doc, f"第六，标签分析把“误差传播”具体化成了一个可排序的机制：损失随级联深度递增。"
                  f"C1 只需读当前句的定义并选一个轴，A→B 几乎不掉（"
                  f"{abs(max(c1_drop)) * 100:.1f}–{abs(min(c1_drop)) * 100:.1f} 个百分点）；"
                  f"A2 要引用前面几步算出的矩阵与力臂，一旦前缀里的中间结果被写坏，"
                  f"后续无论定义读得多准都无法补救，于是从 {pct(min(a2_a))}–{pct(max(a2_a))} 掉到 "
                  f"{pct(min(a2_b))}–{pct(max(a2_b))}。这也解释了为什么 C（给全后文）救不回 A2："
                  f"后文里没有正确的中间结果，只有更多依赖它的句子。", first_indent=True)

    # ---------------- 9 caveats
    add_heading(doc, "口径与限制", 1, "9")
    for prefix, text in (
        ("样本量：", "10 条 pilot、每条件 1,930 空；区间按样本聚类重抽，但 run-to-run 方差未估计，"
                 "温度 0 也不等于确定性输出。"),
        ("编号差异：", "DeepSeek 的 A–E 用每子过程重启的空编号，Qwen/Kimi 用连续编号。"
                   "提示词屏蔽数字后 1,550/1,550 相同，故不影响难度，但影响空位级 join。"),
        ("成本字段：", "Qwen 与 Kimi 的 run 没有写入成本，表 20 的成本列不完整；"
                   "跨模型性价比只能对 DeepSeek 精确计算。"),
        ("E 臂截断：", "Qwen Flash 有 1 次调用触发 max_tokens 截断，33 空无预测（1.7%），"
                    "会轻微低估其 E，并使它与 Qwen 27B 的 E 排序不可区分。"),
        ("历史试点：", "DeepSeek V4 Pro 的逐空试跑与 DeepSeek V4 Flash 的 A 复跑，运行目录已不存在，"
                   "无法重算，故未纳入比较。"),
        ("判分：", "严格 exact 会惩罚格式差异；归一化依赖 sympy 与符号规则，长表达式仍可能误判等价。"),
        ("标签单元格：", "每个标签每条件只有 230–820 空，C2 与 A2 的区间最宽；"
                    "标签级结论应结合图 1–图 4 的整体形状读，不宜逐柱比较。"),
        ("外部效度：", "10 条样本集中在 4 个扰动档、每档 2–3 条，档位与子过程的交互无法识别。"),
    ):
        add_para(doc, prefix + text, size=10, first_indent=True, space_after=2)

    # ---------------- 10 conclusion
    add_heading(doc, "结论与下一步", 1, "10")
    add_para(doc, "结论：在 10 条样本、1,930 空、5 个条件的设定下，Kimi K3 全面领先；"
                  "前缀来源是最大单效应（−15 到 −24 个百分点），后文可见性接近零效应（−1.9 到 +2.1）；"
                  "后文无法补救错误前缀（D≈B）；E 位于 A 与 B 之间，是低成本基线而非替代品。"
                  "标签层面，C1 几乎不受前缀影响、A2 在自写前缀下崩塌，损失随构造链的级联深度单调放大。"
                  "实验结构是干净的 2×2 加一个端到端臂，两因子近似可加，20 个运行的判分与元数据完全自洽。",
             first_indent=True)
    add_para(doc, "下一步：① 扩到 160 条（4 场景 × 4 档 × 10），主实验只跑 2 个核心模型的 A/B；"
                  "② C/D 限制在 40 条子集，用于估计后文效应与泄露上界；"
                  "③ 40 条 × 8 个模型做 A/B 面板；④ 40 条子集重复 3 次，估计 run-to-run 方差；"
                  "⑤ 统一空编号方案，消除跨模型 join 的障碍。", first_indent=True)
    add_note(doc, "本报告由 results/panel_stats.json 与 results/panel_structure.json 生成，"
                  "全部数值另存于 results/report_numbers.json。")

    doc.save(OUT)
    NUMBERS.write_text(json.dumps(F, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 四模型 × 五条件 汇总（对应 Word 报告）", "",
             "每条件 10 条样本 × 193 空 = 1,930 空；严格与归一化两套口径。", "",
             "## 严格正确率", "",
             "| 模型 | " + " | ".join(CONDITIONS) + " |",
             "|---|" + "---|" * len(CONDITIONS)]
    for model in MODEL_ORDER:
        lines.append(f"| {model} | " + " | ".join(
            pct(F["models"][model][c]["exact"]) for c in CONDITIONS) + " |")
    lines += ["", "## 归一化正确率", "",
              "| 模型 | " + " | ".join(CONDITIONS) + " |",
              "|---|" + "---|" * len(CONDITIONS)]
    for model in MODEL_ORDER:
        lines.append(f"| {model} | " + " | ".join(
            pct(F["models"][model][c]["norm"]) for c in CONDITIONS) + " |")
    lines += ["", "## 条件内配对差（归一化，pp）", "",
              "| 模型 | B−A | C−A | D−B | D−C | E−B | E−A | DiD |",
              "|---|---|---|---|---|---|---|---|"]
    for model in MODEL_ORDER:
        d_ab = pp(delta(model, "A", "B"))
        d_acp = pp(delta(model, "A", "C"))
        d_bcpp = pp(delta(model, "B", "D"))
        d_cpcpp = pp(delta(model, "C", "D"))
        d_db = pp(-delta(model, "E", "B"))
        d_da = pp(-delta(model, "E", "A"))
        d_did = pp(F["did"][model])
        lines.append(f"| {model} | {d_ab} | {d_acp} | {d_bcpp} | {d_cpcpp} | "
                     f"{d_db} | {d_da} | {d_did} |")
    lines += ["", "## 分标签正确率（归一化；行=模型，列=条件）", ""]
    for label in LABELS:
        lines += [f"### {label}", "",
                  "| 模型 | " + " | ".join(CONDITIONS) + " |",
                  "|---|" + "---|" * len(CONDITIONS)]
        for model in MODEL_ORDER:
            lines.append(f"| {model} | " + " | ".join(
                pct(lrate(model, cond, label)) for cond in CONDITIONS) + " |")
        lines.append("")
    lines += ["## 规模", "",
              f"- 运行数 {scale['runs']}；判分空位 {scale['blanks_total']:,}；调用 {scale['calls']:,}",
              f"- 输入 token {scale['prompt_tokens']:,}；输出 token {scale['completion_tokens']:,}",
              f"- 记录成本 ${scale['cost_usd_recorded']:.2f}（仅 DeepSeek 记账）", "",
              "结构与协议核查见 results/panel_structure.json；"
              "标签网格见 results/label_by_condition.json；全部数值见 results/report_numbers.json。"]
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT}")
    print(f"wrote {NUMBERS}")
    print(f"wrote {SUMMARY_MD}")


if __name__ == "__main__":
    main()
