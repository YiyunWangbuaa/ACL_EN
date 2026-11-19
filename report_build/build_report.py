# -*- coding: utf-8 -*-
"""Build the Word report: statistics, structure and snippets of the wind-tunnel data.

Run with the project Python (python-docx 1.2 is installed):
    python report_build/build_report.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "风洞数据基准统计报告.docx"
STATS = json.loads((ROOT / "report_build" / "stats.json").read_text(encoding="utf-8"))

NAVY = RGBColor(0x1F, 0x3A, 0x5F)
BLUE = RGBColor(0x2E, 0x74, 0xB5)
DARK = RGBColor(0x26, 0x26, 0x26)
GRAY = RGBColor(0x6B, 0x6B, 0x6B)
RED = RGBColor(0xB0, 0x30, 0x2E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEADER_FILL = "1F3A5F"
ZEBRA_FILL = "EDF2F7"
CODE_FILL = "F5F7FA"
NOTE_FILL = "EEF3F8"

CONTENT_CM = 16.6


# ---------------------------------------------------------------- low level

def set_run_font(run, name: str, size: float, bold: bool = False, color: RGBColor | None = None,
                 italic: bool = False) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), name)


def shade(element, fill: str) -> None:
    ppr = element.get_or_add_tcPr() if hasattr(element, "get_or_add_tcPr") else element.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.append(shd)


def cell_bg(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_margins(table, top=40, bottom=40, left=80, right=80) -> None:
    tbl_pr = table._tbl.tblPr
    mar = OxmlElement("w:tblCellMar")
    for side, value in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")
        mar.append(node)
    tbl_pr.insert_element_before(mar, "w:tblLook")


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant = OxmlElement("w:cantSplit")
    tr_pr.append(cant)


def add_paragraph(doc, text: str, size: float = 10.5, bold: bool = False, color: RGBColor = DARK,
                  align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_before: float = 0, space_after: float = 4,
                  first_indent: bool = False, line: float = 1.45, style: str | None = None):
    p = doc.add_paragraph(style=style)
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


def add_rich_paragraph(doc, parts, size: float = 10.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
                       space_before: float = 0, space_after: float = 4, line: float = 1.45,
                       first_indent: bool = False):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = line
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    for text, bold, color in parts:
        run = p.add_run(text)
        set_run_font(run, "宋体", size, bold, color)
    return p


def add_heading(doc, text: str, level: int = 1, number: str | None = None):
    if level == 0:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(6)
        run = p.add_run(text)
        set_run_font(run, "微软雅黑", 22, True, NAVY)
        return p
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.keep_with_next = True
    if level == 1:
        pf.space_before = Pt(16)
        pf.space_after = Pt(8)
        run = p.add_run((f"{number}  " if number else "") + text)
        set_run_font(run, "微软雅黑", 15, True, NAVY)
        # bottom rule
        ppr = p._p.get_or_add_pPr()
        p_bdr = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "single")
        bottom.set(qn("w:sz"), "6")
        bottom.set(qn("w:space"), "4")
        bottom.set(qn("w:color"), "2E74B5")
        p_bdr.append(bottom)
        ppr.insert_element_before(
            p_bdr, "w:shd", "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap",
            "w:overflowPunct", "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi",
            "w:adjustRightInd", "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing",
            "w:mirrorIndents", "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment",
            "w:textboxTightWrap", "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
        )
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


def add_caption(doc, text: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, "微软雅黑", 9, True, NAVY)
    return p


def add_note(doc, text: str):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.left_indent = Cm(0.2)
    run = p.add_run(text)
    set_run_font(run, "宋体", 9, False, GRAY)
    return p


def add_bullet(doc, text: str, bold_prefix: str | None = None):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.35
    if bold_prefix:
        run = p.add_run(bold_prefix)
        set_run_font(run, "宋体", 10.5, True, NAVY)
    run = p.add_run(text)
    set_run_font(run, "宋体", 10.5, False, DARK)
    return p


def display_width(text: str) -> int:
    total = 0
    for ch in text:
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total


def wrap_code(text: str, width: int = 112) -> str:
    lines = []
    for raw in text.replace("\t", "    ").splitlines():
        raw = raw.rstrip()
        if not raw:
            lines.append("")
            continue
        if display_width(raw) <= width:
            lines.append(raw)
            continue
        buf = ""
        current = 0
        for ch in raw:
            w = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
            if current + w > width:
                lines.append(buf)
                buf = ch
                current = w
            else:
                buf += ch
                current += w
        if buf:
            lines.append(buf)
    return "\n".join(lines)


def add_code(doc, text: str, caption: str | None = None, width: int = 92, max_lines: int | None = None,
             fill: str = CODE_FILL):
    if caption:
        add_caption(doc, caption)
    body = wrap_code(text, width)
    if max_lines is not None:
        rows = body.splitlines()
        if len(rows) > max_lines:
            rows = rows[:max_lines]
            rows.append("……（节选，后略）")
            body = "\n".join(rows)
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    pf.space_after = Pt(8)
    pf.line_spacing = 1.0
    pf.left_indent = Cm(0.15)
    pf.right_indent = Cm(0.15)
    pf.keep_together = True
    ppr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    ppr.insert_element_before(
        shd, "w:tabs", "w:suppressAutoHyphens", "w:kinsoku", "w:wordWrap", "w:overflowPunct",
        "w:topLinePunct", "w:autoSpaceDE", "w:autoSpaceDN", "w:bidi", "w:adjustRightInd",
        "w:snapToGrid", "w:spacing", "w:ind", "w:contextualSpacing", "w:mirrorIndents",
        "w:suppressOverlap", "w:jc", "w:textDirection", "w:textAlignment", "w:textboxTightWrap",
        "w:outlineLvl", "w:divId", "w:cnfStyle", "w:rPr", "w:sectPr",
    )
    p_bdr = OxmlElement("w:pBdr")
    for side in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{side}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "4")
        border.set(qn("w:space"), "4")
        border.set(qn("w:color"), "D5DCE5")
        p_bdr.append(border)
    ppr.insert_element_before(p_bdr, "w:shd")
    run = p.add_run(body)
    set_run_font(run, "Consolas", 8, False, RGBColor(0x1E, 0x1E, 0x1E))
    rfonts = run._element.get_or_add_rPr().find(qn("w:rFonts"))
    if rfonts is not None:
        rfonts.set(qn("w:eastAsia"), "宋体")
    return p


def add_table(doc, headers, rows, widths, font_size: float = 9, align_center: bool = False,
              first_col_bold: bool = False, zebra: bool = True, caption: str | None = None):
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
    keep_row_together(hdr)
    for index, text in enumerate(headers):
        cell = hdr.cells[index]
        cell.width = Cm(widths[index])
        cell_bg(cell, HEADER_FILL)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        run = p.add_run(str(text))
        set_run_font(run, "微软雅黑", font_size, True, WHITE)
    for r_index, row_values in enumerate(rows):
        row = table.add_row()
        keep_row_together(row)
        for c_index, value in enumerate(row_values):
            cell = row.cells[c_index]
            cell.width = Cm(widths[c_index])
            if zebra and r_index % 2 == 1:
                cell_bg(cell, ZEBRA_FILL)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if align_center else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.line_spacing = 1.15
            run = p.add_run(str(value))
            set_run_font(run, "宋体", font_size, bool(first_col_bold and c_index == 0), DARK)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    return table


def add_page_field(paragraph, kind: str):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {kind} "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(text)
    run._r.append(fld_end)


def setup_document(doc) -> None:
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
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
    # header
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("风洞数据基准统计报告  |  2026-11-03")
    set_run_font(r, "微软雅黑", 8, False, GRAY)
    # footer
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run("第 ")
    set_run_font(r, "宋体", 8.5, False, GRAY)
    add_page_field(fp, "PAGE")
    r = fp.add_run(" 页 / 共 ")
    set_run_font(r, "宋体", 8.5, False, GRAY)
    add_page_field(fp, "NUMPAGES")
    r = fp.add_run(" 页")
    set_run_font(r, "宋体", 8.5, False, GRAY)


# ---------------------------------------------------------------- content

def cover(doc):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("风洞数据基准统计报告")
    set_run_font(run, "微软雅黑", 26, True, NAVY)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run("数据规模 · 数据结构 · 标签分布 · 理论上限 · 典型片段")
    set_run_font(run, "微软雅黑", 11.5, False, BLUE)
    # rule
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    ppr = p._p.get_or_add_pPr()
    p_bdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "10")
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), "2E74B5")
    p_bdr.append(bottom)
    ppr.append(p_bdr)
    for _ in range(2):
        doc.add_paragraph()
    add_paragraph(doc, "数据版本  batch_100（100 条 × 193 空）",
                  size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, color=DARK)
    add_paragraph(doc, "对照数据  review_10（9 条）  ·  stress_10（10 条）",
                  size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, color=DARK)
    add_paragraph(doc, "试跑数据  pilot_10（10 条，DeepSeek V4 Flash，逐空协议）",
                  size=10.5, align=WD_ALIGN_PARAGRAPH.CENTER, color=DARK)
    for _ in range(6):
        doc.add_paragraph()
    add_paragraph(doc, "2026 年 11 月 3 日", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)
    add_paragraph(doc, "由 report_build/build_report.py 自动生成；统计口径见第 1 节与附录。",
                  size=9, align=WD_ALIGN_PARAGRAPH.CENTER, color=GRAY)
    doc.add_page_break()


def toc(doc):
    add_heading(doc, "本报告结构", 1, "0")
    rows = [
        ("1", "数据概览", "数据版本、文件清单、规模总览"),
        ("2", "数据结构", "样本行、spec、声明、问题、注入与判分"),
        ("3", "单条数据长度", "评测行 / 完整记录 / 单次 prompt / token"),
        ("4", "填空数量", "193 空 / 四子过程 / 句级调用 / 19,300 空"),
        ("5", "各标签数量", "C1 / C2 / A1 / A2、格式码、每子过程分布"),
        ("6", "理论数据总量上限", "16 因素、定义空间、空数、调用数"),
        ("7", "典型数据片段", "spec / 声明 / 物理常识 / 过程 / 问答 / prompt / 答案 / 记录"),
        ("8", "生成与版本", "生成脚本、评测脚本与版本"),
        ("9", "试跑结果（pilot_10）", "DeepSeek V4 Flash 的 10 条样例结果"),
        ("10", "结论与使用建议", "规模、成本、口径与注意事项"),
    ]
    add_table(doc, ["节", "标题", "内容"], rows, [1.2, 4.2, 11.2], font_size=9.5, align_center=False,
              first_col_bold=True, caption="表 0  报告章节")
    add_note(doc, "页码见页脚；正文中的“条”指一个样本（sample），“空”指一个待填空（blank），“调用”指一次模型请求。")


def summary(doc):
    add_heading(doc, "摘要", 1, "")
    add_paragraph(doc,
                  "本报告统计当前风洞数据基准的规模、结构、标签与理论上限。当前主数据为 batch_100："
                  "100 条定义扰动样本，每条 193 个待填空，共 19,300 空；另有 review_10 的 9 条单因素校核数据"
                  "（1,737 空）与 stress_10 的 10 条全扰动数据（1,930 空）。数据以 JSONL 存储，一行一条，"
                  "包含定义声明、16 项可变定义、193 个空的选项/标签/标准答案。",
                  first_indent=True)
    add_paragraph(doc,
                  "核心数字：单条评测行约 3.17 万字符（41.3 KB）；完整记录约 7.09 万字符；"
                  "单次句级 prompt 平均 4,229 字符（约 1,922 tokens）；一条数据的 155 次调用合计约 65.5 万字符；"
                  "100 条数据合计约 6,556 万字符（约 2,980 万 tokens）。标签分布为 A1 42.5%、C1 27.5%、A2 18.1%、C2 11.9%。"
                  "按 16 项可变定义的全组合计算，理论定义空间为 2^13 × 48^3 = 905,969,664 种，"
                  "对应 1.749×10^11 个空；四档（4/8/12/16 项变更）为 2.234×10^8 种、4.311×10^10 个空。"
                  "当前 100 条只是理论空间的一个极小样本。",
                  first_indent=True)
    cells = [
        ("100", "条主数据"), ("193", "空 / 条"), ("19,300", "空 / 100 条"), ("15,500", "句级调用"),
        ("4,229", "字符 / prompt"), ("1,922", "tokens / prompt"), ("9.06×10^8", "种定义"), ("1.75×10^11", "个空（理论上限）"),
    ]
    table = doc.add_table(rows=2, cols=4)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for index, (value, label) in enumerate(cells):
        row = index // 4
        col = index % 4
        cell = table.cell(row, col)
        cell.width = Cm(4.1)
        cell_bg(cell, NOTE_FILL if row == 0 else "FFFFFF")
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(value)
        set_run_font(run, "微软雅黑", 14, True, NAVY)
        p2 = cell.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p2.paragraph_format.space_before = Pt(0)
        p2.paragraph_format.space_after = Pt(4)
        run = p2.add_run(label)
        set_run_font(run, "宋体", 8.5, False, GRAY)
    add_caption(doc, "表 0-1  关键数字一览")
    add_note(doc, "“理论上限”按全定义空间 × 每条 193 空计算；四档上限见第 6 节。")


def section_overview(doc):
    add_heading(doc, "数据概览", 1, "1")
    add_paragraph(doc,
                  "当前数据分为三套：主数据 batch_100（100 条，四档定义扰动，每档 25 条）、单因素校核 review_10"
                  "（9 条，含标准样本与单项定义改动）、全扰动 stress_10（10 条，所有定义同时改离默认）。"
                  "另有 pilot_10（10 条）作为本地试跑子集。",
                  first_indent=True)
    rows = [
        ("batch_100", "100", "19,300", "batch_100/samples.jsonl（4.13 MB）", "2026-11-03", "四档 25 条/档；16 因素；seed 20261102"),
        ("review_10", "9", "1,737", "review_10/samples.json（1.16 MB）", "2026-11-02", "标准 + 单因素；含完整过程与 prior"),
        ("stress_10", "10", "1,930", "stress_10/samples.json（1.29 MB）", "2026-11-02", "全定义反向 + 轴排列"),
        ("pilot_10", "10", "1,930", "batch_100/pilot_10.json", "2026-11-03", "从 batch_100 按 3/3/2/2 抽取"),
        ("results/pilot_10", "10", "1,930", "results/pilot_10/records.jsonl（454 KB）", "2026-11-02/03", "DeepSeek V4 Flash 试跑记录"),
    ]
    add_table(doc, ["数据集", "条数", "空数", "文件", "生成时间", "说明"], rows,
              [2.4, 1.1, 1.4, 4.7, 2.4, 4.6], font_size=8.5, align_center=False, first_col_bold=True,
              caption="表 1  当前数据清单")
    add_paragraph(doc,
                  "文件组织：原始数据在 batch_100/、review_10/、stress_10/；试跑记录在 results/pilot_10/；"
                  "生成脚本为 batch_generate.py、dataset_generate.py、body_frame_model.py 与四个子过程生成器；"
                  "评测脚本为 cloze_eval.py 与 cloze_score.py。",
                  first_indent=True)
    add_bullet(doc, "16 项可变定义 = 13 个二值定义 + 3 个坐标系；模型安装角 γ_model 不进入任何空，欧拉顺序固定。", "定义空间：")
    add_bullet(doc, "25% / 50% / 75% / 100% 四档分别变更 4 / 8 / 12 / 16 项；100% 只有一种组合，靠三套坐标系的不同声明区分。", "四档：")
    add_bullet(doc, "四个子过程：风轴系到体轴系（81 空）、天平系到体轴系（33 空）、天平系到风轴系（65 空）、"
                    "力矩参考点（14 空）。", "结构：")
    add_bullet(doc, "空数以“待填槽”计；评测时按句子聚合，一个句子内多空一次作答，因此每条数据的调用数为 155 而非 193。",
               "口径：")


def section_structure(doc):
    add_heading(doc, "数据结构", 1, "2")
    add_paragraph(doc,
                  "eval 数据以 JSONL 存储，一行一个样本。每行包含样本编号、扰动档位、16 项定义、定义声明文本，"
                  "以及 193 个问题。问题字段给出子过程、空号、选项、格式码、标签与标准答案。",
                  first_indent=True)
    add_heading(doc, "2.1  一行 samples.jsonl 的字段", 2)
    rows = [
        ("id", "string", "样本编号", "100-001"),
        ("tier", "int", "扰动档位：25 / 50 / 75 / 100", "100"),
        ("changed_count", "int", "变更项数：4 / 8 / 12 / 16", "16"),
        ("changed", "list[str]", "被改离默认的 16 项名称", "alpha, beta, gamma, …"),
        ("title", "string", "标题", "定义变更100%"),
        ("change", "string", "定义变更说明（自然语言）", "定义变更档100%：……"),
        ("spec", "object", "16 项可变定义的完整声明", "见片段 2"),
        ("declaration", "string", "渲染后的定义声明（每次 prompt 都带）", "见片段 3"),
        ("questions", "list[object]", "193 个待填空", "见 2.2"),
    ]
    add_table(doc, ["字段", "类型", "含义", "示例"], rows, [3.0, 2.0, 7.0, 4.6], font_size=8.8,
              first_col_bold=True, caption="表 2  samples.jsonl 的样本级字段")
    add_heading(doc, "2.2  问题（questions）字段", 2)
    rows = [
        ("process", "string", "子过程名", "风轴系到体轴系"),
        ("id", "string", "空号（每子过程内 1 起）", "60"),
        ("options", "list[str] / null", "选项；选择题给出 3–17 项，自由作答为空", "XFD / YFD / …"),
        ("format_code", "string", "输出格式码：无 / 矩阵 / 表达式 / 向量 / 力臂 / 距离 / arctan / arcsin", "表达式"),
        ("label", "string", "题目标签：C1 / C2 / A1 / A2", "A2"),
        ("answer", "string", "标准答案；选择题为选项字母，自由作答为文本", "B / -cos(α_M)*cos(β_M)"),
    ]
    add_table(doc, ["字段", "类型", "含义", "示例"], rows, [2.6, 3.0, 7.6, 3.4], font_size=8.8,
              first_col_bold=True, caption="表 3  问题级字段")
    add_heading(doc, "2.3  spec 的 16 项可变定义", 2)
    rows = [
        ("理论迎角 α", "空速指向（机体上/下）× 符号", "2"),
        ("理论侧滑角 β", "空速指向（机体左/右）× 符号", "2"),
        ("理论滚转角 γ", "右翼指向（机体上/下）× 符号", "2"),
        ("机构迎角 α_M", "天平偏向（风洞上/下）× 符号", "2"),
        ("机构侧滑角 β_M", "天平偏向（风洞左/右）× 符号", "2"),
        ("机构滚转角 γ_M", "右翼指向（机体上/下）× 符号", "2"),
        ("安装方式 placement", "正装 / 反装", "2"),
        ("弹性迎角 α_e", "轴向弯向（天平上/下）× 符号", "2"),
        ("弹性侧滑角 β_e", "轴向弯向（天平左/右）× 符号", "2"),
        ("弹性滚转角 γ_e", "侧向移向（天平上/下）× 符号", "2"),
        ("气流偏角 α_V", "空速指向（机体上/下）× 符号", "2"),
        ("气流偏角 β_V", "空速指向（机体左/右）× 符号", "2"),
        ("天平安装角 γ_tp", "同向 / 反向", "2"),
        ("体轴系", "手性（左/右）+ X、Y 各自沿哪根绝对轴、指向哪一侧", "48（47 非默认）"),
        ("风轴系", "同上", "48（47 非默认）"),
        ("天平系", "同上", "48（47 非默认）"),
    ]
    add_table(doc, ["定义", "取值方式", "取值数"], rows, [3.6, 10.2, 2.8], font_size=8.8,
              first_col_bold=True, caption="表 4  16 项可变定义")
    add_note(doc, "模型安装角 γ_model 不在任何空中使用，四档保持默认；欧拉顺序由功角机构固定为迎角→侧滑角→滚转角，"
                  "不是可变定义。13 个二值定义与 3 个坐标系共同构成定义空间。")
    add_heading(doc, "2.4  从数据到模型输入的注入流程", 2)
    add_paragraph(doc, "评测时一条样本的 193 个空按句子聚合为 155 个请求。每次请求的 prompt 由六部分组成：",
                  first_indent=True)
    for prefix, text in (
        ("① 指令：", "“填写这一句里的全部待填槽，ID 为 …”或“只填写当前空”。"),
        ("② 定义声明：", "该样本的 declaration，约 2.4–2.7 千字符。"),
        ("③ 物理常识：", "PRIOR，174 字符；说明绝对轴、被动变换、动轴内旋。"),
        ("④ 子过程名：", "四个子过程之一。"),
        ("⑤ 过程正文：", "从子过程开头到当前句句号；当前句之前的空全部填该样本的 gold 答案；当前句的空为待填槽；"
                        "当前句之后的句子全部省略。"),
        ("⑥ 输出格式：", "按格式码给出输出示例；多空句要求逐行写成 ID:答案。"),
    ):
        add_bullet(doc, text, prefix)
    add_paragraph(doc,
                  "判分由 cloze_score.judge 完成：选择题按选项字母精确匹配，自由作答先做空白/符号归一化，"
                  "矩阵与表达式再做 sympy 等价比较。记录文件保存 prompt、prediction、gold、exact、line_hit 等字段。",
                  first_indent=True)
    add_heading(doc, "2.5  完整记录与 eval 行", 2)
    add_paragraph(doc,
                  "samples.jsonl 是 eval 行（slim）：只保存 declaration 与 questions，不含过程全文和 prior；"
                  "评测时由 dataset_generate 的过程模板 + PRIOR + 该样本 answers 重新拼出 prompt。"
                  "review_10/samples.json 保存完整记录（build_record）：额外含 prior、四个 processes 全文与 injection_example。"
                  "两者字段不同，长度也不同（见第 3 节）。",
                  first_indent=True)


def section_length(doc):
    add_heading(doc, "单条数据长度", 1, "3")
    s = STATS
    add_paragraph(doc,
                  "“单条数据”有三种口径：① 评测行（samples.jsonl 一行）；② 完整记录（build_record）；"
                  "③ 一次模型输入（句级 prompt）。三者的长度差异很大，下面分别给出。",
                  first_indent=True)
    rc = s["row_chars"]; rb = s["row_bytes"]; dc = s["declaration_chars"]; sc = s["spec_chars"]
    qc = s["questions_chars"]; ac = s["answers_chars"]; fr = s["full_record"]; pc = s["prompt_chars"]
    pe = s["prompt_token_estimate"]; calls = s["calls"]
    rows = [
        ("评测行（samples.jsonl）", "100", f"{rc['mean']:,.1f}（{rc['min']:,}–{rc['max']:,}）",
         f"{rb['mean']:,.0f}（{rb['min']:,}–{rb['max']:,}）", "declaration + spec + 193 个问答"),
        ("完整记录（100-001）", "1", f"{fr['json_chars']:,}", "—", "prior + 四个过程全文 + 193 个问答"),
        ("定义声明", "100", f"{dc['mean']:,.1f}（{dc['min']:,}–{dc['max']:,}）", "—", "每次 prompt 都带"),
        ("spec", "100", f"{sc['mean']:,.1f}", "—", "16 项定义，JSON"),
        ("questions（193 个）", "100", f"{qc['mean']:,.1f}", "—", "选项/标签/格式/答案"),
        ("answers（193 个）", "100", f"{ac['mean']:,.1f}", "—", "标准答案文本"),
        ("单次句级 prompt", f"{pc['count']:,}", f"{pc['mean']:,.1f}（{pc['min']:,}–{pc['max']:,}）",
         "—", f"约 {pe['mean_tokens']:,.0f} tokens"),
        ("一条数据的全部 prompt", "100", f"{calls['per_sample']['mean']:.0f} 次 / 条", "—",
         f"约 {round(calls['per_sample']['mean'] * pc['mean']):,} 字符/条"),
        ("过程全文（TZ_FD）", "1", f"{fr['process_chars']['风轴系到体轴系']:,}", "—", "81 空"),
        ("Markdown 审阅稿", "1", f"{fr['markdown_chars']:,}", "—", "四个过程全文"),
    ]
    add_table(doc, ["对象", "数量", "字符数（均值 / 范围）", "UTF-8 字节", "说明"], rows,
              [4.2, 1.4, 4.4, 3.2, 3.4], font_size=8.3, first_col_bold=True,
              caption="表 5  单条数据长度")
    add_paragraph(doc,
                  f"由表可见：一条评测行约 3.17 万字符、41.3 KB；完整记录约 7.09 万字符；"
                  f"但真正送给模型的单次 prompt 平均只有 {pc['mean']:,.0f} 字符（约 {pe['mean_tokens']:,.0f} tokens），"
                  f"因为每次只保留到当前句，当前句之前的空填 gold，之后的句子全部省略。"
                  f"一条数据需要 {calls['per_sample']['mean']:.0f} 次调用，合计约 "
                  f"{round(calls['per_sample']['mean'] * pc['mean']):,} 字符；100 条数据合计 "
                  f"{s['prompt_total_chars']:,} 字符（约 {pe['total_tokens']:,.0f} tokens）。",
                  first_indent=True)
    add_note(doc, "token 估算按约 2.2 字符/token（中英混合，含公式）折算；不同模型的 tokenizer 会有 ±20% 波动。"
                  "“字符数”按 Python len 计，“字节数”按 UTF-8 计。")
    rows = []
    for item in s["per_tier"]:
        rows.append((f"{item['tier']}%", f"{item['changed'][0]}/16" if item['changed'] else "—",
                     str(item["samples"]), f"{item['row_chars']['mean']:,.0f}",
                     f"{item['declaration_chars']['mean']:,.0f}"))
    add_table(doc, ["档位", "变更项", "条数", "评测行均值（字符）", "声明均值（字符）"], rows,
              [2.0, 2.4, 2.4, 4.8, 5.0], font_size=8.8, align_center=True, first_col_bold=True,
              caption="表 6  按扰动档位的长度")


def section_blanks(doc):
    add_heading(doc, "填空数量", 1, "4")
    add_paragraph(doc,
                  "每条数据固定 193 个空，分布在四个子过程：风轴系到体轴系 81 空、天平系到体轴系 33 空、"
                  "天平系到风轴系 65 空、力矩参考点 14 空。100 条主数据共 19,300 空；review_10 的 9 条共 1,737 空；"
                  "stress_10 的 10 条共 1,930 空。",
                  first_indent=True)
    pc = STATS["prompt_chars"]
    rows = []
    per_labels = STATS["per_process_labels"]
    proc_counts = STATS["process_counts"]
    call_counts = STATS["prompt_by_process"]
    for process in proc_counts:
        cnt = proc_counts[process]
        per_sample = cnt // 100
        labels = per_labels.get(process, {})
        calls = call_counts.get(process, {}).get("count", 0)
        rows.append((process, str(per_sample), f"{cnt:,}", str(labels.get("C1", 0) // 100),
                     str(labels.get("C2", 0) // 100), str(labels.get("A1", 0) // 100),
                     str(labels.get("A2", 0) // 100), f"{calls/100:.0f}" if calls else "—"))
    rows.append(("合计", "193", f"{sum(proc_counts.values()):,}", "53", "23", "82", "35",
                 f"{sum(v['count'] for v in call_counts.values())/100:.0f}"))
    add_table(doc, ["子过程", "空/条", "空/100 条", "C1", "C2", "A1", "A2", "句级调用/条"], rows,
              [4.8, 1.3, 1.6, 1.2, 1.2, 1.2, 1.2, 1.9], font_size=8.6, align_center=True,
              first_col_bold=True, caption="表 7  四个子过程的空数与标签")
    add_paragraph(doc,
                  f"评测时按句子聚合：193 个空被分成 155 个请求（每条）。其中单空句 133 个/条、双空句 13 个/条、"
                  f"三空句 6 个/条、四空句 1 个/条、六空句 2 个/条；多空句合计 22 个/条。"
                  f"100 条数据共 15,500 次调用。",
                  first_indent=True)
    rows = []
    group = STATS["calls"]["group_sizes"]
    labels = {"1": "单空句", "2": "双空句", "3": "三空句", "4": "四空句", "6": "六空句"}
    for key in ("1", "2", "3", "4", "6"):
        count = group.get(key, 0)
        rows.append((labels[key], f"{count:,}", f"{count/100:.0f}", f"{count/15500*100:.1f}%"))
    rows.append(("合计", f"{STATS['calls']['total']:,}", "155", "100.0%"))
    add_table(doc, ["句子类型", "组数/100 条", "组数/条", "占比"], rows,
              [3.2, 3.4, 2.6, 3.0], font_size=8.8, align_center=True, first_col_bold=True,
              caption="表 8  句级调用的空数分布")
    add_note(doc, "多空句合计 2,200 组（22 组/条，占 14.2%）；这些句子要求模型一次输出多行“ID:答案”。")


def section_labels(doc):
    add_heading(doc, "各标签数量", 1, "5")
    add_paragraph(doc,
                  "193 个空按标签分为 C1、C2、A1、A2 四类。C 类为带选项的判断/选择（答案为选项字母），"
                  "A 类为需要给出具体结果的作答（文本、向量、矩阵或表达式）；1 为单步，2 为多步/组合。"
                  "四类的数量与占比固定，不随定义扰动改变。",
                  first_indent=True)
    rows = [
        ("C1", "5,300", "53", "27.5%", "单项选择：轴名、方向、单步判断", "风轴系法向正方向 → B（YFD）"),
        ("C2", "2,300", "23", "11.9%", "概念/链条选择：坐标系、转换起终点", "转换起点 → 风轴系"),
        ("A1", "8,200", "82", "42.5%", "单步作答：符号、方向、向量、单矩阵", "RP1 → [[0,0,-1],[0,-1,0],[1,0,0]]"),
        ("A2", "3,500", "35", "18.1%", "多步组合：总矩阵、公式、载荷表达式", "R_t*[X_V,Y_V,Z_V] 的 YTZ → -cos(α_M)*cos(β_M)"),
    ]
    add_table(doc, ["标签", "空/100 条", "空/条", "占比", "类型", "示例"], rows,
              [1.4, 2.2, 1.4, 1.6, 5.0, 5.0], font_size=8.6, align_center=False, first_col_bold=True,
              caption="表 9  标签数量与类型")
    add_paragraph(doc,
                  "从分布看，A1 占 42.5%，是主体；A2 占 18.1%，是最难的一类（矩阵/表达式/载荷）；"
                  "C1/C2 合计 39.4%，主要考察定义与坐标系的判断。按子过程看，TZ_FD 的 A1 最多（39 空/条），"
                  "TP_FD 的 C1 较多（19 空/条），Ref_point 的 A2 比例最高（6/14）。",
                  first_indent=True)
    add_heading(doc, "5.1  按输出格式", 2)
    fmt = STATS["format_counts"]
    desc = {
        "无": "选择题/符号位；选项作答",
        "矩阵": "3×3 矩阵",
        "表达式": "符号表达式",
        "向量": "三维向量",
        "力臂": "力臂模板（C·臂/L_RT）",
        "距离": "长度表达式",
        "arctan": "arctan 形式",
        "arcsin": "arcsin 形式",
    }
    order = ["无", "矩阵", "表达式", "向量", "力臂", "距离", "arctan", "arcsin"]
    rows = []
    for key in order:
        if key in fmt:
            rows.append((key, f"{fmt[key]:,}", f"{fmt[key]/19300*100:.1f}%", desc.get(key, "")))
    add_table(doc, ["format_code", "数量/100 条", "占比", "含义"], rows,
              [2.4, 2.8, 2.2, 9.2], font_size=8.8, align_center=True, first_col_bold=True,
              caption="表 10  输出格式码分布")
    add_heading(doc, "5.2  按子过程", 2)
    rows = []
    for process in STATS["process_counts"]:
        labels = STATS["per_process_labels"].get(process, {})
        total = STATS["process_counts"][process] // 100
        rows.append((process, str(total), str(labels.get("C1", 0) // 100), str(labels.get("C2", 0) // 100),
                     str(labels.get("A1", 0) // 100), str(labels.get("A2", 0) // 100)))
    add_table(doc, ["子过程", "空/条", "C1", "C2", "A1", "A2"], rows,
              [5.4, 2.0, 2.2, 2.2, 2.2, 2.2], font_size=8.8, align_center=True, first_col_bold=True,
              caption="表 11  各子过程的标签分布（空/条）")


def section_theory(doc):
    add_heading(doc, "理论数据总量上限", 1, "6")
    add_paragraph(doc,
                  "理论上限取决于三件事：16 项可变定义能组合出多少种定义、每种定义有多少个空、以及评测条件有多少种。"
                  "下面按“定义 → 空 → 调用 → 评测条件”逐层给出上限，并区分“全空间”和“四档空间”。",
                  first_indent=True)
    t = STATS["theory"]
    rows = [
        ("可变定义", "16", "13 个二值定义 + 3 个坐标系（γ_model 不计，欧拉顺序固定）"),
        ("每个坐标系的合法声明", "48", "2 手性 × 3 个 X 绝对轴 × 2 个 X 指向 × 2 个 Y 绝对轴 × 2 个 Y 指向"),
        ("单坐标系非默认声明", "47", "48 − 1（默认）"),
        ("全定义空间", "2^13 × 48^3 = 905,969,664", "13 个二值 × 3 个坐标系"),
        ("四档空间（4/8/12/16）", f"{t['tiered_sum']:,}", "四个变更档的精确组合之和"),
        ("每条空数", "193", "81 + 33 + 65 + 14"),
        ("全空间空数", f"{t['blanks_full']:,}", "905,969,664 × 193"),
        ("四档空数", f"{t['blanks_tiered']:,}", f"{t['tiered_sum']:,} × 193"),
        ("每条句级调用", "155", "193 个空按句子聚合"),
        ("全空间调用数", f"{t['calls_full']:,.0f}", "905,969,664 × 155"),
        ("四档调用数", f"{t['calls_tiered']:,.0f}", f"{t['tiered_sum']:,} × 155"),
        ("评测条件", "5", "A / B / C / C′ / C″；C 为整段一次"),
        ("当前实际数据", "100 条 / 19,300 空 / 15,500 调用", "batch_100"),
        ("当前占全空间比例", "≈ 1.1×10^-7", "100 / 905,969,664"),
        ("当前占四档比例", "≈ 4.5×10^-7", "100 / 223,386,623"),
    ]
    add_table(doc, ["项", "取值", "说明"], rows, [4.6, 5.4, 6.4], font_size=8.6,
              first_col_bold=True, caption="表 12  理论数据总量上限")
    add_paragraph(doc,
                  f"直观地说：全空间约 9.06 亿种定义，按每条 193 空计算是 1.749×10^11 个空；"
                  f"若只统计四个变更档（4/8/12/16 项），约 2.234 亿种、4.311×10^10 个空。"
                  f"当前 100 条只覆盖全空间的约 1.1×10^-7。因此本数据不是“穷举集”，而是“分层抽样集”；"
                  f"理论上限用于说明覆盖边界和评测成本，不表示要真的生成全部组合。",
                  first_indent=True)
    rows = [
        ("25%", "4/16", f"{t['tiered']['4']:,}", f"{t['tiered']['4']*193:,}"),
        ("50%", "8/16", f"{t['tiered']['8']:,}", f"{t['tiered']['8']*193:,}"),
        ("75%", "12/16", f"{t['tiered']['12']:,}", f"{t['tiered']['12']*193:,}"),
        ("100%", "16/16", f"{t['tiered']['16']:,}", f"{t['tiered']['16']*193:,}"),
        ("合计", "—", f"{t['tiered_sum']:,}", f"{t['blanks_tiered']:,}"),
    ]
    add_table(doc, ["档位", "变更项", "定义数", "空数"], rows,
              [2.4, 2.6, 5.2, 6.4], font_size=8.8, align_center=True, first_col_bold=True,
              caption="表 13  四个变更档的理论定义数与空数")
    add_heading(doc, "6.1  评测条件对调用量的放大", 2)
    rows = [
        ("A（当前）", "gold", "截断", "逐句", "155", "oracle 上下文上限；局部能力"),
        ("B", "模型自己", "截断", "逐句", "155", "错误向下传递（free-running）"),
        ("C", "无", "完整", "全部一次", "4（按子过程）/ 1（按样本）", "一次性/端到端；上下文泄漏嫌疑"),
        ("C′", "gold", "完整", "逐句", "155", "纯上下文泄漏（与 A 只差后续正文）"),
        ("C″", "模型自己", "完整", "逐句", "155", "记忆 + 后续正文"),
    ]
    add_table(doc, ["条件", "空前", "后续正文", "作答", "调用/条", "目的"], rows,
              [1.6, 2.2, 2.0, 2.2, 3.4, 5.0], font_size=8.4, first_col_bold=True,
              caption="表 14  五种评测条件的调用量")
    add_paragraph(doc,
                  "若 5 个条件 × 10 个模型 × 3 个 seeds 全部跑 100 条数据，逐句条件约 155 × 5 × 10 × 3 × 100 = "
                  "232.5 万次调用（C 的一次性调用远小于此）；全空间 1.40×10^11 次调用在现实中不可行，"
                  "这也是为什么本数据采用分层抽样而不是穷举。",
                  first_indent=True)


def section_snippets(doc):
    add_heading(doc, "典型数据片段", 1, "7")
    add_paragraph(doc,
                  "本节直接截取真实数据，便于一眼看出数据长什么样。片段均来自 batch_100 的 100-001 与 pilot_10 的 025-003；"
                  "长文本做了换行和节选，节选处标注“……”；完整文件见 batch_100/、review_10/、stress_10/。",
                  first_indent=True)
    sn = STATS["snippets"]
    # 片段 1: row
    add_heading(doc, "7.1  一行 samples.jsonl（字段节选）", 2)
    row_json = {
        "id": "100-001",
        "tier": 100,
        "changed_count": 16,
        "changed": ["alpha", "beta", "gamma", "alpha_m", "beta_m", "gamma_m", "placement",
                    "alpha_e", "beta_e", "gamma_e", "alpha_v", "beta_v", "gamma_tp",
                    "body_frame", "wind_frame", "balance_frame"],
        "title": "定义变更100%",
        "change": sn["change_100"][:120] + "……",
        "spec": "……（见片段 2）",
        "declaration": "……（见片段 3）",
        "questions": [
            {"process": "风轴系到体轴系", "id": "1", "options": ["体轴系", "天平系", "风轴系"],
             "format_code": "无", "label": "C2", "answer": "风轴系"},
            {"process": "风轴系到体轴系", "id": "2", "options": ["体轴系", "天平系", "风轴系"],
             "format_code": "无", "label": "C2", "answer": "体轴系"},
            "……（共 193 个问题，后略）",
        ],
    }
    add_code(doc, json.dumps(row_json, ensure_ascii=False, indent=2),
             "片段 1  samples.jsonl 一行的字段结构（100-001；questions 只展示 2 条）")
    add_note(doc, "实际一行 3.17 万字符；这里只展示字段骨架。questions 的 193 条完整答案在 questions 数组中。")
    # 片段 2: spec
    add_heading(doc, "7.2  spec：16 项可变定义", 2)
    add_code(doc, sn["spec_100"], "片段 2  100-001 的 spec（完整 JSON 节选）", max_lines=46)
    # 片段 3: declaration
    add_heading(doc, "7.3  declaration：定义声明", 2)
    add_code(doc, sn["declaration"], "片段 3  100-001 的定义声明（节选）", max_lines=42)
    # 片段 4: prior
    add_heading(doc, "7.4  物理常识", 2)
    add_code(doc, sn["prior"], "片段 4  PRIOR（完整，174 字符）")
    # 片段 5: process
    add_heading(doc, "7.5  过程正文（风轴系到体轴系，前 3 步）", 2)
    add_code(doc, sn["process_tz_fd"], "片段 5  风轴系到体轴系的完整过程（节选）", max_lines=40)
    # 片段 6: question
    add_heading(doc, "7.6  一个具体问题", 2)
    add_code(doc, sn["question_60"], "片段 6  问题 JSON（100-001，风轴系到体轴系 ID 60，A2 表达式）")
    # 片段 7: single prompt
    add_heading(doc, "7.7  单空 prompt（节选）", 2)
    single = elide_common(sn["single_prompt"])
    add_code(doc, single, f"片段 7  025-003 单空 prompt（ID {sn['single_answers'][0]['id']}；节选）", max_lines=38)
    # 片段 8: multi prompt
    add_heading(doc, "7.8  多空句 prompt（节选）", 2)
    multi = elide_common(sn["multi_prompt"])
    add_code(doc, multi, f"片段 8  025-003 多空句 prompt（ID {','.join(a['id'] for a in sn['multi_answers'])}；节选）",
             max_lines=42)
    # 片段 9: answers
    add_heading(doc, "7.9  标准答案与选项", 2)
    lines = []
    for item in sn["multi_answers"]:
        answer = item["answer"]
        options = item.get("options") or []
        if options and answer in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and len(answer) == 1:
            idx = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(answer)
            text = options[idx] if idx < len(options) else ""
            lines.append(f"ID {item['id']}  {item['label']}  {item['format']}  →  {answer}（{text}）")
        else:
            lines.append(f"ID {item['id']}  {item['label']}  {item['format']}  →  {answer}")
    for item in sn["single_answers"]:
        lines.append(f"ID {item['id']}  {item['label']}  {item['format']}  →  {item['answer']}")
    add_code(doc, "\n".join(lines), "片段 9  对应的 gold 答案（选择题给出选项字母及选项原文）")
    # 片段 10: record
    add_heading(doc, "7.10  一条评测记录", 2)
    record = json.loads(sn["record_example"])
    record.pop("prompt", None)
    record["prompt"] = "……（省略；见片段 7/8）"
    add_code(doc, json.dumps(record, ensure_ascii=False, indent=2),
             "片段 10  results/pilot_10/records.jsonl 的一条记录（prompt 省略）")
    # 片段 11: tree
    add_heading(doc, "7.11  数据文件布局", 2)
    tree = """workflow/
├─ batch_100/
│  ├─ samples.jsonl       100 条评测数据（4.13 MB）
│  ├─ pilot_10.json       10 条试跑子集
│  └─ 00_索引.md
├─ review_10/             9 条单因素校核（含完整过程与 prior）
├─ stress_10/             10 条全扰动数据
├─ results/pilot_10/      试跑记录（1,930 条）
├─ batch_generate.py      16 因素、四档、每档 25 条
├─ dataset_generate.py    定义声明、过程模板、注入
├─ body_frame_model.py    坐标系/角度/机构定义
├─ tz_fd_generate.py / tp_tz_generate.py / tp_fd_generate.py / ref_point_generate.py
└─ cloze_eval.py / cloze_score.py   逐句评测与判分"""
    add_code(doc, tree, "片段 11  数据与脚本布局")


def elide_common(prompt: str) -> str:
    """Shorten the repeated declaration/prior inside a long prompt."""
    if not prompt:
        return ""
    prompt = prompt.replace("\r\n", "\n")
    prompt = re.sub(
        r"【定义声明】\n.*?\n\n【物理常识】\n.*?\n\n【子过程】",
        "【定义声明】……（见片段 3）\n\n【物理常识】……（见片段 4）\n\n【子过程】",
        prompt,
        flags=re.S,
    )
    return prompt


def section_versions(doc):
    add_heading(doc, "生成与版本", 1, "8")
    add_paragraph(doc,
                  "batch_100 由 batch_generate.py 生成：seed=20261102，四档各 25 条；13 个二值定义直接翻转，"
                  "3 个坐标系从 47 种非默认声明中抽取；100% 档只有一种变更组合，25 条靠三套坐标系的不同声明区分。"
                  "review_10 与 stress_10 由 dataset_generate.py 的定义变更清单生成；pilot_10 由 batch_generate.py "
                  "按 3/3/2/2 抽取。",
                  first_indent=True)
    add_bullet(doc, "batch_100/samples.jsonl：100 条 eval 行，含 declaration 与 193 个问题。", "主数据：")
    add_bullet(doc, "review_10/samples.json：9 条完整记录，额外含 prior、processes、injection_example。", "校核数据：")
    add_bullet(doc, "stress_10/samples.json：10 条全扰动数据，用于检验定义组合下的程序正确性。", "强扰动：")
    add_bullet(doc, "cloze_eval.py：逐句评测；cloze_score.py：选项精确 + 归一化 + sympy 判分。", "评测：")
    add_bullet(doc, "results/pilot_10/records.jsonl：DeepSeek V4 Flash 的 10 条试跑记录（1,930 条）。", "试跑：")
    add_note(doc, "试跑记录使用逐空协议（每条 193 次调用）；当前 cloze_eval.py 已改为逐句协议（每条 155 次调用），"
                  "因此试跑结果与后续正式运行在调用口径上不同，比较时需注意。")


def parse_pilot_report() -> dict:
    path = ROOT / "results" / "pilot_10" / "report.md"
    data = {"total": None, "tiers": {}, "labels": {}, "samples": []}
    if not path.is_file():
        return data
    text = path.read_text(encoding="utf-8")
    current = None
    for line in text.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            continue
        match = re.search(r"空数\s*([\d,]+)，整行一致\s*([\d,]+)（([\d.]+)%）", line)
        if match and current:
            total = int(match.group(1).replace(",", ""))
            exact = int(match.group(2).replace(",", ""))
            rate = float(match.group(3))
            entry = {"total": total, "exact": exact, "rate": rate}
            if current == "全部":
                data["total"] = entry
            elif current in ("25%", "50%", "75%", "100%"):
                data["tiers"][current] = entry
            elif current in ("C1", "C2", "A1", "A2"):
                data["labels"][current] = entry
            elif current.startswith("数据 "):
                data["samples"].append({"name": current, **entry})
    return data


def section_pilot(doc):
    add_heading(doc, "试跑结果（pilot_10）", 1, "9")
    data = parse_pilot_report()
    if data["total"]:
        total = data["total"]
        add_paragraph(doc,
                      f"pilot_10 使用 DeepSeek V4 Flash，thinking=disabled，temperature=0.0，逐空协议，"
                      f"共 {total['total']:,} 空、{total['exact']:,} 空严格一致（{total['rate']:.1f}%）。"
                      f"该结果用于验证评测管道与提示模板，不代表模型最终能力；且它是逐空协议，"
                      f"与当前逐句协议（155 次/条）不可直接比较。",
                      first_indent=True)
        rows = []
        for tier in ("25%", "50%", "75%", "100%"):
            if tier in data["tiers"]:
                e = data["tiers"][tier]
                rows.append((tier, f"{e['total']:,}", f"{e['exact']:,}", f"{e['rate']:.1f}%"))
        add_table(doc, ["档位", "空数", "严格一致", "准确率"], rows,
                  [2.4, 3.0, 3.0, 3.0], font_size=8.8, align_center=True, first_col_bold=True,
                  caption="表 15  试跑按档位")
        rows = []
        for label in ("C1", "C2", "A1", "A2"):
            if label in data["labels"]:
                e = data["labels"][label]
                rows.append((label, f"{e['total']:,}", f"{e['exact']:,}", f"{e['rate']:.1f}%"))
        add_table(doc, ["标签", "空数", "严格一致", "准确率"], rows,
                  [2.4, 3.0, 3.0, 3.0], font_size=8.8, align_center=True, first_col_bold=True,
                  caption="表 16  试跑按标签")
        add_note(doc, "严格一致 = 模型整段回复 strip 后与 gold 完全相等；空格、符号写法不同即判错。"
                      "试跑中的 A2 只有 34.0%，大量是矩阵/表达式的格式差异，不是物理错误。")
    else:
        add_paragraph(doc, "未找到 results/pilot_10/report.md。", first_indent=True)


def section_conclusion(doc):
    add_heading(doc, "结论与使用建议", 1, "10")
    for prefix, text in (
        ("规模：", "100 条主数据、19,300 空、15,500 次句级调用；另有 review_10 的 9 条（1,737 空）与 stress_10 的 10 条（1,930 空）。"),
        ("单条长度：", "评测行约 3.17 万字符（41.3 KB）；完整记录约 7.09 万字符；单次 prompt 平均 4,229 字符（约 1,922 tokens）；"
                     "一条数据 155 次调用合计约 65.5 万字符；100 条约 6,556 万字符（约 2,980 万 tokens）。"),
        ("标签：", "A1 42.5%、C1 27.5%、A2 18.1%、C2 11.9%；C 类为选项判断，A 类为具体结果，A2 最难。"),
        ("理论：", "全定义空间 905,969,664 种、1.749×10^11 个空；四档空间 2.234×10^8 种、4.311×10^10 个空；"
                 "当前 100 条仅覆盖约 1.1×10^-7，属于分层抽样而非穷举。"),
        ("成本：", "5 个评测条件 × 10 个模型 × 3 个 seeds × 100 条 ≈ 232.5 万次逐句调用；"
                 "实际预算应先用 pilot 估算单次 token 与单价，再按条件分批跑。"),
        ("口径：", "报告长度以字符/字节计，token 按约 2.2 字符/token 估算；选择题答案是选项字母，自由题是文本；"
                 "试跑是逐空协议、严格 exact，正式评测是逐句协议、归一化 + sympy。"),
    ):
        add_bullet(doc, text, prefix)
    add_heading(doc, "使用建议", 2)
    add_bullet(doc, "对外报告长度时同时给字符、字节与 token 估算，并注明 tokenizer 差异（±20%）。", "长度：")
    add_bullet(doc, "以 100 条为 pilot，再扩到 250 条/档或全量 1,000 条；全空间不可穷举，分层抽样即可支撑统计结论。", "规模：")
    add_bullet(doc, "主表用 A（gold 前缀）与 B（模型自己前缀）成对比较；C 作为一次性条件；"
                    "若要断言“上下文泄露”，补 C′（gold + 完整后续正文）。", "对照：")
    add_bullet(doc, "固定每题的温度/thinking/max_tokens；记录 prompt、response、usage/cost；"
                    "用归一化 + sympy 判分，同时保留严格 exact 作为对照。", "复现：")
    add_note(doc, "本报告的所有数字由 report_build/stats.py 从 batch_100、review_10、stress_10 与 results/pilot_10 直接统计；"
                  "统计口径与假设见各表脚注。")


def main() -> None:
    doc = Document()
    setup_document(doc)
    cover(doc)
    toc(doc)
    summary(doc)
    section_overview(doc)
    section_structure(doc)
    section_length(doc)
    section_blanks(doc)
    section_labels(doc)
    section_theory(doc)
    section_snippets(doc)
    section_versions(doc)
    section_pilot(doc)
    section_conclusion(doc)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
