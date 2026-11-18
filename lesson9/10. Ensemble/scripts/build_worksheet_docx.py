"""Convert the completed Markdown worksheet into an editable illustrated Word file."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "第10讲 集成学习与Bootstrapping 学生工作单完成版.docx"


def font(run, size: float = 10.5, bold: bool = False, east: str = "宋体") -> None:
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east)


def inline(p, text: str, size: float = 10.5, bold: bool = False) -> None:
    # Handle emphasis/code/links while retaining each source word and number.
    pattern = r"(\*\*.*?\*\*|`[^`]*`|\[[^\]]+\]\([^)]+\))"
    for part in re.split(pattern, text):
        if not part:
            continue
        match = re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)", part)
        if match:
            link = OxmlElement("w:hyperlink")
            link.set(qn("r:id"), p.part.relate_to(match[2], RT.HYPERLINK, is_external=True))
            run = OxmlElement("w:r")
            props = OxmlElement("w:rPr")
            color = OxmlElement("w:color")
            color.set(qn("w:val"), "245678")
            props.append(color)
            run.append(props)
            t = OxmlElement("w:t")
            t.text = match[1]
            run.append(t)
            link.append(run)
            p._p.append(link)
        else:
            strong = part.startswith("**")
            value = part[2:-2] if strong else part.strip("`")
            r = p.add_run(value)
            font(r, size, bold or strong)
            if part.startswith("`"):
                r.font.name = "Consolas"
                r.font.size = Pt(size - .5)


def table(doc, lines: list[str]) -> None:
    rows = [[v.strip() for v in line.strip().strip("|").split("|")] for line in lines]
    rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", v) for v in r)]
    n = len(rows[0])
    tbl = doc.add_table(rows=0, cols=n)
    tbl.autofit = False
    widths = {
        3: [3.2, 6.8, 7.0],
        4: [4.0, 3.3, 4.0, 5.7],
        5: [3.2, 2.6, 2.6, 3.4, 5.2],
        8: [2.2] + [14.8 / 7] * 7,
    }.get(n, [17 / n] * n)
    if n == 5 and "解释成本" in rows[0]:
        widths = [2.5, 2.1, 2.6, 4.2, 5.6]
    if n == 5 and "预测波动" in rows[0][3]:
        widths = [3.1, 2.2, 2.6, 4.4, 4.7]
    if n == 5 and "TN" in rows[0]:
        widths = [5, 3, 3, 3, 3]
    for col, width in zip(tbl.columns, widths):
        col.width = Cm(width)
    props = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "bottom", "left", "right", "insideH", "insideV"]:
        item = OxmlElement("w:" + edge)
        for k, v in {"val": "single", "sz": "4", "color": "D9D9D9"}.items():
            item.set(qn("w:" + k), v)
        borders.append(item)
    props.append(borders)
    for i, values in enumerate(rows):
        row = tbl.add_row()
        trpr = row._tr.get_or_add_trPr()
        trpr.append(OxmlElement("w:cantSplit"))
        if i == 0:
            trpr.append(OxmlElement("w:tblHeader"))
        for j, (cell, value) in enumerate(zip(row.cells, values)):
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr = cell._tc.get_or_add_tcPr()
            margins = OxmlElement("w:tcMar")
            for edge, amount in [("top", "70"), ("bottom", "70"), ("left", "90"), ("right", "90")]:
                node = OxmlElement("w:" + edge)
                node.set(qn("w:w"), amount)
                node.set(qn("w:type"), "dxa")
                margins.append(node)
            tcpr.append(margins)
            if i == 0:
                sh = OxmlElement("w:shd")
                sh.set(qn("w:fill"), "E9EFF5")
                tcpr.append(sh)
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            p.paragraph_format.keep_with_next = i < len(rows) - 1
            short = len(re.sub(r"[A-Za-z]", "", value)) < 18 and len(value) < 38
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if short else WD_ALIGN_PARAGRAPH.LEFT
            inline(p, value, size=9 if n >= 5 else 9.5, bold=i == 0)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = Pt(2)
    p.add_run().font.size = Pt(2)


def main() -> None:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2)
    sec.top_margin, sec.bottom_margin = Cm(1.9), Cm(1.8)
    sec.footer_distance = Cm(.8)
    for name in ["Normal", "Title", "Heading 1", "Heading 2", "Caption"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        style.paragraph_format.space_after = Pt(3)
        style.paragraph_format.line_spacing = 1.10
    for border in doc.styles.element.xpath(".//w:pBdr"):
        border.getparent().remove(border)
    doc.styles["Normal"].font.size = Pt(10.5)
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    p._p.append(field)
    lines = (BASE / "student-worksheet.md").read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        text = lines[i].strip()
        if not text:
            i += 1
            continue
        if text.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            table(doc, block)
            continue
        if text.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                p = doc.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.keep_with_next = i + 1 < len(lines) and not lines[i + 1].strip().startswith("```")
                r = p.add_run(lines[i])
                font(r, 9)
                r.font.name = "Consolas"
                i += 1
            i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", text)
        if image:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(BASE / image[2]), width=Cm(17))
            picture = p._p.xpath(".//wp:docPr")[0]
            picture.set("descr", image[1])
        elif text.startswith("# "):
            p = doc.add_paragraph(style="Title")
            r = p.add_run("第10讲 集成学习与Bootstrapping\n学生工作单完成版")
            font(r, 20, True, "黑体")
            p.paragraph_format.space_after = Pt(10)
        elif text.startswith("## "):
            clean = text[3:].replace(". ", " ").replace("：", " ").replace(" / ", "与")
            p = doc.add_paragraph(style="Heading 1")
            r = p.add_run(clean)
            font(r, 14, True, "黑体")
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
        else:
            caption = text.startswith("*图") and text.endswith("*")
            if caption:
                text = text[1:-1]
            p = doc.add_paragraph(style="Caption" if caption else "Normal")
            p.paragraph_format.widow_control = True
            question = bool(re.match(r"\d+\. ", text))
            p.paragraph_format.keep_with_next = question
            if text == "**研究备忘录：**":
                p.paragraph_format.keep_with_next = True
            if text.startswith("在本次WDBC实验中"):
                p.paragraph_format.keep_together = True
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            inline(p, text[2:] if text.startswith("- ") else text, 9 if caption else 10.5, question)
        i += 1
    doc.core_properties.title = "第10讲 集成学习与Bootstrapping 学生工作单完成版"
    doc.core_properties.subject = "WDBC集成学习实验与学生工作单"
    doc.core_properties.author = ""
    doc.save(OUT)
    print(OUT)
    print("tables", len(doc.tables), "figures", len(doc.inline_shapes))


if __name__ == "__main__":
    main()
