"""Build the editable academic report from docs/Report.md.

Requires python-docx. PDF export: Word or LibreOffice, followed by visual QA.
"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    sec.left_margin = sec.right_margin = Cm(2)
    for name, size in [("Normal", 11), ("Title", 23), ("Heading 1", 16), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        for border in style.element.findall('.//' + qn('w:pBdr')):
            border.getparent().remove(border)
        for fonts in style.element.findall('.//' + qn('w:rFonts')):
            for attr in list(fonts.attrib):
                if attr.endswith('Theme'):
                    del fonts.attrib[attr]
    normal = doc.styles["Normal"].paragraph_format
    normal.space_after = Pt(7)
    normal.line_spacing = 1.1
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)
    lines = (ROOT / "docs/Report.md").read_text(encoding="utf-8").splitlines()
    i = 0
    first = True
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "<!-- page -->":
            doc.add_page_break()
        elif line.startswith("# "):
            doc.add_paragraph(line[2:], "Title" if first else "Heading 1")
            first = False
        elif line.startswith("## "):
            doc.add_paragraph(line[3:], "Heading 2")
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            image_path = (ROOT / "docs" / match[2]).resolve()
            from PIL import Image
            with Image.open(image_path) as img:
                w, h = img.size
            # Keep plots readable and leave room for context on the same page.
            width_cm = min(17, 7.3 * w / h)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(image_path), width=Cm(width_cm))
            caption = doc.add_paragraph(match[1])
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in caption.runs:
                run.font.size = Pt(9)
                run.italic = True
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                values = [v.strip() for v in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[-: ]+", v) for v in values):
                    rows.append(values)
                i += 1
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.autofit = False
            widths = {2: [7.0, 10.0], 3: [5.5, 3.0, 8.5], 4: [7.0, 3.3, 3.3, 3.4],
                      7: [3.5, 2.1, 2.1, 2.4, 2.4, 2.0, 2.0]}.get(len(rows[0]))
            if widths:
                for col, width in zip(table.columns, widths):
                    col.width = Cm(width)
            for row_index, row_values in enumerate(rows):
                cells = table.add_row().cells
                for idx, (cell, value) in enumerate(zip(cells, row_values)):
                    if widths:
                        cell.width = Cm(widths[idx])
                    cell.text = value
                    tcpr = cell._tc.get_or_add_tcPr()
                    borders = OxmlElement("w:tcBorders")
                    for edge in ["top", "left", "bottom", "right"]:
                        tag = OxmlElement("w:" + edge)
                        for key, val in [("val", "single"), ("sz", "4"), ("color", "D9D9D9")]:
                            tag.set(qn("w:" + key), val)
                        borders.append(tag)
                    tcpr.append(borders)
                    fill = OxmlElement("w:shd")
                    fill.set(qn("w:fill"), "E2E8EE" if row_index == 0 else ("F5F7F9" if row_index % 2 == 0 else "FFFFFF"))
                    tcpr.append(fill)
                    margins = OxmlElement("w:tcMar")
                    for edge in ["top", "left", "bottom", "right"]:
                        tag = OxmlElement("w:" + edge)
                        tag.set(qn("w:w"), "70")
                        tag.set(qn("w:type"), "dxa")
                        margins.append(tag)
                    tcpr.append(margins)
                    valign = OxmlElement("w:vAlign"); valign.set(qn("w:val"), "center"); tcpr.append(valign)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.line_spacing = 1.0
                        if idx > 0 and len(rows[0]) > 3:
                            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        for run in p.runs:
                            run.font.size = Pt(9)
                            run.bold = row_index == 0
                if row_index == 0:
                    repeat = OxmlElement("w:tblHeader"); table.rows[-1]._tr.get_or_add_trPr().append(repeat)
                cant_split = OxmlElement("w:cantSplit")
                table.rows[-1]._tr.get_or_add_trPr().append(cant_split)
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
            continue
        else:
            doc.add_paragraph(line)
        i += 1
    doc.core_properties.title = "Dự đoán đăng ký tiền gửi có kỳ hạn"
    doc.core_properties.author = "Nhóm BTL_ML — Lợi, Trí, Danh, Nhân"
    path = ROOT / "reports/report_draft.docx"
    doc.save(path)
    print(path)


if __name__ == "__main__":
    build()
