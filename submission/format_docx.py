"""Light formatting of the pandoc Word file: table borders, 9 pt table text, 15 cm figures, 2.2 cm margins,
page numbers in the footer, and tables/headings/figures kept together with what follows (layout only)."""
import sys
import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

d = docx.Document(sys.argv[1])
for t in d.tables:
    borders = OxmlElement("w:tblBorders")
    for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        x = OxmlElement(f"w:{edge}")
        x.set(qn("w:val"), "single"); x.set(qn("w:sz"), "4"); x.set(qn("w:color"), "999999")
        borders.append(x)
    t._tbl.tblPr.append(borders)
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
for s in d.inline_shapes:
    ratio = s.height / s.width
    s.width = Cm(15); s.height = int(Cm(15) * ratio)
for sec in d.sections:
    sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Cm(2.2)
    # Page number centred in the footer.
    fp = sec.footer.paragraphs[0]
    fp.alignment = 1
    run = fp.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text
        run._r.append(el)
# Keep every table on one page where it fits: each row's paragraphs stay with the next row,
# rows do not split, and the paragraph before a table (its heading) stays with it.
for t in d.tables:
    rows = list(t.rows)
    for i, row in enumerate(rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i < len(rows) - 1 and len(rows) <= 40:   # long annex tables may break across pages
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.keep_with_next = True
# Headings, figures and bold run-in labels stay with the next paragraph.
for p in d.paragraphs:
    if p.style.name.startswith("Heading") or p._p.xpath(".//w:drawing"):
        p.paragraph_format.keep_with_next = True
    elif p.runs and all(r.bold for r in p.runs if r.text.strip()) and len(p.text) < 120:
        p.paragraph_format.keep_with_next = True   # bold label lines such as "Limitations and deployment risks."
    elif p.runs and (p.runs[0].bold or p.runs[0].italic) and p.text.strip().startswith(("A", "Table", "Changes")) \
            and (len(p.text) < 160 or p.text.strip().startswith("Table")):
        p.paragraph_format.keep_with_next = True
d.save(sys.argv[1])
