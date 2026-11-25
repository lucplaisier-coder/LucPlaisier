"""Light formatting of the pandoc Word file: table borders, 9 pt table text, 15 cm figures, 2.2 cm margins."""
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
d.save(sys.argv[1])
