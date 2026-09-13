"""
Build a test COC whose tables are drawn with STROKED LINES rather than filled
rectangles -- the opposite primitive from the HSA document -- to check that the
extractor is agnostic to how borders were produced.

Covers: a 2-row header with a horizontal merge, a vertical merge in the stub
column, a full-width banner row, and a table that runs across a page break.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)

OUT = "/tmp/line_drawn_coc.pdf"

H1 = ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=18, spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=14, spaceAfter=8)
H3 = ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=12, spaceAfter=6)
BODY = ParagraphStyle("BODY", fontName="Helvetica", fontSize=10, leading=13, spaceAfter=6)
CELL = ParagraphStyle("CELL", fontName="Helvetica", fontSize=9, leading=11)
CELLB = ParagraphStyle("CELLB", fontName="Helvetica-Bold", fontSize=9, leading=11)


def footer(canvas, doc):
    """A section-specific running footer, like the real document has."""
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(0.9 * inch, 0.5 * inch, f"{doc.page}  Schedule of Benefits")
    canvas.restoreState()


def benefit_rows(start, count):
    rows = []
    for i in range(start, start + count):
        rows.append([
            Paragraph(f"Covered Service {i}", CELL),
            Paragraph(f"${10 * i} copay", CELL),
            Paragraph(f"{20 + i}% after deductible", CELL),
            Paragraph("Prior authorization required." if i % 3 == 0 else "", CELL),
        ])
    return rows


def build():
    doc = BaseDocTemplate(OUT, pagesize=letter,
                          leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                          topMargin=0.9 * inch, bottomMargin=0.9 * inch)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame], onPage=footer)])

    story = [
        Paragraph("Section 1: Covered Health Care Services", H1),
        Paragraph(
            "Benefits described in this section are subject to the Annual Deductible "
            "and Coinsurance shown below, and to Medical Necessity review.", BODY),
        Paragraph("Schedule of Benefits Table", H2),
    ]

    header = [
        # row 0: stub (vertical merge) + a horizontal merge over two columns
        [Paragraph("Covered Health Care Service", CELLB),
         Paragraph("The Amount You Pay", CELLB), "",
         Paragraph("Limitations & Exceptions", CELLB)],
        # row 1: the two sub-labels under the merged header
        ["", Paragraph("Network", CELLB), Paragraph("Out-of-Network", CELLB), ""],
    ]
    banner = [[Paragraph("Acupuncture Services", CELLB), "", "", ""]]

    data = header + banner + benefit_rows(1, 14) + \
        [[Paragraph("Ambulance Services", CELLB), "", "", ""]] + benefit_rows(15, 16)

    style = TableStyle([
        # Stroked grid lines -- the whole point of this fixture.
        ("GRID", (0, 0), (-1, -1), 0.75, colors.black),
        ("BOX", (0, 0), (-1, -1), 1.25, colors.black),
        ("SPAN", (0, 0), (0, 1)),          # vertical merge in the stub column
        ("SPAN", (1, 0), (2, 0)),          # horizontal merge across the header
        ("SPAN", (3, 0), (3, 1)),          # vertical merge on the last column
        ("SPAN", (0, 2), (-1, 2)),         # full-width banner row
        ("SPAN", (0, 17), (-1, 17)),       # second full-width banner row
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ])
    table = Table(data, colWidths=[1.6 * inch, 1.4 * inch, 1.4 * inch, 2.3 * inch],
                  repeatRows=2, style=style)
    story.append(table)

    story += [
        Spacer(1, 14),
        Paragraph("Section 2: Exclusions and Limitations", H1),
        Paragraph("Experimental or Investigational Services", H3),
        Paragraph(
            "Services that are Experimental or Investigational are excluded, except "
            "as described under Clinical Trials in Section 1.", BODY),
    ]
    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
