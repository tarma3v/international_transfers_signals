"""Render the canonical publication-applicability research, without rewriting it."""
from pathlib import Path
import html
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak

from research.build_round3_report import register_fonts

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "research/publication_applicability/report-source.md"
OUTPUT = ROOT / "output/pdf/ivan_cbr_after_publication_applicability.pdf"
NAVY = colors.HexColor("#12263a")
BLUE = colors.HexColor("#176b93")


def inline(text):
    text = html.escape(text).replace("—", "-").replace("–", "-").replace("‑", "-")
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2" color="#176b93"><u>\1</u></a>', text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def build(source=SOURCE, output=OUTPUT, title="Курс ЦБ на завтра: когда его можно использовать"):
    register_fonts()
    styles = {
        "body": ParagraphStyle("body", fontName="Arial", fontSize=10, leading=14.1, textColor=NAVY, spaceAfter=7),
        "title": ParagraphStyle("title", fontName="Arial-Bold", fontSize=25, leading=29, textColor=NAVY, spaceAfter=15),
        "h2": ParagraphStyle("h2", fontName="Arial-Bold", fontSize=18, leading=22, textColor=NAVY, spaceAfter=12),
        "h3": ParagraphStyle("h3", fontName="Arial-Bold", fontSize=11.5, leading=15, textColor=BLUE, spaceAfter=6, spaceBefore=5),
        "cell": ParagraphStyle("cell", fontName="Arial", fontSize=8.8, leading=11.8, textColor=NAVY),
        "th": ParagraphStyle("th", fontName="Arial-Bold", fontSize=8.8, leading=11.8, textColor=colors.white),
    }
    p = lambda text, kind="body": Paragraph(inline(text), styles[kind])
    story = []
    for page in source.read_text().split("---page---"):
        if story:
            story.append(PageBreak())
        blocks = re.split(r"\n\s*\n", page.strip())
        for block in blocks:
            if block.startswith("### "):
                story.append(p(block[4:], "h3"))
            elif block.startswith("## "):
                story.append(p(block[3:], "h2"))
            elif block.startswith("# "):
                story.append(p(block[2:], "title"))
            elif block.startswith("|"):
                rows = [line.strip().strip("|").split("|") for line in block.splitlines() if not re.match(r"\|[-: |]+\|", line)]
                width = (174*mm)/len(rows[0])
                table = Table([[p(c.strip(), "th" if i==0 else "cell") for c in row] for i,row in enumerate(rows)], colWidths=[width]*len(rows[0]), repeatRows=1)
                table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),NAVY), ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#edf4f8"),colors.white]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
                story += [table, Spacer(1,8)]
            elif block.startswith("- "):
                story += [p("• "+line[2:]) for line in block.splitlines()]
            elif re.match(r"\d\. ",block):
                story += [p(line) for line in block.splitlines()]
            else:
                story.append(p(" ".join(block.splitlines())))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c5d6e2"))
        canvas.line(18*mm, 15*mm, 192*mm,15*mm)
        canvas.setFont("Arial",7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(18*mm,10*mm,"international_transfers_signals | проверка доступности и пользы")
        canvas.drawRightString(192*mm,10*mm,f"06.09.2026  •  {doc.page}")
        canvas.restoreState()
    output.parent.mkdir(parents=True,exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18*mm, rightMargin=18*mm, topMargin=17*mm, bottomMargin=21*mm, title=title, author="international_transfers_signals")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    build()
