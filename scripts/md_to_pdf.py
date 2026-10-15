#!/usr/bin/env python
"""Markdown -> PDF for the written reports (bao_cao/): headings, paragraphs, nested bullet / numbered lists, tables,
bold, inline code, horizontal rules. DejaVu Sans for full Vietnamese support. Needs reportlab, e.g.:

    uv run --no-project --with reportlab python scripts/md_to_pdf.py bao_cao/x.md bao_cao/x.pdf "Title" "Author"
"""

import re
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FD = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", FD + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DVB", FD + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DVM", FD + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DVB", italic="DV", boldItalic="DVB")

INK, MUTED, RULE, HEAD_BG = colors.HexColor("#1f2933"), colors.HexColor("#52606d"), colors.HexColor("#cbd2d9"), colors.HexColor("#e4e7eb")
ACCENT = colors.HexColor("#1d4e89")
S = {
    "title": ParagraphStyle("title", fontName="DVB", fontSize=17, leading=22, textColor=ACCENT, spaceAfter=10),
    "h2": ParagraphStyle("h2", fontName="DVB", fontSize=13, leading=17, textColor=ACCENT, spaceBefore=12, spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="DVB", fontSize=11, leading=15, textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.3, leading=10.8, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="DVB", fontSize=8.3, leading=10.8, textColor=INK),
}


def inline(t: str) -> str:
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", lambda m: f'<font name="DVM" size="8.3" color="#3e4c59">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"\1", t)  # *emphasis*: no italic DejaVu face registered, keep plain
    return t


def list_style(level: int, numbered: bool) -> ParagraphStyle:
    ind = 14 + 14 * level
    return ParagraphStyle(f"li{level}{numbered}", parent=S["p"], leftIndent=ind, bulletIndent=ind - 10, spaceAfter=2)


def table(rows: list[list[str]], width: float):
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]
    lens = [max(min(len(r[c]), 60) for r in rows) + 4 for c in range(ncol)]
    widths = [max(width * n / sum(lens), 1.6 * cm) for n in lens]
    k = width / sum(widths)
    widths = [w * k for w in widths]
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def build(md_path: str, pdf_path: str, title: str, author: str):
    lines = open(md_path, encoding="utf-8").read().splitlines()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=title, author=author)
    W = A4[0] - 4 * cm
    story, i = [], 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s:
            i += 1
            continue
        if s.startswith("# "):
            story.append(Paragraph(inline(s[2:]), S["title"]))
        elif s.startswith("## "):
            story.append(Paragraph(inline(s[3:]), S["h2"]))
        elif s.startswith("### "):
            story.append(Paragraph(inline(s[4:]), S["h3"]))
        elif s == "---":
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=4, spaceAfter=6))
        elif s.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            lead = []  # heading / lead-in sentence right before the table: never separate them from it
            while story and isinstance(story[-1], Paragraph) and getattr(story[-1].style, "keepWithNext", 0):
                lead.insert(0, story.pop())
            story.append(KeepTogether(lead + [Spacer(1, 3), table(rows, W)]))
            story.append(Spacer(1, 6))
            continue
        else:
            m = re.match(r"^(\s*)(- |\d+\. )(.*)$", ln)
            if m:
                level = len(m.group(1)) // 2
                numbered = m.group(2) != "- "
                text = m.group(3)
                j = i + 1
                # continuation lines: indented deeper, not a new list item / table / heading
                while j < len(lines) and lines[j].strip() and not re.match(r"^\s*(- |\d+\. |\||#)", lines[j]) \
                        and len(lines[j]) - len(lines[j].lstrip()) > len(m.group(1)):
                    text += " " + lines[j].strip()
                    j += 1
                k = j
                while k < len(lines) and not lines[k].strip():
                    k += 1
                lead_in = k < len(lines) and lines[k].strip().startswith("|")
                bullet = m.group(2).strip() if numbered else ("•" if level == 0 else "–")
                st = list_style(level, numbered)
                if lead_in:
                    st = ParagraphStyle(st.name + "k", parent=st, keepWithNext=1)
                story.append(Paragraph(inline(text), st, bulletText=bullet))
                i = j
                continue
            text = s
            j = i + 1
            while j < len(lines) and lines[j].strip() and not re.match(r"^\s*(- |\d+\. |\||#|---|\*\*)", lines[j]):
                text += " " + lines[j].strip()
                j += 1
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            lead_in = k < len(lines) and lines[k].strip().startswith("|")  # sentence introducing a table: keep with it
            story.append(Paragraph(inline(text), ParagraphStyle("pk", parent=S["p"], keepWithNext=1) if lead_in else S["p"]))
            i = j
            continue
        i += 1

    def footer(canv, d):
        canv.saveState()
        canv.setFont("DV", 8)
        canv.setFillColor(MUTED)
        canv.drawString(2 * cm, 1.1 * cm, f"{author} · {title}")
        canv.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Trang {d.page}")
        canv.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
