from io import BytesIO
from typing import Dict, List


def _safe(text: str) -> str:
    return text.encode("latin-1", "ignore").decode("latin-1")


def build_research_pdf(
    topic: str,
    research: Dict[str, str],
    summary: str,
    sources: List[Dict] = None,
    mode: str = "general",
) -> bytes:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=20)

    # Title
    pdf.set_fill_color(23, 107, 135)
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_font("Arial", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(15, 12)
    pdf.multi_cell(180, 10, _safe(f"Research Report"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Arial", "", 14)
    pdf.set_xy(15, 28)
    pdf.multi_cell(180, 8, _safe(topic), new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(21, 32, 51)
    pdf.set_xy(15, 52)

    source_label = "Source: Web Research (Tavily)" if mode == "general" else "Source: Uploaded PDF"
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(92, 107, 122)
    pdf.cell(0, 6, _safe(source_label), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Summary box
    if summary:
        pdf.set_fill_color(231, 246, 244)
        pdf.set_draw_color(184, 223, 217)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(15, 79, 74)
        pdf.cell(0, 8, "Executive Summary", new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(21, 32, 51)
        pdf.multi_cell(0, 6, _safe(summary), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(6)

    # Sections
    for section, content in research.items():
        pdf.set_fill_color(244, 247, 251)
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(23, 107, 135)
        pdf.cell(0, 9, _safe(section), new_x="LMARGIN", new_y="NEXT", fill=True)
        pdf.ln(1)
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(21, 32, 51)
        pdf.multi_cell(0, 6, _safe(content), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

    # Sources
    if sources:
        pdf.add_page()
        pdf.set_font("Arial", "B", 13)
        pdf.set_text_color(23, 107, 135)
        pdf.cell(0, 9, "Sources & References", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Arial", "", 10)
        pdf.set_text_color(21, 32, 51)
        for i, s in enumerate(sources, 1):
            pdf.multi_cell(0, 6, _safe(f"{i}. {s.get('title', '')}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(23, 107, 135)
            pdf.multi_cell(0, 5, _safe(f"   {s.get('url', '')}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(21, 32, 51)
            pdf.ln(2)

    output = pdf.output(dest="S")
    return output.encode("latin-1") if isinstance(output, str) else bytes(output)
