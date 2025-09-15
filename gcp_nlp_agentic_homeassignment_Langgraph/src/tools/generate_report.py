from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, ListFlowable, ListItem, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.lib import colors
from reportlab.pdfgen import canvas
import os
import tempfile


def build_report_platypus(out_path: str, diagram_path: str):
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    title_style.alignment = TA_CENTER
    h_style = styles['Heading2']
    body = styles['BodyText']
    body_lead = ParagraphStyle(name='BodyLead', parent=body, leading=16, spaceAfter=6)

    doc = SimpleDocTemplate(out_path, pagesize=LETTER, leftMargin=50, rightMargin=50, topMargin=54, bottomMargin=54)
    story = []

    story.append(Paragraph("Architecture & Agent Design Report", title_style))
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("Overview", h_style))
    story.append(Paragraph(
        "This prototype implements an end-to-end text analytics pipeline on Google Cloud, producing entity extraction, "
        "sentiment analysis, and concise summaries for each row in a CSV dataset. It also includes a lightweight agentic "
        "workflow that retrieves the most relevant rows to a user query, analyzes them with tool calls, and synthesizes a final answer.",
        body_lead))
    story.append(Paragraph(
        "The design emphasizes fast setup, resilience (graceful fallbacks when cloud access is unavailable), and clarity of outputs. "
        "It runs locally against a sample dataset and can optionally use Cloud Storage (gs://) and Vertex AI.",
        body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("GCP Services Used", h_style))
    bullets = [
        "Cloud Natural Language API (v2): entity and sentiment extraction.",
        "Vertex AI (optional fallback): Gemini models via Vertex when API key path isn't used.",
        "Google Generative AI (Gemini): summarization via API key.",
        "Cloud Storage (optional): load datasets from gs:// when configured.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, body), bulletColor=colors.black) for b in bullets], bulletType='bullet'))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Agentic Workflow", h_style))
    story.append(Paragraph(
        "The agent composes three capabilities into a small chain:", body))
    bullets = [
        "Retrieval: ranks rows by keyword overlap with the query; selects top-k candidates.",
        "Per-document tools: for each candidate, calls GCP Language (entities, sentiment) and Gemini (summary).",
        "Synthesis: concatenates per-document summaries and asks Gemini to produce a concise answer with context.",
        "Resilience: exceptions from any tool are captured; the chain continues with available signals.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, body)) for b in bullets], bulletType='bullet'))
    story.append(Paragraph(
        "Tool routing: if a `GOOGLE_API_KEY` is set, Gemini API is used directly; otherwise, if `USE_VERTEX_SUMMARY=true`, "
        "Vertex AI is used; else a dependency-free local summarizer provides deterministic behavior.", body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Results, Challenges, and Trade-offs", h_style))
    story.append(Paragraph(
        "Outputs include `results.csv` (row-level entities, sentiment, summary), `eda.txt` (dataset stats), and a log. "
        "Keyword retrieval is simple and fast but less robust than embeddings; Language v2 entities may be generic (NUMBER/OTHER) "
        "and do not always include salience; the local summarizer guarantees offline execution but is extractive. The design trades "
        "recall/nuance for simplicity and reliability in a short assignment window.", body))
    story.append(Paragraph(
        "With ADC configured, entities and sentiment populate fully. With only a Gemini API key, summaries still work and the system "
        "provides a useful agent answer with supporting snippets.", body))
    story.append(PageBreak())

    story.append(Paragraph("High-level Architecture", h_style))
    if os.path.exists(diagram_path):
        img = Image(diagram_path, width=6.7*inch, height=3.8*inch)
        story.append(img)
        story.append(Spacer(1, 0.05 * inch))
        story.append(Paragraph("Figure: Data flows from CSV (or GCS) through preprocessing, GCP Language tools, and Gemini summarization. The agent composes retrieval → tools → synthesis.", body))
    else:
        story.append(Paragraph(f"Diagram missing at: {diagram_path}", body))
    story.append(Spacer(1, 0.1 * inch))

    # Agent flow diagram (SVG → PNG on the fly)
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    agent_svg = os.path.join(root, 'diagrams', 'agent_flow.svg')
    story.append(Paragraph("Agent Flow Diagram", h_style))
    try:
        if os.path.exists(agent_svg):
            from svglib.svglib import svg2rlg  # type: ignore
            from reportlab.graphics import renderPM  # type: ignore
            drawing = svg2rlg(agent_svg)
            tmp_png = os.path.join(root, 'docs', 'agent_flow.png')
            renderPM.drawToFile(drawing, tmp_png, fmt='PNG')
            story.append(Image(tmp_png, width=6.7*inch, height=4.2*inch))
        else:
            story.append(Paragraph(f"Diagram missing at: {agent_svg}", body))
    except Exception:
        story.append(Paragraph("Could not render SVG agent flow; ensure svglib is installed.", body))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Implementation Notes", h_style))
    bullets = [
        "Configuration via .env; uses GOOGLE_API_KEY for Gemini path.",
        "Vertex path initialized with project/region; local summarizer guarantees no-internet runs.",
        "VS Code launches run module mode to keep package imports stable.",
        "Errors are surfaced in results CSV fields without stopping the run.",
    ]
    story.append(ListFlowable([ListItem(Paragraph(b, body)) for b in bullets], bulletType='bullet'))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph("Extensions", h_style))
    story.append(Paragraph(
        "Replace keyword retrieval with embeddings + cosine similarity; filter entity types and emit JSON; add more evaluation; "
        "and deploy as a Cloud Run service with GCS triggers.", body))

    doc.build(story)


def build_report_canvas(out_path: str, diagram_path: str):
    c = canvas.Canvas(out_path, pagesize=LETTER)
    width, height = LETTER
    x_margin, y_margin = 54, 60
    y = height - y_margin

    def write_line(text, size=12, dy=16, bold=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.drawString(x_margin, y, text)
        y -= dy

    write_line("Architecture & Agent Design Report", size=18, dy=24, bold=True)
    y -= 8
    write_line("Overview", size=14, dy=20, bold=True)
    write_line("This prototype performs entity extraction, sentiment analysis, and summarization.")
    write_line("It uses GCP Natural Language API for NLP and Gemini for summarization.")

    y -= 10
    write_line("GCP Services Used", size=14, dy=20, bold=True)
    for bullet in [
        "• Cloud Natural Language API (v2)",
        "• Vertex AI (optional)",
        "• Google Generative AI (Gemini)",
        "• Cloud Storage (optional)",
    ]:
        write_line(bullet)

    y -= 10
    write_line("Agentic Workflow", size=14, dy=20, bold=True)
    for bullet in [
        "• Retrieval (keyword scoring) to select top-k rows",
        "• Per-row entities, sentiment, and summarization",
        "• Synthesis of a final answer from summaries",
        "• Graceful fallback if any tool fails",
    ]:
        write_line(bullet)

    c.showPage()
    y = height - y_margin
    write_line("High-level Architecture", size=14, dy=20, bold=True)
    if os.path.exists(diagram_path):
        # Fit image within the page width
        iw, ih = 500, 280
        c.drawImage(diagram_path, x_margin, y - ih, width=width - 2 * x_margin, height=ih, preserveAspectRatio=True, anchor='n')
        y -= ih + 20
    else:
        write_line(f"Diagram missing at: {diagram_path}")

    write_line("Implementation Notes", size=14, dy=20, bold=True)
    for bullet in [
        "• .env config; GOOGLE_API_KEY for Gemini",
        "• Vertex path uses project/region; local summarizer fallback",
        "• VS Code launches as module; stable imports",
        "• Errors captured in outputs without stopping run",
    ]:
        write_line(bullet)

    c.save()


def build_report(out_path: str, diagram_path: str):
    # Try Platypus first; if something goes wrong, fallback to canvas
    try:
        build_report_platypus(out_path, diagram_path)
    except Exception:
        build_report_canvas(out_path, diagram_path)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    out = os.path.join(root, 'docs', 'Architecture_Agent_Design.pdf')
    diagram = os.path.join(root, 'diagrams', 'architecture.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    build_report(out, diagram)
