# ===============================================================
# modules/report_generator.py
# Combine figures into one multipage PDF summary
# Supports both Matplotlib and Plotly figures (via Kaleido)
# ===============================================================

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import plotly.io as pio

# ---------------------------------------------------------------
# Helper: Convert Plotly figure to PNG Bytes using Kaleido
# ---------------------------------------------------------------
def plotly_fig_to_png_bytes(plotly_fig, scale=3):
    """
    Convert a Plotly figure to PNG bytes using Kaleido.
    This allows Plotly visualizations to be embedded in PDF reports.
    """
    try:
        png_bytes = pio.to_image(plotly_fig, format="png", scale=scale)
        return BytesIO(png_bytes)
    except Exception as e:
        print(f"[Warning] Could not convert Plotly figure: {e}")
        return None

# ---------------------------------------------------------------
# Main PDF Report Builder
# ---------------------------------------------------------------
def create_report(figures, output_path="outputs/reports/Bibliometric_Report.pdf"):
    """
    Accepts a dict of {title: figure}, where each figure can be:
      - a Matplotlib Figure
      - a Plotly Figure
      - a BytesIO image buffer
    Creates a multi-page PDF with each figure and its caption.
    """
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    for title, fig_data in figures.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))

        # Handle existing BytesIO image
        if isinstance(fig_data, BytesIO):
            img_data = fig_data

        # Handle Plotly figure
        elif hasattr(fig_data, "to_plotly_json"):
            img_data = plotly_fig_to_png_bytes(fig_data)
            if img_data is None:
                continue  # skip if failed

        # Handle Matplotlib figure
        else:
            img_data = BytesIO()
            try:
                fig_data.savefig(img_data, format="png", dpi=300, bbox_inches="tight")
            except Exception as e:
                print(f"[Warning] Failed to render Matplotlib figure: {e}")
                continue
            img_data.seek(0)

        # Add to PDF
        elements.append(Image(img_data, width=480, height=320))
        elements.append(Spacer(1, 20))

    doc.build(elements)
    return output_path
