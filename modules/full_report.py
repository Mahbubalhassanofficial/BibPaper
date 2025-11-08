# ===============================================================
# modules/full_report.py
# One-click integrated bibliometric report generator
# ===============================================================

import os
from io import BytesIO
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

def generate_full_report(df: pd.DataFrame, figs_dict: dict, output_dir="outputs/reports"):
    """Generates a full bibliometric PDF report."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"Bibliometric_Full_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf_path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Bibliometric Analysis Full Report</b>", styles["Title"]))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    for title, fig in figs_dict.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        if fig is not None:
            img_data = BytesIO()
            fig.savefig(img_data, format="png", dpi=300, bbox_inches="tight")
            img_data.seek(0)
            elements.append(Image(img_data, width=460, height=320))
        elements.append(Spacer(1, 16))
        elements.append(PageBreak())

    elements.append(Paragraph("<b>End of Report</b>", styles["Heading3"]))
    doc.build(elements)
    return pdf_path
