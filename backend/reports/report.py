from __future__ import annotations

import os
from datetime import datetime
from typing import Dict, Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(BASE_DIR, "reports_out")
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_application_report(app_row, extracted: Dict[str, Any]) -> str:
    path = os.path.join(REPORTS_DIR, f"application_{app_row.id}_report.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    width, height = A4
    y = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "AI Credit Underwriting Report")
    y -= 30
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated: {datetime.utcnow().isoformat()}Z")
    y -= 20
    c.drawString(50, y, f"Applicant: {app_row.applicant_name}")
    y -= 15
    c.drawString(50, y, f"DOB: {app_row.dob}")
    y -= 15
    c.drawString(50, y, f"Aadhaar: {app_row.aadhaar_number}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Extracted Documents")
    y -= 18
    c.setFont("Helvetica", 10)

    for k, v in extracted.items():
        c.drawString(50, y, f"{k}:")
        y -= 15
        for kk, vv in (v or {}).items():
            c.drawString(70, y, f"{kk}: {vv}")
            y -= 12
            if y < 80:
                c.showPage(); y = height - 50

    c.showPage()
    c.save()
    return path

