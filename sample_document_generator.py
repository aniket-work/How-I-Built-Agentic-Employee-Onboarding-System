# sample_document_generator.py
from fpdf import FPDF, XPos, YPos
import os


def create_hr_policies():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)  # Use core font instead of Arial

    # Header
    pdf.set_font(size=14, style='B')
    pdf.cell(200, 10, text="Aniket AI Company HR Policies", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    content = [
        ("Leave Policy:", [
            "- 18 paid vacation days per year",
            "- 5 sick days with doctor's note required after 3 consecutive days",
            "- Parental leave: 12 weeks paid for primary caregivers"
        ]),
        ("Remote Work Policy:", [
            "- Hybrid model: 3 days office/2 days remote",
            "- Home office stipend: $500/year",
            "- Core hours: 10 AM - 3 PM local time"
        ]),
        ("Code of Conduct:", [
            "1. All employees must complete AI ethics training",
            "2. Strict ban on using customer data for model training",
            "3. Mandatory security clearance for LLM projects"
        ])
    ]

    pdf.set_font(size=12)
    for section, items in content:
        pdf.set_font(style='B')
        pdf.cell(0, 10, text=section, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(style='')
        for item in items:
            pdf.multi_cell(pdf.w - 2 * pdf.l_margin, 7, text=item)  # Fixed width calculation
        pdf.ln(5)

    pdf.output("hr_policies.pdf")
    print("Generated hr_policies.pdf")


def create_technical_handbook():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)

    # Header
    pdf.set_font(size=14, style='B')
    pdf.cell(200, 10, text="Aniket AI Technical Handbook", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.ln(10)

    sections = [
        ("Development Standards:", [
            "All ML models must include:",
            " - Model cards with performance metrics",
            " - Bias mitigation documentation",
            " - Versioning using MLflow",
            "\nCode review requirements:",
            " - 2 senior engineer approvals for production models",
            " - Static analysis with Bandit and Pylint"
        ]),
        ("AI Infrastructure:", [
            "Approved Tools:",
            "- Vector DB: ChromaDB",
            "- LLM Gateway: Groq Cloud",
            "- Monitoring: Prometheus + Grafana",
            "\nSecurity Protocols:",
            "1. All API keys rotated every 90 days",
            "2. VPC isolation for training clusters",
            "3. Daily vulnerability scans"
        ])
    ]

    for title, content in sections:
        pdf.set_font(style='B')
        pdf.cell(0, 10, text=title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font(style='')
        pdf.multi_cell(pdf.w - 2 * pdf.l_margin, 7, text='\n'.join(content))
        pdf.ln(5)

    pdf.output("technical_handbook.pdf")
    print("Generated technical_handbook.pdf")


if __name__ == "__main__":
    # Create sample documents
    create_hr_policies()
    create_technical_handbook()

    # Initialize and ingest documents
    from ai_onboarding import HRAssistant

    assistant = HRAssistant()

    print("\nIngesting sample documents...")
    assistant.db.ingest_document("hr_policies.pdf")
    assistant.db.ingest_document("technical_handbook.pdf")

    # Demo query
    print("\nSample query results:")
    query = "How many vacation days do I get as a new employee?"
    print(f"Q: {query}")
    print(f"A: {assistant.answer_query(query)}")