import PyPDF2
from reportlab.pdfgen import canvas
import os

def load_pdf(file_path):
    text = ""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return ""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content = page.extract_text()
            if content: text += content + "\n"
    return text

def chunk_text(text, chunk_size=1000):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def create_sample_pdf(output_path):
    """Explicitly generates a PDF with testable data."""
    folder = os.path.dirname(output_path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    
    c = canvas.Canvas(output_path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "PROJECT AETHER: OFFICIAL REPORT 2026")
    c.setFont("Helvetica", 12)
    lines = [
        "Lead Scientist: Dr. Elena Vance",
        "Mission Objective: Reduce latency in neural-liquid cooling.",
        "Key Finding: Latency was reduced by exactly 45 percent.",
        "Secure Lab Access Code: ALPHA-99",
        "Budget Allocated: 12.5 Million USD",
        "End of Report."
    ]
    y = 700
    for line in lines:
        c.drawString(100, y, line)
        y -= 30
    c.showPage()
    c.save()
    print(f"--- SUCCESS: Created {output_path} ---")
