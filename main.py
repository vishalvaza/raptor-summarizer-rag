import os
import argparse
from src.raptor_rag.engine import RaptorSummarizerRAG
from src.raptor_rag.utils import load_pdf, chunk_text, create_sample_pdf

def run_validation():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, required=True, help="OpenAI API Key")
    args = parser.parse_args()

    pdf_path = "data/sample_test.pdf"
    
    # 1. Create and Load Data
    if not os.path.exists(pdf_path):
        create_sample_pdf(pdf_path)
    
    print("--- STEP 1: Processing PDF ---")
    text = load_pdf(pdf_path)
    chunks = chunk_text(text)
    
    # 2. Initialize Engine
    engine = RaptorSummarizerRAG(api_key=args.key)
    engine.build_tree(chunks)
    
    # 3. Automated Validation
    print("\n--- STEP 2: Running Automated Validation ---")
    test_cases = [
        ("Who is the lead scientist?", "Dr. Elena Vance"),
        ("What is the lab access code?", "ALPHA-99"),
        ("What was the result of the cooling system?", "latency by 45%")
    ]
    
    for query, expected in test_cases:
        ans = engine.query(query)
        print(f"Q: {query}\nAI Answer: {ans}")
        status = "PASS" if expected.lower() in ans.lower() else "FAIL"
        print(f"Status: {status}\n")

if __name__ == "__main__":
    run_validation()
