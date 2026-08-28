import os
import sys

# Make project root importable when running this file directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pymupdf

from agents.performance_agent import PerformanceAgent


def extract_pdf(path: str) -> str:
    """Extract text from a PDF."""
    pdf = pymupdf.open(path)

    extracted_text = ""

    for page_num in range(len(pdf)):
        page = pdf.load_page(page_num)
        extracted_text += page.get_text() + "\n"

    pdf.close()

    return extracted_text


def main():

    # =========================================================
    # 1. PDF TO TEST
    # =========================================================

    pdf_path = (
        "/Users/sanjanapatnaik/Downloads/"
        "Copy of Addiction_Detection_AI_Project_Documentation.pdf"
    )

    print("=" * 80)
    print("PERFORMANCE RAG TEST")
    print("=" * 80)

    print(f"\nLoading PDF:")
    print(pdf_path)

    extracted_text = extract_pdf(pdf_path)

    if not extracted_text.strip():
        print("\nERROR: No text could be extracted from the PDF.")
        return

    print(f"\nExtracted {len(extracted_text)} characters.")


    # =========================================================
    # 2. CREATE source_documents
    #    Same structure used by main.py
    # =========================================================

    source_documents = [
        {
            "filename": os.path.basename(pdf_path),
            "content": extracted_text,
        }
    ]


    # =========================================================
    # 3. INITIALIZE PERFORMANCE AGENT
    # =========================================================

    agent = PerformanceAgent()


    # =========================================================
    # 4. BUILD VECTOR DB ONCE
    # =========================================================

    print("\n" + "=" * 80)
    print("BUILDING UPLOADED-DOCUMENT VECTOR DB")
    print("=" * 80)

    uploaded_vectorstore = agent._build_uploaded_vectorstore(
        source_documents
    )

    if uploaded_vectorstore is None:
        print("\nERROR: Vector DB could not be created.")
        return

    print("\nVector DB created successfully.")


    # =========================================================
    # 5. CLAIMS FROM THE SYSTEM RESPONSE
    # =========================================================
    #
    # These should be claims that your actual LLM response
    # generated and that you want to verify against the PDF.
    #
    # =========================================================

    claims = [

        "Both groups showed leftward asymmetry (AI < 0), indicating larger left NAcc volume.",

        "The CUD group had slightly stronger leftward asymmetry compared to HC.",

        "Using nonlinear SyN registration (ANTsPy) and the Harvard-Oxford atlas, an Asymmetry Index was calculated for each subject.",

        "The project measured left and right nucleus accumbens volumes in Cocaine Use Disorder and Healthy Control participants.",

    ]


    # =========================================================
    # 6. TEST EACH CLAIM
    # =========================================================

    threshold = 0.3

    for claim_number, claim in enumerate(claims, start=1):

        print("\n" + "=" * 80)
        print(f"CLAIM {claim_number}")
        print("=" * 80)
        print(f"Claim: {claim}")

        # Test OUR actual function
        score, top_docs = agent.calculate_factuality_score(
            claim,
            uploaded_vectorstore
        )

        print(f"\nReturned score: {score:.4f}")
        print(f"Returned chunks: {len(top_docs)}")

        print(f"\nThreshold: {threshold}")

        if score >= threshold:
            print("Decision: DETERMINISTIC → SUPPORTED")
        else:
            print("Decision: FALLBACK → LLM")

        print("\nTop returned chunks:")

        for rank, doc in enumerate(top_docs, start=1):
            print(f"\n--- CHUNK {rank} ---")
            print(doc.page_content)
            print(f"Source: {doc.metadata.get('source', 'unknown')}")



if __name__ == "__main__":
    main()