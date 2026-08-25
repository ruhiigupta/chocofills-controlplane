from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
import uvicorn
import pymupdf # PyMuPDF for documents
import easyocr # For images
import io
from llm_guard.input_scanners import PromptInjection

app = FastAPI(title="ControlPlane.ai Gateway")

# 1. Initialize Security Scanners
injection_scanner = PromptInjection(threshold=0.5)
reader = easyocr.Reader(['en'], gpu=False) # Keep GPU false for local hackathon dev

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Helper function to extract text from PDFs and Images."""
    extracted_text = ""
    
    if filename.lower().endswith('.pdf'):
        # Parse PDF
        pdf_document = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()
            
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Parse Image via OCR
        results = reader.readtext(file_bytes, detail=0)
        extracted_text = " ".join(results)
        
    return extracted_text

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    final_input_text = ""

    # ==========================================
    # STEP 0: Multimodal Extraction
    # ==========================================
    if file:
        print(f"[Processing File] {file.filename}")
        file_bytes = await file.read()
        extracted_text = extract_text_from_file(file_bytes, file.filename)
        final_input_text += extracted_text + "\n"
    
    if prompt:
        final_input_text += prompt

    if not final_input_text.strip():
        raise HTTPException(status_code=400, detail="Must provide either a text prompt or a file.")

    print(f"\n[Evaluating Input] Text to scan: {final_input_text.strip()[:100]}...")

    # ==========================================
    # STEP 1: The Pre-Flight Firewall (Input Check)
    # ==========================================
    sanitized_prompt, is_valid, risk_score = injection_scanner.scan(final_input_text)
    
    if not is_valid:
        print(f"[BLOCKED] Prompt Injection Detected! Risk Score: {risk_score}")
        return {
            "status": "BLOCKED",
            "reason": "Prompt Injection / Malicious Payload Detected",
            "risk_score": risk_score,
        }

    print(f"[PASSED] Input is clean. Proceeding to target LLM.")
    
    # ==========================================
    # STEP 2 & 3: LLM Execution & LangGraph
    # ==========================================
    # Target LLM call and your 3 agents will go here next.
    
    return {
        "status": "ALLOW",
        "processed_text": final_input_text.strip(),
        "preflight_risk_score": risk_score
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)