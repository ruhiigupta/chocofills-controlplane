from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
import uvicorn
import pymupdf  
from llm_guard.input_scanners import PromptInjection

# Import our LangGraph State & Workflow
from graph.workflow import controlplane_graph
from graph.state import ControlPlaneState

app = FastAPI(title="ControlPlane.ai Gateway")

# 1. Initialize Security Scanner
injection_scanner = PromptInjection(threshold=0.5)
ocr_reader = None

def get_ocr_reader():
    global ocr_reader
    if ocr_reader is None:
        import easyocr
        print("[OCR] Initializing EasyOCR reader...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)
    return ocr_reader

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    extracted_text = ""
    if filename.lower().endswith('.pdf'):
        pdf_document = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        try:
            reader = get_ocr_reader()
            results = reader.readtext(file_bytes, detail=0)
            extracted_text = " ".join(results)
        except Exception as e:
            print(f"[OCR Error] Could not process image: {e}")
            raise HTTPException(status_code=500, detail="OCR processing failed.")
    return extracted_text

def call_target_llm(prompt: str) -> str:
    """Simulated call to the target enterprise model."""
    return "This is a simulated response from Gemini containing employee account details: email=john.doe@enterprise.com."

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    final_input_text = ""

    if file:
        file_bytes = await file.read()
        extracted_text = extract_text_from_file(file_bytes, file.filename)
        final_input_text += extracted_text + "\n"
    
    if prompt:
        final_input_text += prompt

    if not final_input_text.strip():
        raise HTTPException(status_code=400, detail="Must provide either a text prompt or a file.")

    # Pre-Flight Firewall 
    sanitized_prompt, is_valid, risk_score = injection_scanner.scan(final_input_text)
    
    if not is_valid:
        return {
            "status": "BLOCKED",
            "reason": "Prompt Injection / Malicious Payload Detected",
            "preflight_risk_score": risk_score,
            "cost_incurred": "$0.00"
        }
    
    # Target LLM Execution 
    llm_response = call_target_llm(final_input_text)
    
    # LangGraph Orchestration
    initial_state: ControlPlaneState = {
        "user_id": user_id,
        "user_prompt": final_input_text.strip(),
        "llm_response": llm_response,
        "model_name": "gemini-1.5-pro",
        "preflight_risk_score": risk_score,
        "performance_score": 0.0,
        "performance_status": "PENDING",
        "factual_findings": [],
        "relevance_findings": [],
        "security_score": 0.0,
        "security_status": "PENDING",
        "security_findings": [],
        "cost_score": 0.0,
        "cost_status": "PENDING",
        "input_tokens": len(final_input_text.split()),
        "output_tokens": len(llm_response.split()),
        "estimated_cost": 0.0004,
        "ttft_latency_ms": 280.0,
        "unified_risk_score": 0.0,
        "final_action": "ALLOW",
        "audit_log": {}
    }

    final_state = controlplane_graph.invoke(initial_state)

    return {
        "status": final_state["final_action"],
        "processed_input": final_input_text.strip(),
        "llm_response": final_state["llm_response"],
        "governance": {
            "unified_risk_score": final_state["unified_risk_score"],
            "security_status": final_state["security_status"],
            "performance_status": final_state["performance_status"],
            "cost_status": final_state["cost_status"],
            "estimated_cost": final_state["estimated_cost"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)