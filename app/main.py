import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # Disables broken TF on Mac

# ... (rest of your imports)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import Optional
import uvicorn
import pymupdf  
from llm_guard.input_scanners import PromptInjection
from dotenv import load_dotenv

load_dotenv()

# Import our LangGraph State & Workflow
from graph.workflow import CorrectedPromptInjection, controlplane_graph, security_agent_instance
from graph.state import ControlPlaneState

app = FastAPI(title="ControlPlane.ai Gateway")

# 1. Initialize Security Scanner
injection_scanner = CorrectedPromptInjection(threshold=0.5)
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

from google import genai
import os
import time

def call_target_llm(prompt: str) -> tuple[str, bool]:
    """Real call to the target enterprise Gemini model."""
    api_key = os.getenv("GEMINI_API_KEY", "dummy_key_if_none")
    if api_key == "dummy_key_if_none":
        print("[Warning] GEMINI_API_KEY not set, using simulated response.")
        return (
            "This is a simulated response from Gemini containing employee account details: "
            "email=john.doe@enterprise.com.",
            False,
        )

    model = os.getenv("GEMINI_MODEL", "google/gemma-3-4b-it")
    last_error = ""

    for attempt in range(3):
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text, False
        except Exception as e:
            last_error = str(e)
            transient = any(
                marker in last_error.lower()
                for marker in ("429", "500", "502", "503", "504", "unavailable", "resource exhausted")
            )
            if not transient or attempt == 2:
                break
            time.sleep(2 ** attempt)

    print(f"[LLM Error] {last_error}")
    return f"Error communicating with Gemini: {last_error}", True

@app.post("/chat")
async def chat_endpoint(
    user_id: str = Form(...),
    use_case: str = Form("internal_copilot"),
    prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    final_input_text = ""
    destination = "external_vendor"

    if file:
        file_bytes = await file.read()
        extracted_text = extract_text_from_file(file_bytes, file.filename)
        final_input_text += extracted_text + "\n"
    
    if prompt:
        final_input_text += prompt

    if not final_input_text.strip():
        raise HTTPException(status_code=400, detail="Must provide either a text prompt or a file.")

    # LangGraph owns the complete request lifecycle, including preflight and target execution.
    initial_state: ControlPlaneState = {
        "user_id": user_id,
        "use_case": use_case,
        "source": "internal_api",
        "destination": "external_vendor",
        "user_prompt": final_input_text.strip(),
        "system_prompt": None,
        "source_documents": [{"filename": file.filename, "content": extracted_text}] if file else [],
        "llm_response": "",
        "model_name": os.getenv("GEMINI_MODEL", "gemini-3.6-flash"),
        "target_llm": call_target_llm,
        "preflight_scanner": injection_scanner.scan,
        "llm_failed": False,
        "preflight_risk_score": 0.0,
        "preflight_blocked": False,
        "preflight_reason": "",
        "preflight_findings": [],
        "performance_score": 0.0,
        "performance_status": "PENDING",
        "factual_findings": [],
        "relevance_findings": [],
        "security_score": 0.0,
        "security_status": "PENDING",
        "security_decision": "PENDING",
        "security_findings": [],
        "matched_policies": [],
        "policy_source": "PENDING",
        "cost_score": 0.0,
        "cost_status": "PENDING",
        "input_tokens": 0,
        "output_tokens": 0,
        "estimated_cost": 0.0,
        "ttft_latency_ms": 0.0,
        "cost_agent": {},
        "tool_calls": [],
        "unified_risk_score": 0.0,
        "final_action": "PENDING",
        "audit_log": {},
        "final_response": "",
    }

    final_state = controlplane_graph.invoke(initial_state)

    return {
        "status": "BLOCKED" if final_state["preflight_blocked"] else final_state["final_action"],
        "processed_input": final_input_text.strip(),
        "llm_response": final_state.get("llm_response", "") if final_state["final_action"] == "ALLOW" else "",
        "final_response": final_state.get("final_response", ""),
        "governance": {
            "unified_risk_score": final_state["unified_risk_score"],
            "preflight_risk_score": final_state["preflight_risk_score"],
            "preflight_blocked": final_state["preflight_blocked"],
            "preflight_reason": final_state["preflight_reason"],
            "security_status": final_state["security_status"],
            "security_decision": final_state["security_decision"],
            "security_findings": final_state["security_findings"],
            "matched_policies": final_state["matched_policies"],
            "policy_source": final_state["policy_source"],
            "performance_status": final_state["performance_status"],
            "cost_status": final_state["cost_status"],
            "estimated_cost": final_state["estimated_cost"],
            "audit_log": final_state["audit_log"],
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
