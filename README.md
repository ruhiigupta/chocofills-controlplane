# ControlPlane.ai 🛡️

## 📌 Project Overview
ControlPlane.ai is an enterprise-grade governance gateway that sits between users and internal AI models. It acts as an invisible shield: intercepting requests, blocking malicious prompt injections (including multimodal attacks), and routing LLM outputs through parallel governance agents before returning the final response.

## 🚀 Pipeline
The application runs one compiled LangGraph pipeline: preflight, target LLM, parallel Performance/Security/Cost evaluation, decision, final response, and audit logging. Preflight blocks never call the target LLM.

## 📁 Strict Modularization Guide
To keep the codebase clean, please keep business logic separated:
* Do not put agent evaluation logic in `main.py`.
* Do not put API routing in the `agents/` folder.
* Use `graph/workflow.py` exclusively for LangGraph routing logic.

## 🏗️ Architecture & Flow
1. **Pre-Flight Input Firewall:** Scans text and extracted image text (via EasyOCR) for prompt injections using `llm-guard`.
2. **Target Execution:** Safely calls the target LLM (e.g., Gemini) only if the prompt passes the firewall.
3. **LangGraph Orchestrator:** Distributes the LLM's response to three parallel governance agents to evaluate risk.
4. **Governance Agents:** 
   * **Security & Orchestration (Ruhi):** Scans output for PII, API keys, and sensitive data leaks. Routes data via LangGraph.
   * **Performance (Sanjana):** Validates factual accuracy and hallucination rates using a local RAG/embedding system.
   * **Cost (Tvisha):** Calculates token usage, budget constraints, and latency.

## 💻 Run Locally
1. Clone the repository and enter it: `cd chocofills-controlplane`
2. Create and activate a virtual environment: `python -m venv venv` then `.\venv\Scripts\Activate.ps1`
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Create `.env` with `GEMINI_API_KEY` and optionally `GEMINI_MODEL` (default: `gemini-3.6-flash`). Evaluation agents optionally use `OPENAI_API_KEY` or `OPENROUTER_API_KEY`; tests do not require network credentials.
5. Start the API: `python -m app.main`
6. Open Swagger at `http://localhost:8000/docs`, or send a request:

```powershell
curl.exe -X POST http://localhost:8000/chat -F "user_id=local" -F "prompt=What is the capital of France?"
```

The response includes the final action, released response when allowed, agent statuses, unified risk, and audit information. Audit entries are appended to `data/audit_log.jsonl`.

## Tests
Run the complete suite:

```powershell
python -m unittest discover -v
python -m unittest discover -s agents\cost_agent -p "test_*.py" -v
```