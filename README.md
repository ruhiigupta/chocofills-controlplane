# ControlPlane.ai 🛡️

## 📌 Project Overview
ControlPlane.ai is an enterprise-grade governance gateway that sits between users and internal AI models. It acts as an invisible shield: intercepting requests, blocking malicious prompt injections (including multimodal attacks), and routing LLM outputs through parallel governance agents before returning the final response.

## 🏗️ Architecture & Flow
1. **Pre-Flight Input Firewall:** Scans text and extracted image text (via EasyOCR) for prompt injections using `llm-guard`.
2. **Target Execution:** Safely calls the target LLM (e.g., Gemini) only if the prompt passes the firewall.
3. **LangGraph Orchestrator:** Distributes the LLM's response to three parallel governance agents to evaluate risk.
4. **Governance Agents:** 
   * **Security (Tvisha):** Scans output for PII, API keys, and sensitive data leaks using regex and NLP.
   * **Performance (Sanjana):** Validates factual accuracy and hallucination rates using a local RAG/embedding system.
   * **Cost (Ruhi/Team):** Calculates token usage, budget constraints, and latency.

## 🚀 Current Progress (Baseline)
* **API Gateway (`app/main.py`):** FastAPI server is live with multimodal file upload support and active `llm-guard` injection blocking.
* **Orchestrator (`graph/workflow.py`):** LangGraph state definition and node routing skeleton are configured.
* **Security Module (`agents/security_agent.py`):** Initial regex scanning logic established.

## 📁 Strict Modularization Guide
To keep the codebase clean, please keep business logic separated:
* Do not put agent evaluation logic in `main.py`.
* Do not put API routing in the `agents/` folder.
* Use `graph/workflow.py` exclusively for LangGraph routing logic.

## 🌿 Git Branching Strategy
**Do not push directly to `main`.** Please checkout your assigned branch before coding:
* `feature/gateway-and-orchestration` (Ruhi)
* `feature/security-agent` (Tvisha)
* `feature/performance-agent` (Sanjana)

## 💻 How to Run Locally
1. Activate the virtual environment: `.\venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the server: `python app/main.py`
4. Access the Swagger Testing UI: `http://localhost:8000/docs`