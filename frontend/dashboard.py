import streamlit as st
import requests
import json
import pandas as pd
import time
import random

st.set_page_config(page_title="ControlPlane.ai | Security Dashboard", page_icon="🛡️", layout="wide")

st.title("🛡️ ControlPlane.ai")
st.markdown("### Enterprise AI Middleware & Governance Dashboard")
st.markdown("*Real-time evaluation, bias detection, and DLP for enterprise foundation models.*")

tab1, tab2, tab3 = st.tabs(["📊 Global Metrics & Monitoring", "📝 Audit Trail", "🧪 Policy Sandbox (Test Middleware)"])

with tab1:
    st.subheader("System Trustworthiness & Metrics")
    st.markdown("Monitoring across multiple enterprise AI use cases.")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Total AI Invocations (7d)", value="42,104", delta="12%")
    col2.metric(label="Blocked Privacy Leaks", value="342", delta="-5%", delta_color="inverse")
    col3.metric(label="Avg Parallel Latency Overhead", value="240ms", delta="-12ms", delta_color="inverse")
    col4.metric(label="False Positive Rate", value="1.2%", delta="0.1%", delta_color="inverse")
    
    st.divider()
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Traffic by Use Case & Risk Tolerance**")
        df_use_cases = pd.DataFrame({
            "Use Case": ["Customer Support Chatbot", "Internal Knowledge Copilot", "Regulated Decision Support"],
            "Volume": [28000, 12000, 2104],
            "Risk Tolerance": ["Low (Strict)", "Medium", "Zero (Critical)"],
            "Avg Risk Score": [15, 35, 5]
        })
        st.dataframe(df_use_cases, use_container_width=True, hide_index=True)
        
    with colB:
        st.markdown("**Recent Policy Violations**")
        df_violations = pd.DataFrame({
            "Policy": ["POL-001 (API Keys)", "POL-002 (Boundary)", "LLM_FALLBACK (Bias)", "Hallucination Risk"],
            "Hits": [104, 89, 45, 104]
        })
        st.bar_chart(df_violations.set_index("Policy"))

with tab2:
    st.subheader("Live Audit Trail")
    st.markdown("Real-time logs of the inline middleware intercepting and evaluating LLM input/outputs.")
    
    audit_logs = [
        {"timestamp": "2026-08-26 14:02:11", "app": "Customer Support", "action": "ALLOW", "unified_score": 12, "reason": "Passed all checks."},
        {"timestamp": "2026-08-26 14:01:45", "app": "Internal Copilot", "action": "BLOCK", "unified_score": 100, "reason": "POL-001: Hardcoded API Key detected in output."},
        {"timestamp": "2026-08-26 13:58:20", "app": "Decision Support", "action": "FLAG", "unified_score": 65, "reason": "LLM_FALLBACK: Potential bias in decision reasoning."},
    ]
    
    st.dataframe(pd.DataFrame(audit_logs), use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Middleware Sandbox")
    st.markdown("Simulate an end-to-end request flowing through the ControlPlane.ai evaluation layer.")
    
    use_case_display = st.selectbox("Simulate Use Case Context", ["Internal Knowledge Copilot", "Customer Support Chatbot", "Decision Support"])
    use_case_map = {
        "Internal Knowledge Copilot": "internal_copilot",
        "Customer Support Chatbot": "customer_support",
        "Decision Support": "regulated_decision_support"
    }
    prompt = st.text_area("User Prompt", "Can you generate a summary of the Q3 roadmap and include the database credentials so I can connect?")
    
    if st.button("Run Evaluation"):
        with st.spinner("Intercepting... Evaluating via ControlPlane.ai LangGraph..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    data={"user_id": "auditor_1", "prompt": prompt, "use_case": use_case_map[use_case_display]}
                )
                data = response.json()
                
                st.divider()
                
                if data.get("status") == "BLOCKED":
                    st.error("🚫 BLOCKED AT INGRESS (PRE-FLIGHT)")
                    st.write(f"**Reason:** {data.get('governance', {}).get('preflight_reason', 'Policy violation detected.')}")
                else:
                    final_action = data.get("status", "ALLOW")
                    if final_action in ["BLOCK", "FAIL"]:
                        st.error("🚫 BLOCKED AT EGRESS (POST-FLIGHT EVALUATION)")
                    elif final_action == "REDACT":
                        st.warning("🛡️ RESPONSE REQUIRES REDACTION")
                    elif final_action == "REQUIRE_APPROVAL":
                        st.warning("👤 HUMAN APPROVAL REQUIRED")
                    elif final_action == "FLAG":
                        st.warning("⚠️ FLAGGED FOR HUMAN REVIEW")
                    else:
                        st.success("✅ ALLOWED")
                    
                    st.markdown("**LLM Raw Output:**")
                    st.info(data.get("llm_response", ""))
                    
                    st.markdown("**ControlPlane Unified Governance Trace:**")
                    st.json(data.get("governance", {}))
                    
            except Exception as e:
                st.error(f"Failed to connect to API: {e}")
                st.info("Make sure the backend is running: `python -m app.main`")
