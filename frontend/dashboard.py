import streamlit as st
import requests

st.set_page_config(
    page_title="ControlPlane.ai | Security Dashboard",
    page_icon="🛡️",
    layout="wide"
)


st.markdown("""
<style>
    /* Hide Streamlit header */
    [data-testid="stHeader"] {
        display: none;
    }

    /* Main page */
    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(214, 248, 214, 0.22),rgba(127, 198, 164, 0.22), transparent 35%),
            radial-gradient(circle at 95% 90%, rgba(214, 248, 214, 0.22), rgba(127, 198, 164, 0.22),rgba(41, 171, 135,0.22), transparent 35%),
            linear-gradient(135deg, #070b14 0%, #0b1220 50%, #111a2b 100%);
        color: #f5f5f0;
    }

    /* Subtle grid */
    .stApp::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.025;
        background-image:
            linear-gradient(rgba(255,255,255,0.8) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.8) 1px, transparent 1px);
        background-size: 48px 48px;
    }

    /* Main content width */
    .block-container {
        max-width: 1200px;
        padding-top: 4rem;
        padding-bottom: 4rem;
    }

    /* Typography */
    h1, h2, h3, p, label {
        color: #f5f5f0 !important;
    }

    /* Selectbox + text area */
    div[data-baseweb="select"] > div,
    textarea {
        background-color: transparent !important;
        border: 1px solid rgba(86, 179, 134, 0.80) !important;
        border-radius: 16px !important;
        color: #f5f5f0 !important;
    }

    /* Selectbox when clicked */
    /* Selectbox focus */
    div[data-baseweb="select"] > div {
        border-color: #56b386 !important;
        box-shadow: none !important;
    }

    /* Override BaseWeb focus styling */
    div[data-baseweb="select"] *:focus {
        outline: none !important;
        box-shadow: none !important;
    }

    textarea {
        padding: 16px !important;
    }

    [data-testid="stTextArea"] > div {
        background-color: transparent !important;
    }

    [data-testid="stTextArea"] textarea {
        background-color: rgba(40, 60, 80, 0.35) !important;
    }

    /* Remove Streamlit selectbox wrapper background */
    [data-testid="stSelectbox"] > div {
        background-color: transparent !important;
    }

    /* Selectbox itself */
    [data-baseweb="select"] > div {
        background-color: rgba(40, 60, 80, 0.35) !important;
        border: 1px solid rgba(127, 198, 164, 0.25) !important;
        border-radius: 16px !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #56b386 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.65rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: none !important;
    }

    .stButton > button:hover {
        background-color: #3EB489 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: none !important;
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stButton > button:focus-visible,
    .stButton > button:active {
        background-color: #3EB489 !important;
        color: #ffffff !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* Dividers */
    hr {
        border-color: rgba(255, 255, 255, 0.12);
    }

   /* Navbar */
    .navbar {
        display: flex;
        align-items: center;
        padding: 18px 24px;
        margin-bottom: -20px;
        background: transparent;
        border: none;
        border-radius: 0;
    }

    .logo {
        font-size: 24px;
        font-weight: 700;
        color: #f5f5f0;
    }

    .nav-left {
        display: flex;
        gap: 28px;
        margin-left: 50px;
        color: #f5f5f0;
    }

    .nav-right {
        display: flex;
        gap: 28px;
        margin-left: auto;
        color: #f5f5f0;
    }

    .nav-left span,
    .nav-right span {
        cursor: pointer;
    }

    /* File uploader */
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    [data-testid="stFileUploader"] section > div {
        background: transparent !important;
        border: none !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

</style>
""", unsafe_allow_html=True)

# Top navigation
current_page = st.query_params.get("page", "home").lower()
if current_page not in {"home", "info"}:
    current_page = "home"

st.markdown(f"""
<div class="navbar">
<div class="logo">ControlPlane.ai</div>
<div class="nav-left">
<a href="?page=home" style="color:#f5f5f0; text-decoration:none;">Home</a>
<a href="?page=info" style="color:#f5f5f0; text-decoration:none;">Info</a>
</div>
<div class="nav-right">
<span>Log in</span>
<span>Sign up</span>
</div>
</div>
""", unsafe_allow_html=True)

st.divider()


def render_monitoring_and_audit():
    try:
        metrics = requests.get("http://localhost:8000/metrics", timeout=5).json()
    except Exception:
        metrics = {}

    if not metrics:
        st.info("No audit records yet. Run an evaluation to create the first record.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Requests", metrics.get("total_requests", 0))
        c2.metric("ALLOW", metrics.get("allowed", 0))
        c3.metric("BLOCK", metrics.get("blocked", 0))
        c4.metric("ESCALATE", metrics.get("escalated", 0))
        c5.metric("REWRITE", metrics.get("rewritten", 0))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Security Score", f"{metrics.get('average_security_score', 0.0):.2f}")
        c2.metric("Avg Performance Score", f"{metrics.get('average_performance_score', 0.0):.2f}")
        c3.metric("Avg Cost Score", f"{metrics.get('average_cost_score', 0.0):.2f}")
        c4.metric("Avg Unified Risk Score", f"{metrics.get('average_unified_risk_score', 0.0):.2f}")

        evaluated_requests = int(metrics.get("evaluated_requests", 0) or 0)
        if evaluated_requests > 0:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Security Accuracy", f"{metrics.get('accuracy', 0.0):.2%}")
            c2.metric("False Positive Rate", f"{metrics.get('false_positive_rate', 0.0):.2%}")
            c3.metric("False Negative Rate", f"{metrics.get('false_negative_rate', 0.0):.2%}")
            c4.metric("Precision", f"{metrics.get('precision', 0.0):.2%}")
            c5.metric("Recall", f"{metrics.get('recall', 0.0):.2%}")
        else:
            st.info("No ground-truth evaluation data available yet.")

    try:
        audit_data = requests.get("http://localhost:8000/audit?limit=20", timeout=5).json()
        records = audit_data.get("records", [])
    except Exception:
        records = []

    if not records:
        st.info("No audit records yet. Run an evaluation to create the first record.")
    else:
        compact_rows = []
        for record in records:
            compact_rows.append({
                "timestamp": record.get("timestamp") or "",
                "request_id": record.get("request_id") or "",
                "sensitivity": record.get("sensitivity") or "",
                "security_score": record.get("security_score"),
                "performance_score": record.get("performance_score"),
                "cost_score": record.get("cost_score"),
                "unified_risk_score": record.get("unified_risk_score"),
                "final_action": record.get("final_action") or "",
            })

        st.dataframe(compact_rows, use_container_width=True, hide_index=True)

        record_ids = [record.get("request_id") for record in records if record.get("request_id")]
        if record_ids:
            selected_request_id = st.selectbox(
                "Inspect a record",
                options=["Select a request"] + record_ids,
                index=0,
            )
            if selected_request_id and selected_request_id != "Select a request":
                selected_record = next((r for r in records if r.get("request_id") == selected_request_id), None)
                if selected_record:
                    st.subheader("Record Details")
                    detail = {
                        "request_id": selected_record.get("request_id"),
                        "timestamp": selected_record.get("timestamp"),
                        "user_id": selected_record.get("user_id"),
                        "use_case": selected_record.get("use_case"),
                        "sensitivity": selected_record.get("sensitivity"),
                        "security_decision": selected_record.get("security_decision"),
                        "security_status": selected_record.get("security_status"),
                        "performance_status": selected_record.get("performance_status"),
                        "cost_status": selected_record.get("cost_status"),
                        "final_action": selected_record.get("final_action"),
                        "expected_action": selected_record.get("expected_action"),
                        "ground_truth": selected_record.get("ground_truth"),
                        "evaluation_result": selected_record.get("evaluation_result"),
                        "user_prompt": selected_record.get("user_prompt") or "[redacted / empty]",
                        "llm_response": selected_record.get("llm_response") or "[redacted / empty]",
                    }
                    st.json(detail)


# ============================================================
# COMMON USE CASE OPTIONS
# ============================================================

use_case_display_options = [
    "Internal Knowledge Copilot",
    "Customer Support Chatbot",
    "Decision Support"
]

use_case_map = {
    "Internal Knowledge Copilot": "internal_copilot",
    "Customer Support Chatbot": "customer_support",
    "Decision Support": "regulated_decision_support"
}


if current_page == "info":
    st.subheader("System Monitoring & Audit")
    render_monitoring_and_audit()
else:
    st.subheader("Flag and Evaluate LLM Responses")

    main_tab_1, main_tab_2 = st.tabs([
        "Generate & Evaluate",
        "Evaluate Existing Response",
    ])

    # ============================================================
    # TAB 1 — GENERATE & EVALUATE
    # ============================================================

    with main_tab_1:

        use_case_display = st.selectbox(
            "Use Case",
            use_case_display_options,
            key="generate_use_case"
        )

        prompt = st.text_area(
            "User Prompt",
            "eg: Is tomato a fruit or a vegetable?",
            height=150,
            key="generate_prompt"
        )

        uploaded_file = st.file_uploader(
            "Upload a file (optional)",
            type=["pdf", "png", "jpg", "jpeg"],
            key="generate_file"
        )

        if st.button("Evaluate with ControlPlane", key="generate_button"):

            with st.spinner("Intercepting and evaluating via ControlPlane.ai..."):

                try:
                    files = None

                    if uploaded_file:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type
                            )
                        }

                    response = requests.post(
                        "http://localhost:8000/chat",
                        data={
                            "user_id": "auditor_1",
                            "prompt": prompt,
                            "use_case": use_case_map[use_case_display],
                            "evaluation_mode": "generate"
                        },
                        files=files
                    )

                    data = response.json()

                    st.divider()

                    if data.get("status") == "BLOCKED":

                        st.error("🚫 BLOCKED AT INGRESS (PRE-FLIGHT)")

                        governance = data.get("governance", {})

                        st.write(
                            f"**Reason:** "
                            f"{governance.get('preflight_reason', 'Policy violation detected.')}"
                        )

                    else:

                        final_action = data.get("status", "ALLOW")

                        if final_action in ["BLOCK", "FAIL"]:
                            st.error("🚫 BLOCKED AT EGRESS (POST-FLIGHT EVALUATION)")

                        elif final_action == "REDACT":
                            st.warning("RESPONSE REQUIRES REDACTION")

                        elif final_action == "REQUIRE_APPROVAL":
                            st.warning("HUMAN APPROVAL REQUIRED")

                        elif final_action == "FLAG":
                            st.warning("⚠️ FLAGGED FOR HUMAN REVIEW")

                        else:
                            st.success("✅ ALLOWED")

                        st.markdown("**LLM Response**")
                        st.info(data.get("llm_response", ""))

                        governance = data.get("governance", {})

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.metric(
                                "Security Risk",
                                f"{governance.get('security_risk', 0):.2f}"
                            )

                        with col2:
                            st.metric(
                                "Performance Score",
                                f"{governance.get('performance_score', 0):.2f}"
                            )

                        with col3:
                            st.metric(
                                "Cost Risk",
                                f"{governance.get('cost_risk', 0):.2f}"
                            )

                        with col4:
                            st.metric(
                                "Unified Risk",
                                f"{governance.get('unified_risk_score', 0):.2f}"
                            )

                        with col5:
                            st.metric(
                                "Action",
                                governance.get("final_action", "UNKNOWN")
                            )

                except Exception as e:

                    st.error(f"Failed to connect to API: {e}")

                    st.info(
                        "Make sure the backend is running: "
                        "`python -m app.main`"
                    )

    # ============================================================
    # TAB 2 — EVALUATE EXISTING RESPONSE
    # ============================================================

    with main_tab_2:

        use_case_display_existing = st.selectbox(
            "Use Case",
            use_case_display_options,
            key="existing_use_case"
        )

        existing_prompt = st.text_area(
            "User Prompt",
            "Enter the original user prompt...",
            height=130,
            key="existing_prompt"
        )

        existing_response = st.text_area(
            "LLM Response",
            "Paste the LLM response you want ControlPlane to evaluate...",
            height=200,
            key="existing_response"
        )

        uploaded_file_existing = st.file_uploader(
            "Upload a file (optional)",
            type=["pdf", "png", "jpg", "jpeg"],
            key="existing_file"
        )

        if st.button("Evaluate Response", key="existing_button"):

            with st.spinner("Evaluating response via ControlPlane.ai..."):

                try:
                    files = None

                    if uploaded_file_existing:
                        files = {
                            "file": (
                                uploaded_file_existing.name,
                                uploaded_file_existing.getvalue(),
                                uploaded_file_existing.type
                            )
                        }

                    response = requests.post(
                        "http://localhost:8000/chat",
                        data={
                            "user_id": "auditor_1",
                            "prompt": existing_prompt,
                            "use_case": use_case_map[use_case_display_existing],
                            "evaluation_mode": "evaluate_existing",
                            "existing_response": existing_response
                        },
                        files=files
                    )

                    data = response.json()

                    st.divider()

                    if data.get("status") == "BLOCKED":

                        st.error("🚫 BLOCKED AT INGRESS (PRE-FLIGHT)")

                        governance = data.get("governance", {})

                        st.write(
                            f"**Reason:** "
                            f"{governance.get('preflight_reason', 'Policy violation detected.')}"
                        )

                    else:

                        final_action = data.get("status", "ALLOW")

                        if final_action in ["BLOCK", "FAIL"]:
                            st.error("🚫 BLOCKED AT EGRESS (POST-FLIGHT EVALUATION)")

                        elif final_action == "REDACT":
                            st.warning("RESPONSE REQUIRES REDACTION")

                        elif final_action == "REQUIRE_APPROVAL":
                            st.warning("HUMAN APPROVAL REQUIRED")

                        elif final_action == "FLAG":
                            st.warning("⚠️ FLAGGED FOR HUMAN REVIEW")

                        else:
                            st.success("✅ ALLOWED")

                        governance = data.get("governance", {})

                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            st.metric(
                                "Security Risk",
                                f"{governance.get('security_risk', 0):.2f}"
                            )

                        with col2:
                            st.metric(
                                "Performance Score",
                                f"{governance.get('performance_score', 0):.2f}"
                            )

                        with col3:
                            st.metric(
                                "Cost Risk",
                                f"{governance.get('cost_risk', 0):.2f}"
                            )

                        with col4:
                            st.metric(
                                "Unified Risk",
                                f"{governance.get('unified_risk_score', 0):.2f}"
                            )

                        with col5:
                            st.metric(
                                "Action",
                                governance.get("final_action", "UNKNOWN")
                            )

                except Exception as e:

                    st.error(f"Failed to connect to API: {e}")

                    st.info(
                        "Make sure the backend is running: "
                        "`python -m app.main`"
                    )