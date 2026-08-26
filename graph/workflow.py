from langgraph.graph import StateGraph, START, END
from graph.state import ControlPlaneState
from agents.security_agent import SecurityAgent
from schemas.core import EvaluationTrace
from agents.evaluators import (
    ClaimExtractionAgent, 
    FactualityAgent, 
    RelevanceAgent, 
    MockVectorDBRetriever,
    IndependentCriticAgent
)
from agents.policy_engine import PolicyEngine

# Initialize the real Agents & Engines
security_agent_instance = SecurityAgent()
claim_extractor = ClaimExtractionAgent()
factuality_agent = FactualityAgent()
relevance_agent = RelevanceAgent()
retriever = MockVectorDBRetriever()
critic_agent = IndependentCriticAgent()
policy_engine = PolicyEngine()

def security_agent_node(state: ControlPlaneState) -> dict:
    """Executes real-time security scanning on the LLM's response."""
    llm_response = state.get("llm_response", "")
    user_prompt = state.get("user_prompt", "")
    
    # Run the real evaluation using the regex patterns + LLM context
    results = security_agent_instance.scan_output(
        llm_response=llm_response,
        user_prompt=user_prompt,
        source=state.get("source", "internal_api"),
        dest=state.get("destination", "external_vendor"),
    )
    
    print(f"[LangGraph Node - Security] Score: {results['security_score']} | Status: {results['security_status']}")
    
    # Return ONLY the updated fields
    return {
        "security_score": results["security_score"],
        "security_status": results["security_status"],
        "security_decision": results["security_decision"],
        "security_findings": results["security_findings"],
        "matched_policies": results["matched_policies"],
        "policy_source": results["policy_source"]
    }

def cost_agent_node(state: ControlPlaneState) -> dict:
    # Dummy placeholder for Cost Agent
    return {
        "cost_score": 5.0,
        "cost_status": "PASS",
        "estimated_cost": 0.002
    }

# --- Performance Evaluation Pipeline ---

def init_trace_node(state: ControlPlaneState) -> ControlPlaneState:
    if "evaluation_trace" not in state:
        state["evaluation_trace"] = EvaluationTrace(
            user_prompt=state.get("user_prompt", ""),
            llm_response=state.get("llm_response", "")
        )
    return state

def extract_claims_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    trace.claims = claim_extractor.extract(trace.llm_response)
    print(f"[LangGraph Node - Claim Extractor] Extracted {len(trace.claims)} claims.")
    return state

def retrieve_evidence_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    for claim in trace.claims:
        trace.retrieved_evidence[claim.id] = retriever.retrieve(claim.text)
    print(f"[LangGraph Node - RAG Retriever] Fetched evidence for {len(trace.claims)} claims.")
    return state

def evaluate_factuality_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    trace.factuality_metrics = [] # Reset for idempotency
    for claim in trace.claims:
        evidence = trace.retrieved_evidence.get(claim.id, [])
        metric = factuality_agent.evaluate(claim, evidence)
        trace.factuality_metrics.append(metric)
    print(f"[LangGraph Node - Factuality Agent] Evaluated {len(trace.factuality_metrics)} claims.")
    return state

def evaluate_relevance_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    trace.relevance_metrics = []
    for claim in trace.claims:
        metric = relevance_agent.evaluate(trace.user_prompt, claim)
        trace.relevance_metrics.append(metric)
    print(f"[LangGraph Node - Relevance Agent] Evaluated {len(trace.relevance_metrics)} claims.")
    return state

def independent_critic_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    trace.critic_metrics = []
    
    # Map metrics for easy lookup
    fact_map = {m.claim_id: m for m in trace.factuality_metrics}
    rel_map = {m.claim_id: m for m in trace.relevance_metrics}
    
    for claim in trace.claims:
        evidence = trace.retrieved_evidence.get(claim.id, [])
        fact_metric = fact_map.get(claim.id)
        rel_metric = rel_map.get(claim.id)
        
        if fact_metric and rel_metric:
            critic_metric = critic_agent.evaluate(trace.user_prompt, claim, evidence, fact_metric, rel_metric)
            trace.critic_metrics.append(critic_metric)
            
    print(f"[LangGraph Node - Critic Agent] Verified {len(trace.critic_metrics)} claims.")
    return state

def policy_engine_node(state: ControlPlaneState) -> ControlPlaneState:
    trace = state["evaluation_trace"]
    decision = policy_engine.evaluate_trace(trace)
    trace.policy_decision = decision
    print(f"[LangGraph Node - Policy Engine] Decision: {decision.final_action}")
    return state

# --- Routing / Decision Layer (Final Merge) ---


def decision_layer_node(state: ControlPlaneState) -> dict:
    # Weighted Unified Risk Score Calculation
    sec = state.get("security_score", 0)
    perf = state.get("performance_score", 0)
    cost = state.get("cost_score", 0)
    
    # Weighting: Security 40%, Performance 40%, Cost 20%
    unified_risk = (0.4 * sec) + (0.4 * perf) + (0.2 * cost)
    
    use_case = state.get("use_case", "internal_copilot")

    # Policy decisions are authoritative; the score remains an observability metric.
    security_decision = state.get("security_decision", "UNKNOWN")
    if security_decision == "BLOCK":
        final_action = "BLOCK"
    elif security_decision == "REDACT":
        final_action = "REDACT"
    elif security_decision == "REQUIRE_APPROVAL":
        final_action = "REQUIRE_APPROVAL"
    elif security_decision == "FLAG":
        final_action = "FLAG"
    elif security_decision == "ALLOW":
        final_action = "ALLOW"
    else:
        final_action = "FLAG"
        
    print(f"[LangGraph Decision] Use Case: {use_case} | Action: {final_action} | Unified Risk: {unified_risk}")
    
    # Audit Trail Logging (Mock writing to a DB)
    audit_entry = {
        "user_id": state.get("user_id"),
        "use_case": use_case,
        "unified_risk": unified_risk,
        "action": final_action,
        "security_findings": state.get("security_findings", [])
    }
    
    import json
    import os
    audit_file = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.jsonl")
    os.makedirs(os.path.dirname(audit_file), exist_ok=True)
    with open(audit_file, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")
        
    return {
        "unified_risk_score": unified_risk,
        "final_action": final_action
    }

# --- Construct the LangGraph Workflow ---
builder = StateGraph(ControlPlaneState)

builder.add_node("security_agent", security_agent_node)
builder.add_node("cost_agent", cost_agent_node)
builder.add_node("init_trace", init_trace_node)
builder.add_node("extract_claims", extract_claims_node)
builder.add_node("retrieve_evidence", retrieve_evidence_node)
builder.add_node("evaluate_factuality", evaluate_factuality_node)
builder.add_node("evaluate_relevance", evaluate_relevance_node)
builder.add_node("independent_critic", independent_critic_node)
builder.add_node("policy_engine", policy_engine_node)
builder.add_node("decision_layer", decision_layer_node)

# Start all independent agents in parallel
builder.add_edge(START, "security_agent")
builder.add_edge(START, "cost_agent")
builder.add_edge(START, "init_trace")

# Performance Pipeline Sequence
builder.add_edge("init_trace", "extract_claims")
builder.add_edge("extract_claims", "retrieve_evidence")
builder.add_edge("retrieve_evidence", "evaluate_factuality")
builder.add_edge("retrieve_evidence", "evaluate_relevance")

# We use a fan-in approach: critic waits for BOTH factuality and relevance
builder.add_edge(["evaluate_factuality", "evaluate_relevance"], "independent_critic")

builder.add_edge("independent_critic", "policy_engine")
builder.add_edge("policy_engine", "decision_layer")

# Other agents go to final decision layer
builder.add_edge("security_agent", "decision_layer")
builder.add_edge("cost_agent", "decision_layer")

builder.add_edge("decision_layer", END)

# Compile Workflow
controlplane_graph = builder.compile()