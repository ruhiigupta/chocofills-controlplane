from langgraph.graph import StateGraph, START, END
from graph.state import ControlPlaneState
from agents.security_agent import SecurityAgent

# Initialize the real Security Agent
security_agent_instance = SecurityAgent()

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

def performance_agent_node(state: ControlPlaneState) -> dict:
    # Dummy placeholder for Sanjana
    return {
        "performance_score": 15.0,
        "performance_status": "PASS",
        "factual_findings": [],
        "relevance_findings": []
    }

def cost_agent_node(state: ControlPlaneState) -> dict:
    # Dummy placeholder for Cost Agent
    return {
        "cost_score": 5.0,
        "cost_status": "PASS",
        "estimated_cost": 0.002
    }

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

# Construct the LangGraph Workflow
builder = StateGraph(ControlPlaneState)

# Add Agent Nodes
builder.add_node("security_agent", security_agent_node)
builder.add_node("performance_agent", performance_agent_node)
builder.add_node("cost_agent", cost_agent_node)
builder.add_node("decision_layer", decision_layer_node)

# Define Parallel Execution Flow
builder.add_edge(START, "security_agent")
builder.add_edge(START, "performance_agent")
builder.add_edge(START, "cost_agent")

builder.add_edge("security_agent", "decision_layer")
builder.add_edge("performance_agent", "decision_layer")
builder.add_edge("cost_agent", "decision_layer")
builder.add_edge("decision_layer", END)

# Compile Workflow
controlplane_graph = builder.compile()