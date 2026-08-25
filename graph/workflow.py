from langgraph.graph import StateGraph, END
from graph.state import ControlPlaneState
from agents.security_agent import SecurityAgent

# Initialize the real Security Agent
security_agent_instance = SecurityAgent()

def security_agent_node(state: ControlPlaneState) -> ControlPlaneState:
    """Executes real-time security scanning on the LLM's response."""
    llm_response = state.get("llm_response", "")
    
    # Run the real evaluation using the regex patterns
    results = security_agent_instance.scan_output(llm_response)
    
    # Update the LangGraph State
    state["security_score"] = results["security_score"]
    state["security_status"] = results["security_status"]
    state["security_findings"] = results["security_findings"]
    
    print(f"[LangGraph Node - Security] Score: {results['security_score']} | Status: {results['security_status']}")
    return state

def performance_agent_node(state: ControlPlaneState) -> ControlPlaneState:
    # Dummy placeholder for Sanjana
    state["performance_score"] = 15.0
    state["performance_status"] = "PASS"
    state["factual_findings"] = []
    state["relevance_findings"] = []
    return state

def cost_agent_node(state: ControlPlaneState) -> ControlPlaneState:
    # Dummy placeholder for Cost Agent
    state["cost_score"] = 5.0
    state["cost_status"] = "PASS"
    state["estimated_cost"] = 0.002
    return state

def decision_layer_node(state: ControlPlaneState) -> ControlPlaneState:
    # Weighted Unified Risk Score Calculation
    sec = state.get("security_score", 0)
    perf = state.get("performance_score", 0)
    cost = state.get("cost_score", 0)
    
    # Weighting: Security 40%, Performance 40%, Cost 20%
    unified_risk = (0.4 * sec) + (0.4 * perf) + (0.2 * cost)
    state["unified_risk_score"] = unified_risk
    
    # Policy Decision Engine
    if sec >= 80 or unified_risk >= 70:
        state["final_action"] = "BLOCK"
    elif unified_risk >= 40:
        state["final_action"] = "REWRITE"
    else:
        state["final_action"] = "ALLOW"
        
    print(f"[LangGraph Node - Decision] Final Action: {state['final_action']} | Unified Risk: {unified_risk}")
    return state

# Construct the LangGraph Workflow
builder = StateGraph(ControlPlaneState)

# Add Agent Nodes
builder.add_node("security_agent", security_agent_node)
builder.add_node("performance_agent", performance_agent_node)
builder.add_node("cost_agent", cost_agent_node)
builder.add_node("decision_layer", decision_layer_node)

# Define Parallel Execution Flow
builder.set_entry_point("security_agent")

builder.add_edge("security_agent", "decision_layer")
builder.add_edge("performance_agent", "decision_layer")
builder.add_edge("cost_agent", "decision_layer")
builder.add_edge("decision_layer", END)

# Compile Workflow
controlplane_graph = builder.compile()