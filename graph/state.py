from typing import TypedDict, List, Dict, Any

class ControlPlaneState(TypedDict):
    # Input & LLM Metadata
    user_id: str
    user_prompt: str
    llm_response: str
    model_name: str
    
    # Pre-Flight Gateway Signals
    preflight_risk_score: float
    
    # Performance Agent Outputs
    performance_score: float
    performance_status: str
    factual_findings: List[Dict[str, Any]]
    relevance_findings: List[Dict[str, Any]]
    
    # Security Agent Outputs
    security_score: float
    security_status: str
    security_findings: List[Dict[str, Any]]
    
    # Cost Agent Outputs
    cost_score: float
    cost_status: str
    input_tokens: int
    output_tokens: int
    estimated_cost: float
    ttft_latency_ms: float
    
    # Summarizing Layer Decisions
    unified_risk_score: float
    final_action: str  # "ALLOW", "REWRITE", "BLOCK", or "ESCALATE"
    audit_log: Dict[str, Any]