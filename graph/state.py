from typing import TypedDict, List, Dict, Any, Optional
from schemas.core import EvaluationTrace


class ControlPlaneState(TypedDict):
    # Input & LLM Metadata
    user_id: str
    use_case: str
    source: str
    destination: str
    user_prompt: str
    system_prompt: Optional[str]
    source_documents: List[Dict[str, str]]
    llm_response: str
    model_name: str

    # Pre-Flight Gateway Signals
    preflight_risk_score: float

    # Performance Agent Outputs
    performance_score: float
    performance_status: str
    factual_findings: List[Dict[str, Any]]
    relevance_findings: List[Dict[str, Any]]
    evaluation_trace: EvaluationTrace

    # Security Agent Outputs
    security_score: float
    security_status: str
    security_decision: str
    security_findings: List[Dict[str, Any]]
    matched_policies: List[Dict[str, Any]]
    policy_source: str

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