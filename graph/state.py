from typing import TypedDict, List, Dict, Any, Optional, Callable
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
    target_llm: Optional[Callable[..., Any]]
    preflight_scanner: Optional[Callable[..., Any]]
    llm_failed: bool

    # Pre-Flight Gateway Signals
    preflight_risk_score: float
    preflight_blocked: bool
    preflight_reason: str
    preflight_findings: List[Dict[str, Any]]

    # Performance Agent Outputs
    performance_score: float
    performance_status: str
    factual_findings: List[Dict[str, Any]]
    relevance_findings: List[Dict[str, Any]]
    evaluation_trace: Optional[EvaluationTrace]

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
    cost_agent: Dict[str, Any]
    tool_calls: List[Dict[str, Any]]

    # Summarizing Layer Decisions
    unified_risk_score: float
    final_action: str  # "ALLOW", "REWRITE", "BLOCK", or "ESCALATE"
    audit_log: Dict[str, Any]
    final_response: str