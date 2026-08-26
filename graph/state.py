from typing import TypedDict, List, Dict, Any, Optional
from schemas.core import EvaluationTrace

class ControlPlaneState(TypedDict):
    # Inputs
    user_prompt: str
    llm_response: str
    
    # Security Outputs (Legacy/Mocked)
    security_score: float
    security_status: str
    security_findings: List[Dict[str, Any]]
    
    # Structured Performance Evaluation Trace
    evaluation_trace: EvaluationTrace
    
    # Cost Outputs (Legacy/Mocked)
    cost_score: float
    cost_status: str
    estimated_cost: float
    
    # Final Routing
    unified_risk_score: float
    final_action: str

