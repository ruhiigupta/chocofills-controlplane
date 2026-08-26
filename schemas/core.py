from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- Base Data Models ---

class Claim(BaseModel):
    id: str = Field(description="Unique identifier for the claim")
    text: str = Field(description="The atomic, verifiable claim text")

class EvidenceChunk(BaseModel):
    id: str = Field(description="Unique identifier for the evidence chunk")
    text: str = Field(description="The actual text content of the retrieved chunk")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the source (e.g., URL, author, page number)")
    relevance_score: float = Field(description="Retrieval relevance score (e.g., cosine similarity from vector DB)")

# --- Evaluation Metric Models ---

class FactualityMetric(BaseModel):
    claim_id: str
    entailment_score: float = Field(ge=0.0, le=1.0, description="Score representing if the evidence supports the claim (1.0 = full support, 0.0 = contradiction)")
    evidence_coverage: float = Field(ge=0.0, le=1.0, description="Score representing how much of the claim is covered by the provided evidence")
    evidence_quality: float = Field(ge=0.0, le=1.0, description="Assessment of the reliability and clarity of the evidence used")
    confidence: float = Field(ge=0.0, le=1.0, description="The LLM's confidence in this evaluation")
    reasoning: str = Field(description="Detailed reasoning explaining the scores")
    evidence_used: List[str] = Field(description="List of EvidenceChunk IDs used to evaluate this claim")

class RelevanceMetric(BaseModel):
    claim_id: str
    relevance_score: float = Field(ge=0.0, le=1.0, description="Score representing how relevant this claim is to answering the user prompt")
    reasoning: str = Field(description="Detailed reasoning for the relevance score")

class SystemMetrics(BaseModel):
    latency_ms: float = 0.0
    token_usage: int = 0
    model_used: str = ""

class CriticMetric(BaseModel):
    claim_id: str
    agrees_with_factuality: bool = Field(description="True if the critic agrees with the factuality agent's assessment")
    agrees_with_relevance: bool = Field(description="True if the critic agrees with the relevance agent's assessment")
    critic_factuality_score: float = Field(ge=0.0, le=1.0, description="The critic's own independent factuality/entailment score")
    critic_relevance_score: float = Field(ge=0.0, le=1.0, description="The critic's own independent relevance score")
    contradiction_flag: bool = Field(description="True if a severe contradiction or hallucination is detected")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the verification assessment")
    reasoning: str = Field(description="Detailed reasoning for agreement or disagreement")

class PolicyDecision(BaseModel):
    final_action: str = Field(description="One of: PASS, RETRY, BLOCK, NEEDS_REVIEW")
    reasoning: str = Field(description="Why this decision was made based on the configured thresholds")
    flagged_claims: List[str] = Field(default_factory=list, description="IDs of claims that triggered a non-PASS action")

# --- Aggregated Trace ---

class EvaluationTrace(BaseModel):
    """The complete structured trace of the evaluation pipeline."""
    user_prompt: str
    llm_response: str
    claims: List[Claim] = []
    retrieved_evidence: Dict[str, List[EvidenceChunk]] = {} # Map claim_id to EvidenceChunks
    factuality_metrics: List[FactualityMetric] = []
    relevance_metrics: List[RelevanceMetric] = []
    critic_metrics: List[CriticMetric] = []
    policy_decision: Optional[PolicyDecision] = None
    system_metrics: SystemMetrics = Field(default_factory=SystemMetrics)

