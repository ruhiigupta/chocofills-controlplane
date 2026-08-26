from typing import List
from schemas.core import EvaluationTrace, PolicyDecision, FactualityMetric, RelevanceMetric, CriticMetric

class PolicyEngine:
    """
    Configurable rules engine that translates evaluation metrics into routing decisions.
    Separating this from the evaluators allows non-engineers to tweak thresholds.
    """
    
    def __init__(self):
        # Configurable thresholds
        self.min_factuality = 0.7
        self.min_relevance = 0.5
        self.min_evidence_coverage = 0.5
        
    def evaluate_trace(self, trace: EvaluationTrace) -> PolicyDecision:
        flagged_claims = []
        action = "PASS"
        reasons = []
        
        # We need to map claims to their respective metrics for easier evaluation
        factuality_map = {m.claim_id: m for m in trace.factuality_metrics}
        relevance_map = {m.claim_id: m for m in trace.relevance_metrics}
        critic_map = {m.claim_id: m for m in trace.critic_metrics}
        
        for claim in trace.claims:
            claim_id = claim.id
            fact_metric = factuality_map.get(claim_id)
            rel_metric = relevance_map.get(claim_id)
            critic_metric = critic_map.get(claim_id)
            
            claim_flagged = False
            
            if not fact_metric or not rel_metric or not critic_metric:
                action = "NEEDS_REVIEW"
                reasons.append(f"Missing metrics for claim {claim_id}.")
                claim_flagged = True
                continue
                
            # 1. Severe Critic Contradiction (Overrules everything -> BLOCK)
            if critic_metric.contradiction_flag:
                action = "BLOCK"
                reasons.append(f"Critic detected severe contradiction in claim {claim_id}.")
                claim_flagged = True
                
            # 2. Factuality Check
            elif fact_metric.entailment_score < self.min_factuality:
                # If it's a minor failure, we might RETRY
                if action != "BLOCK":
                    action = "RETRY"
                reasons.append(f"Claim {claim_id} failed factuality threshold ({fact_metric.entailment_score} < {self.min_factuality}).")
                claim_flagged = True
                
            # 3. Evidence Coverage Check (If low coverage, RETRY to fetch more docs)
            elif fact_metric.evidence_coverage < self.min_evidence_coverage:
                if action not in ["BLOCK", "RETRY"]:
                    action = "NEEDS_REVIEW"
                reasons.append(f"Claim {claim_id} lacks sufficient evidence coverage.")
                claim_flagged = True
                
            # 4. Relevance Check
            elif rel_metric.relevance_score < self.min_relevance:
                if action not in ["BLOCK"]:
                    action = "RETRY"
                reasons.append(f"Claim {claim_id} is irrelevant to the prompt.")
                claim_flagged = True
                
            # 5. Critic Disagreement Check
            elif not critic_metric.agrees_with_factuality or not critic_metric.agrees_with_relevance:
                if action not in ["BLOCK", "RETRY"]:
                    action = "NEEDS_REVIEW"
                reasons.append(f"Critic disagreed with evaluators for claim {claim_id}.")
                claim_flagged = True
                
            if claim_flagged:
                flagged_claims.append(claim_id)
                
        if action == "PASS":
            reasons.append("All claims passed required thresholds and critic verification.")
            
        return PolicyDecision(
            final_action=action,
            reasoning=" | ".join(reasons),
            flagged_claims=flagged_claims
        )

