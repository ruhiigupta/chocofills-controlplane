"""
ControlPlane.ai - Cost Evaluation Engine & LangGraph Agent Node
Author: Ishaan (Cost/Flow Agent - feature/cost-agent)
Deterministic Cost Checker node for ControlPlane.ai orchestrator.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

try:
    from .usage_analyzer import UsageAnalyzer, UsageMetrics
    from .cost_analyzer import CostAnalyzer, CostBreakdown
    from .budget_analyzer import BudgetAnalyzer, BudgetEvaluationResult, BudgetConfig
    from .latency_analyzer import LatencyAnalyzer, LatencyEvaluationResult, LatencySLAConfig
except ImportError:
    from usage_analyzer import UsageAnalyzer, UsageMetrics
    from cost_analyzer import CostAnalyzer, CostBreakdown
    from budget_analyzer import BudgetAnalyzer, BudgetEvaluationResult, BudgetConfig
    from latency_analyzer import LatencyAnalyzer, LatencyEvaluationResult, LatencySLAConfig


@dataclass
class CostEvaluationOutput:
    cost_score: float
    cost_score_normalized: float
    status: str
    verdict: str
    total_cost_usd: float
    model_cost_usd: float
    tools_cost_usd: float
    usage: UsageMetrics
    cost_breakdown: CostBreakdown
    budget_eval: BudgetEvaluationResult
    latency_eval: LatencyEvaluationResult


class CostEvaluationEngine:
    def __init__(
        self,
        budget_config: Optional[BudgetConfig] = None,
        latency_config: Optional[LatencySLAConfig] = None
    ):
        self.usage_analyzer = UsageAnalyzer()
        self.cost_analyzer = CostAnalyzer()
        self.budget_analyzer = BudgetAnalyzer(budget_config)
        self.latency_analyzer = LatencyAnalyzer(latency_config)

    def evaluate(self, state: Dict[str, Any]) -> CostEvaluationOutput:
        usage: UsageMetrics = self.usage_analyzer.analyze(state)
        cost: CostBreakdown = self.cost_analyzer.analyze(state, usage)
        budget: BudgetEvaluationResult = self.budget_analyzer.analyze(state, usage, cost)
        latency: LatencyEvaluationResult = self.latency_analyzer.analyze(state)

        raw_score = 0.65 * budget.budget_compliance_score + 0.35 * cost.cost_efficiency_score

        anomaly_penalty = 0.0
        for anomaly in budget.anomalies:
            if anomaly.severity == "CRITICAL":
                anomaly_penalty += 45.0
            elif anomaly.severity == "HIGH":
                anomaly_penalty += 25.0
            elif anomaly.severity == "MEDIUM":
                anomaly_penalty += 15.0
            elif anomaly.severity == "LOW":
                anomaly_penalty += 5.0

        final_cost_score = max(0.0, min(100.0, raw_score - anomaly_penalty))
        final_cost_score = round(final_cost_score, 2)
        score_norm = round(final_cost_score / 100.0, 4)

        if budget.is_budget_exceeded or budget.has_critical_anomaly or final_cost_score < 40.0:
            status = "CRITICAL"
        elif final_cost_score < 70.0 or len(budget.anomalies) > 0:
            status = "FLAG"
        else:
            status = "PASS"

        if status == "PASS":
            verdict = f"Passed cost checks: ${cost.total_cost_usd:.5f} total cost ({usage.total_tokens} tokens) within budget."
        elif status == "FLAG":
            reasons = [a.anomaly_type for a in budget.anomalies] or ["Moderate efficiency"]
            verdict = f"Flagged for cost review: ${cost.total_cost_usd:.5f} total cost. Notices: {', '.join(reasons)}."
        else:
            reasons = [a.anomaly_type for a in budget.anomalies] or ["Budget ceiling breached"]
            verdict = f"Critical cost alert: ${cost.total_cost_usd:.5f} total cost. Violations: {', '.join(reasons)}."

        return CostEvaluationOutput(
            cost_score=final_cost_score,
            cost_score_normalized=score_norm,
            status=status,
            verdict=verdict,
            total_cost_usd=cost.total_cost_usd,
            model_cost_usd=cost.model_cost_usd,
            tools_cost_usd=cost.tools_cost_usd,
            usage=usage,
            cost_breakdown=cost,
            budget_eval=budget,
            latency_eval=latency
        )


default_cost_engine = CostEvaluationEngine()


def cost_checker_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph Node interface for ControlPlaneState.
    Returns all top-level keys required by ControlPlaneState in graph/state.py.
    """
    evaluation = default_cost_engine.evaluate(state)
    
    return {
        **state,
        "cost_score": evaluation.cost_score,
        "cost_status": evaluation.status,
        "estimated_cost": evaluation.total_cost_usd,
        "input_tokens": evaluation.usage.input_tokens,
        "output_tokens": evaluation.usage.output_tokens,
        "ttft_latency_ms": state.get("ttft_latency_ms", 0.0) or (evaluation.latency_eval.ttft_ms or 0.0),
        "cost_agent": {
            "score": evaluation.cost_score,
            "score_normalized": evaluation.cost_score_normalized,
            "status": evaluation.status,
            "verdict": evaluation.verdict,
            "total_cost_usd": evaluation.total_cost_usd,
            "model_cost_usd": evaluation.model_cost_usd,
            "tools_cost_usd": evaluation.tools_cost_usd,
            "expansion_ratio": evaluation.usage.expansion_ratio,
            "budget_compliance_score": evaluation.budget_eval.budget_compliance_score,
            "anomalies": [asdict(a) for a in evaluation.budget_eval.anomalies],
            "latency_score": evaluation.latency_eval.latency_score
        }
    }
