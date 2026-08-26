"""
ControlPlane.ai - Budget Analysis Module
Deterministic evaluation of cost thresholds, budget compliance scores, and cost anomalies.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import re

try:
    from .usage_analyzer import UsageMetrics
    from .cost_analyzer import CostBreakdown
except ImportError:
    from usage_analyzer import UsageMetrics
    from cost_analyzer import CostBreakdown


@dataclass(frozen=True)
class BudgetConfig:
    target_cost_usd: float = 0.015
    max_cost_usd: float = 0.060
    max_output_tokens: int = 4096
    repetition_anomaly_threshold: float = 0.35


@dataclass
class AnomalyFinding:
    anomaly_type: str
    severity: str
    description: str
    impact_usd: float


@dataclass
class BudgetEvaluationResult:
    target_cost_usd: float
    max_cost_usd: float
    cost_vs_target_ratio: float
    cost_vs_max_ratio: float
    budget_compliance_score: float
    is_budget_exceeded: bool
    anomalies: List[AnomalyFinding]
    has_critical_anomaly: bool


class BudgetAnalyzer:
    def __init__(self, config: Optional[BudgetConfig] = None):
        self.config = config or BudgetConfig()

    def _detect_repetition_loop(self, text: str) -> float:
        words = [w.lower() for w in re.findall(r"\b\w+\b", text)]
        if len(words) < 20:
            return 0.0
        
        n = 4
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            return 0.0
        
        unique_ngrams = set(ngrams)
        repetition_ratio = 1.0 - (len(unique_ngrams) / float(len(ngrams)))
        return round(repetition_ratio, 4)

    def analyze(self, state: Dict[str, Any], usage: UsageMetrics, cost: CostBreakdown) -> BudgetEvaluationResult:
        total_cost = cost.total_cost_usd
        target = self.config.target_cost_usd
        max_limit = self.config.max_cost_usd
        
        cost_vs_target = round(total_cost / max(1e-6, target), 3)
        cost_vs_max = round(total_cost / max(1e-6, max_limit), 3)
        
        if total_cost <= target:
            compliance_score = 100.0
        elif total_cost >= max_limit:
            compliance_score = 0.0
        else:
            decay_ratio = (total_cost - target) / (max_limit - target)
            compliance_score = round(100.0 * (1.0 - decay_ratio), 2)

        is_exceeded = total_cost > max_limit
        anomalies: List[AnomalyFinding] = []
        
        if is_exceeded:
            anomalies.append(AnomalyFinding(
                anomaly_type="BUDGET_CEILING_BREACH",
                severity="CRITICAL",
                description=f"Total cost (${total_cost:.4f}) exceeded hard budget limit (${max_limit:.4f}) by {(cost_vs_max - 1.0) * 100:.1f}%.",
                impact_usd=round(total_cost - max_limit, 6)
            ))
            
        llm_response = state.get("llm_response", "") or ""
        repetition_score = self._detect_repetition_loop(llm_response)
        if repetition_score > self.config.repetition_anomaly_threshold and usage.output_tokens > 250:
            anomalies.append(AnomalyFinding(
                anomaly_type="RUNAWAY_GENERATION_LOOP",
                severity="CRITICAL",
                description=f"Detected recursive looping generation (repetition index {repetition_score:.2f} > {self.config.repetition_anomaly_threshold}). Wasted tokens.",
                impact_usd=round(cost.output_cost_usd * 0.8, 6)
            ))

        if usage.input_tokens > 20_000 and usage.output_tokens < 15:
            anomalies.append(AnomalyFinding(
                anomaly_type="CONTEXT_STUFFING_LOW_YIELD",
                severity="MEDIUM",
                description=f"Massive input context ({usage.input_tokens} tokens) yielded only {usage.output_tokens} output tokens.",
                impact_usd=round(cost.input_cost_usd * 0.5, 6)
            ))

        if cost.input_rate_per_m >= 2.5 and usage.total_tokens < 60 and usage.document_count == 0:
            anomalies.append(AnomalyFinding(
                anomaly_type="MODEL_ROUTING_OVERKILL",
                severity="LOW",
                description=f"High-cost model '{cost.model_name}' was used for a trivial {usage.total_tokens}-token query. Routing to a flash model could save ~95% cost.",
                impact_usd=round(cost.total_cost_usd * 0.95, 6)
            ))

        has_critical = any(a.severity == "CRITICAL" for a in anomalies)

        return BudgetEvaluationResult(
            target_cost_usd=target,
            max_cost_usd=max_limit,
            cost_vs_target_ratio=cost_vs_target,
            cost_vs_max_ratio=cost_vs_max,
            budget_compliance_score=compliance_score,
            is_budget_exceeded=is_exceeded,
            anomalies=anomalies,
            has_critical_anomaly=has_critical
        )
