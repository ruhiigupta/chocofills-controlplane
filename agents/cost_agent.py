"""
ControlPlane.ai - Governance Cost & Latency Checker Agent
Author: Ishaan (Cost/Flow Agent - feature/cost-agent)
Branch: feature/cost-agent

Deterministic Cost Model, Tokenizer, SLA Latency Evaluator, and Anomaly Detector.
Compatible with ControlPlaneState and LangGraph Orchestrator.
"""

import re
import math
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


# =====================================================================
# 1. PRICING CATALOG (Deterministic USD Rates)
# =====================================================================

@dataclass(frozen=True)
class ModelPricing:
    model_name: str
    provider: str
    input_cost_per_million: float        # USD per 1M input tokens
    output_cost_per_million: float       # USD per 1M output tokens
    cached_input_cost_per_million: Optional[float] = None
    context_window_threshold: int = 128_000   # Threshold where tiered pricing applies
    tiered_input_cost_per_million: Optional[float] = None   # Pricing above threshold (>128k)
    tiered_output_cost_per_million: Optional[float] = None  # Pricing above threshold (>128k)


@dataclass(frozen=True)
class ToolPricing:
    tool_type: str
    cost_per_unit: float  # USD per unit (page, query, call)
    unit_name: str


MODEL_CATALOG: Dict[str, ModelPricing] = {
    # Google Gemini Models
    "gemini-1.5-pro": ModelPricing(
        model_name="gemini-1.5-pro",
        provider="google",
        input_cost_per_million=3.50,
        output_cost_per_million=10.50,
        cached_input_cost_per_million=0.875,
        context_window_threshold=128_000,
        tiered_input_cost_per_million=7.00,
        tiered_output_cost_per_million=21.00
    ),
    "gemini-1.5-flash": ModelPricing(
        model_name="gemini-1.5-flash",
        provider="google",
        input_cost_per_million=0.075,
        output_cost_per_million=0.30,
        cached_input_cost_per_million=0.01875,
        context_window_threshold=128_000,
        tiered_input_cost_per_million=0.15,
        tiered_output_cost_per_million=0.60
    ),
    "gemini-1.5-flash-8b": ModelPricing(
        model_name="gemini-1.5-flash-8b",
        provider="google",
        input_cost_per_million=0.0375,
        output_cost_per_million=0.15,
        cached_input_cost_per_million=0.01
    ),
    "gemini-2.0-flash": ModelPricing(
        model_name="gemini-2.0-flash",
        provider="google",
        input_cost_per_million=0.10,
        output_cost_per_million=0.40,
        cached_input_cost_per_million=0.025
    ),
    # OpenAI Models
    "gpt-4o": ModelPricing(
        model_name="gpt-4o",
        provider="openai",
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
        cached_input_cost_per_million=1.25
    ),
    "gpt-4o-mini": ModelPricing(
        model_name="gpt-4o-mini",
        provider="openai",
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        cached_input_cost_per_million=0.075
    ),
    # Anthropic Models
    "claude-3-5-sonnet": ModelPricing(
        model_name="claude-3-5-sonnet",
        provider="anthropic",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        cached_input_cost_per_million=0.30
    ),
    "claude-3-5-haiku": ModelPricing(
        model_name="claude-3-5-haiku",
        provider="anthropic",
        input_cost_per_million=0.80,
        output_cost_per_million=4.00,
        cached_input_cost_per_million=0.08
    ),
    # Open / Self-hosted baseline
    "llama-3.1-70b": ModelPricing(
        model_name="llama-3.1-70b",
        provider="meta",
        input_cost_per_million=0.60,
        output_cost_per_million=0.80
    ),
    "llama-3.1-8b": ModelPricing(
        model_name="llama-3.1-8b",
        provider="meta",
        input_cost_per_million=0.10,
        output_cost_per_million=0.10
    )
}

TOOL_CATALOG: Dict[str, ToolPricing] = {
    "ocr_pdf_page": ToolPricing(tool_type="ocr_pdf_page", cost_per_unit=0.0015, unit_name="page"),
    "ocr_image": ToolPricing(tool_type="ocr_image", cost_per_unit=0.0010, unit_name="image"),
    "bi_encoder_embedding": ToolPricing(tool_type="bi_encoder_embedding", cost_per_unit=0.02 / 1_000_000, unit_name="token"),
    "web_search_api": ToolPricing(tool_type="web_search_api", cost_per_unit=0.005, unit_name="query"),
    "vector_retrieval_query": ToolPricing(tool_type="vector_retrieval_query", cost_per_unit=0.0005, unit_name="query"),
}


def get_model_pricing(model_name: str) -> ModelPricing:
    clean_name = model_name.lower().strip()
    if clean_name in MODEL_CATALOG:
        return MODEL_CATALOG[clean_name]
    for key, pricing in MODEL_CATALOG.items():
        if key in clean_name or clean_name in key:
            return pricing
    return ModelPricing(
        model_name=model_name,
        provider="generic",
        input_cost_per_million=1.50,
        output_cost_per_million=6.00
    )


# =====================================================================
# 2. TOKENIZER SERVICE (TikTokenizer + BPE Fallback)
# =====================================================================

class TokenizerService:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoder = None

    def count_tokens(self, text: Optional[str]) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
        token_count = 0
        for token in tokens:
            if token.isspace():
                token_count += max(1, len(token) // 4)
            elif len(token) > 4:
                token_count += max(1, int(round(len(token) / 3.8)))
            else:
                token_count += 1
        return max(1, token_count) if text.strip() else 0

    def count_documents_tokens(self, source_documents: Optional[List[Dict[str, Any]]]) -> int:
        if not source_documents:
            return 0
        total = 0
        for doc in source_documents:
            total += self.count_tokens(doc.get("content", ""))
        return total


default_tokenizer = TokenizerService()


# =====================================================================
# 3. USAGE ANALYZER
# =====================================================================

@dataclass
class UsageMetrics:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    prompt_tokens: int
    document_tokens: int
    document_count: int
    expansion_ratio: float
    document_overhead_ratio: float
    output_chars: int
    prompt_chars: int
    requests_per_minute: float
    rolling_tokens_per_minute: int


class UsageAnalyzer:
    def __init__(self, tokenizer: Optional[TokenizerService] = None, window_seconds: int = 60):
        self.tokenizer = tokenizer or default_tokenizer
        self.window_seconds = window_seconds
        self._request_history: deque = deque()

    def _update_and_get_volume_rates(self, current_tokens: int) -> tuple[float, int]:
        now = time.time()
        self._request_history.append((now, current_tokens))
        cutoff = now - self.window_seconds
        while self._request_history and self._request_history[0][0] < cutoff:
            self._request_history.popleft()
        req_count = len(self._request_history)
        tot_tokens = sum(item[1] for item in self._request_history)
        scale = 60.0 / max(1.0, self.window_seconds)
        return req_count * scale, int(tot_tokens * scale)

    def analyze(self, state: Dict[str, Any]) -> UsageMetrics:
        user_prompt = state.get("user_prompt", "") or ""
        llm_response = state.get("llm_response", "") or ""
        source_docs = state.get("source_documents", []) or []
        
        prompt_tokens = self.tokenizer.count_tokens(user_prompt)
        doc_tokens = self.tokenizer.count_documents_tokens(source_docs)
        calculated_input_tokens = prompt_tokens + doc_tokens
        calculated_output_tokens = self.tokenizer.count_tokens(llm_response)
        
        input_tokens = state.get("input_tokens")
        if input_tokens is None or input_tokens <= 0:
            input_tokens = calculated_input_tokens
            
        output_tokens = state.get("output_tokens")
        if output_tokens is None or output_tokens < 0:
            output_tokens = calculated_output_tokens
            
        total_tokens = input_tokens + output_tokens
        expansion_ratio = round(output_tokens / max(1, input_tokens), 4)
        doc_overhead_ratio = round(doc_tokens / max(1, input_tokens), 4)
        rpm, tpm = self._update_and_get_volume_rates(total_tokens)
        
        return UsageMetrics(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            document_tokens=doc_tokens,
            document_count=len(source_docs),
            expansion_ratio=expansion_ratio,
            document_overhead_ratio=doc_overhead_ratio,
            output_chars=len(llm_response),
            prompt_chars=len(user_prompt),
            requests_per_minute=rpm,
            rolling_tokens_per_minute=tpm
        )


# =====================================================================
# 4. COST ANALYZER
# =====================================================================

@dataclass
class CostBreakdown:
    model_name: str
    provider: str
    input_rate_per_m: float
    output_rate_per_m: float
    input_cost_usd: float
    output_cost_usd: float
    model_cost_usd: float
    tools_cost_usd: float
    total_cost_usd: float
    tool_cost_details: Dict[str, float]
    cost_per_1k_output_tokens: float
    cost_efficiency_score: float


class CostAnalyzer:
    def _calculate_tool_costs(self, state: Dict[str, Any], doc_tokens: int) -> tuple[float, Dict[str, float]]:
        details: Dict[str, float] = {}
        total_tool_cost = 0.0
        
        source_docs = state.get("source_documents", []) or []
        for doc in source_docs:
            fname = str(doc.get("filename", "")).lower()
            if fname.endswith(".pdf"):
                page_cost = TOOL_CATALOG["ocr_pdf_page"].cost_per_unit
                details["ocr_pdf"] = details.get("ocr_pdf", 0.0) + page_cost
                total_tool_cost += page_cost
            elif fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                img_cost = TOOL_CATALOG["ocr_image"].cost_per_unit
                details["ocr_image"] = details.get("ocr_image", 0.0) + img_cost
                total_tool_cost += img_cost

        if doc_tokens > 0:
            embed_rate = TOOL_CATALOG["bi_encoder_embedding"].cost_per_unit
            embed_cost = doc_tokens * embed_rate
            details["bi_encoder_embedding"] = details.get("bi_encoder_embedding", 0.0) + embed_cost
            total_tool_cost += embed_cost

        tool_calls = state.get("tool_calls", []) or []
        for tc in tool_calls:
            tname = tc if isinstance(tc, str) else tc.get("name", "")
            tname_clean = tname.lower()
            if tname_clean in TOOL_CATALOG:
                c = TOOL_CATALOG[tname_clean].cost_per_unit
                details[tname_clean] = details.get(tname_clean, 0.0) + c
                total_tool_cost += c
            else:
                default_tool_cost = 0.001
                details[tname_clean] = details.get(tname_clean, 0.0) + default_tool_cost
                total_tool_cost += default_tool_cost

        return total_tool_cost, details

    def analyze(self, state: Dict[str, Any], usage: UsageMetrics) -> CostBreakdown:
        model_name = state.get("model_name", "gemini-1.5-pro") or "gemini-1.5-pro"
        pricing: ModelPricing = get_model_pricing(model_name)
        
        input_tokens = usage.input_tokens
        if pricing.tiered_input_cost_per_million and input_tokens > pricing.context_window_threshold:
            base_tokens = pricing.context_window_threshold
            tier_tokens = input_tokens - base_tokens
            input_cost = (base_tokens / 1_000_000.0) * pricing.input_cost_per_million + \
                         (tier_tokens / 1_000_000.0) * pricing.tiered_input_cost_per_million
            applied_in_rate = pricing.tiered_input_cost_per_million
        else:
            input_cost = (input_tokens / 1_000_000.0) * pricing.input_cost_per_million
            applied_in_rate = pricing.input_cost_per_million

        output_tokens = usage.output_tokens
        if pricing.tiered_output_cost_per_million and input_tokens > pricing.context_window_threshold:
            output_cost = (output_tokens / 1_000_000.0) * pricing.tiered_output_cost_per_million
            applied_out_rate = pricing.tiered_output_cost_per_million
        else:
            output_cost = (output_tokens / 1_000_000.0) * pricing.output_cost_per_million
            applied_out_rate = pricing.output_cost_per_million

        model_cost = input_cost + output_cost
        tools_cost, tool_details = self._calculate_tool_costs(state, usage.document_tokens)
        total_cost = model_cost + tools_cost
        cost_per_1k_out = (total_cost / max(1, output_tokens)) * 1000.0 if output_tokens > 0 else 0.0
        cost_efficiency_score = round(100.0 * math.exp(-total_cost / 0.025), 2)

        return CostBreakdown(
            model_name=pricing.model_name,
            provider=pricing.provider,
            input_rate_per_m=applied_in_rate,
            output_rate_per_m=applied_out_rate,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            model_cost_usd=round(model_cost, 6),
            tools_cost_usd=round(tools_cost, 6),
            total_cost_usd=round(total_cost, 6),
            tool_cost_details=tool_details,
            cost_per_1k_output_tokens=round(cost_per_1k_out, 6),
            cost_efficiency_score=cost_efficiency_score
        )


# =====================================================================
# 5. BUDGET & ANOMALY ANALYZER
# =====================================================================

@dataclass(frozen=True)
class BudgetConfig:
    target_cost_usd: float = 0.015       # Soft target (1.5 cents)
    max_cost_usd: float = 0.060          # Hard ceiling (6.0 cents)
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
        return round(1.0 - (len(unique_ngrams) / float(len(ngrams))), 4)

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


# =====================================================================
# 6. LATENCY ANALYZER (Whimsical Branch)
# =====================================================================

@dataclass(frozen=True)
class LatencySLAConfig:
    ttft_target_ms: float = 600.0
    ttft_max_ms: float = 2500.0
    total_latency_target_ms: float = 2000.0
    total_latency_max_ms: float = 12000.0


@dataclass
class LatencyEvaluationResult:
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]
    tool_latency_ms: Optional[float]
    ttft_score: float
    total_latency_score: float
    latency_score: float


class LatencyAnalyzer:
    def __init__(self, config: Optional[LatencySLAConfig] = None):
        self.config = config or LatencySLAConfig()

    def _score_metric(self, val_ms: Optional[float], target: float, max_val: float) -> float:
        if val_ms is None or val_ms <= 0:
            return 100.0
        if val_ms <= target:
            return 100.0
        if val_ms >= max_val:
            return 0.0
        decay = (val_ms - target) / (max_val - target)
        return round(100.0 * (1.0 - decay), 2)

    def analyze(self, state: Dict[str, Any]) -> LatencyEvaluationResult:
        ttft = state.get("ttft_latency_ms") or state.get("ttft_ms")
        total_lat = state.get("total_latency_ms")
        tool_lat = state.get("tool_latency_ms")
        
        req_start = state.get("request_start")
        req_end = state.get("request_end")
        if total_lat is None and req_start is not None and req_end is not None:
            try:
                total_lat = (float(req_end) - float(req_start)) * 1000.0
            except Exception:
                pass

        ttft_s = self._score_metric(ttft, self.config.ttft_target_ms, self.config.ttft_max_ms)
        tot_s = self._score_metric(total_lat, self.config.total_latency_target_ms, self.config.total_latency_max_ms)
        composite = round(0.40 * ttft_s + 0.60 * tot_s, 2)

        return LatencyEvaluationResult(
            ttft_ms=ttft,
            total_latency_ms=total_lat,
            tool_latency_ms=tool_lat,
            ttft_score=ttft_s,
            total_latency_score=tot_s,
            latency_score=composite
        )


# =====================================================================
# 7. MAIN COST EVALUATION ENGINE & LANGGRAPH NODE
# =====================================================================

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
