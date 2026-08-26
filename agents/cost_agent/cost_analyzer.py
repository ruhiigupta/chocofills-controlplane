"""
ControlPlane.ai - Cost Analysis Module
Deterministic computation of model token pricing, tool/retrieval execution costs, and cost efficiency.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import math

try:
    from .pricing_catalog import get_model_pricing, TOOL_CATALOG, ModelPricing
    from .usage_analyzer import UsageMetrics
except ImportError:
    from pricing_catalog import get_model_pricing, TOOL_CATALOG, ModelPricing
    from usage_analyzer import UsageMetrics


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
    def __init__(self):
        pass

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
