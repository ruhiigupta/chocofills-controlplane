"""
ControlPlane.ai - Pricing Catalog
Deterministic pricing configurations for LLMs, Vector Embedding Models, OCR Tools, and Web Search APIs.
All pricing is standardized in USD per 1,000,000 (1M) tokens or per call.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelPricing:
    model_name: str
    provider: str
    input_cost_per_million: float        # USD per 1M input tokens
    output_cost_per_million: float       # USD per 1M output tokens
    cached_input_cost_per_million: Optional[float] = None
    context_window_threshold: int = 128_000   # Token threshold where tiered pricing applies
    tiered_input_cost_per_million: Optional[float] = None   # Pricing above threshold (>128k)
    tiered_output_cost_per_million: Optional[float] = None  # Pricing above threshold (>128k)


@dataclass(frozen=True)
class ToolPricing:
    tool_type: str
    cost_per_unit: float  # USD per unit (page, query, call)
    unit_name: str        # e.g., "page", "query", "token"


# Comprehensive Deterministic Pricing Catalog
MODEL_CATALOG: Dict[str, ModelPricing] = {
    # Google Gemini Models
    "gemini-1.5-pro": ModelPricing(
        model_name="gemini-1.5-pro",
        provider="google",
        input_cost_per_million=3.50,        # $3.50 / 1M (<128k tokens)
        output_cost_per_million=10.50,      # $10.50 / 1M (<128k tokens)
        cached_input_cost_per_million=0.875,
        context_window_threshold=128_000,
        tiered_input_cost_per_million=7.00, # $7.00 / 1M (>128k tokens)
        tiered_output_cost_per_million=21.00 # $21.00 / 1M (>128k tokens)
    ),
    "gemini-1.5-flash": ModelPricing(
        model_name="gemini-1.5-flash",
        provider="google",
        input_cost_per_million=0.075,       # $0.075 / 1M (<128k tokens)
        output_cost_per_million=0.30,       # $0.30 / 1M (<128k tokens)
        cached_input_cost_per_million=0.01875,
        context_window_threshold=128_000,
        tiered_input_cost_per_million=0.15, # $0.15 / 1M (>128k tokens)
        tiered_output_cost_per_million=0.60 # $0.60 / 1M (>128k tokens)
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

    # Meta Llama Models
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

# Tool & Retrieval API Execution Costs
TOOL_CATALOG: Dict[str, ToolPricing] = {
    "ocr_pdf_page": ToolPricing(tool_type="ocr_pdf_page", cost_per_unit=0.0015, unit_name="page"),
    "ocr_image": ToolPricing(tool_type="ocr_image", cost_per_unit=0.0010, unit_name="image"),
    "bi_encoder_embedding": ToolPricing(tool_type="bi_encoder_embedding", cost_per_unit=0.02 / 1_000_000, unit_name="token"),
    "web_search_api": ToolPricing(tool_type="web_search_api", cost_per_unit=0.005, unit_name="query"),
    "vector_retrieval_query": ToolPricing(tool_type="vector_retrieval_query", cost_per_unit=0.0005, unit_name="query"),
}


def get_model_pricing(model_name: str) -> ModelPricing:
    """Retrieve pricing for model with intelligent fallback matching."""
    clean_name = model_name.lower().strip()
    if clean_name in MODEL_CATALOG:
        return MODEL_CATALOG[clean_name]
    
    for key, pricing in MODEL_CATALOG.items():
        if key in clean_name or clean_name in key:
            return pricing
            
    # Default fallback rate
    return ModelPricing(
        model_name=model_name,
        provider="generic",
        input_cost_per_million=1.50,
        output_cost_per_million=6.00
    )
