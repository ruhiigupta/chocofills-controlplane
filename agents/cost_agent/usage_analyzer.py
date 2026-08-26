"""
ControlPlane.ai - Usage Analysis Module
Deterministic evaluation of token consumption, context overhead, expansion ratios, and request volume.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

try:
    from .tokenizer_service import default_tokenizer, TokenizerService
except ImportError:
    from tokenizer_service import default_tokenizer, TokenizerService


@dataclass
class UsageMetrics:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    prompt_tokens: int
    document_tokens: int
    document_count: int
    expansion_ratio: float           # output_tokens / input_tokens
    document_overhead_ratio: float   # document_tokens / input_tokens
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
        rpm = req_count * scale
        tpm = int(tot_tokens * scale)
        return rpm, tpm

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
