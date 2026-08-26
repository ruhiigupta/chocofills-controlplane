"""
ControlPlane.ai - Latency Evaluation Module
Deterministic evaluation of TTFT, Total Latency, and Tool/API latency against SLAs.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class LatencySLAConfig:
    ttft_target_ms: float = 600.0       # Target Time to First Token (0.6s)
    ttft_max_ms: float = 2500.0         # Max allowable TTFT (2.5s)
    total_latency_target_ms: float = 2000.0  # Target Total Latency (2.0s)
    total_latency_max_ms: float = 12000.0    # Max allowable Total Latency (12.0s)


@dataclass
class LatencyEvaluationResult:
    ttft_ms: Optional[float]
    total_latency_ms: Optional[float]
    tool_latency_ms: Optional[float]
    ttft_score: float
    total_latency_score: float
    latency_score: float  # Composite 0 - 100 scale


class LatencyAnalyzer:
    def __init__(self, config: Optional[LatencySLAConfig] = None):
        self.config = config or LatencySLAConfig()

    def _score_metric(self, val_ms: Optional[float], target: float, max_val: float) -> float:
        if val_ms is None or val_ms <= 0:
            return 100.0  # Default neutral/good if not measured
        if val_ms <= target:
            return 100.0
        if val_ms >= max_val:
            return 0.0
        decay = (val_ms - target) / (max_val - target)
        return round(100.0 * (1.0 - decay), 2)

    def analyze(self, state: Dict[str, Any]) -> LatencyEvaluationResult:
        """
        Deterministically evaluate latency metrics if present in ControlPlaneState.
        """
        ttft = state.get("ttft_ms")
        total_lat = state.get("total_latency_ms")
        tool_lat = state.get("tool_latency_ms")
        
        # Calculate from request_start / request_end timestamps if available
        req_start = state.get("request_start")
        req_end = state.get("request_end")
        if total_lat is None and req_start is not None and req_end is not None:
            try:
                total_lat = (float(req_end) - float(req_start)) * 1000.0
            except Exception:
                pass

        ttft_s = self._score_metric(ttft, self.config.ttft_target_ms, self.config.ttft_max_ms)
        tot_s = self._score_metric(total_lat, self.config.total_latency_target_ms, self.config.total_latency_max_ms)

        # Composite latency score (40% TTFT, 60% Total Latency)
        composite = round(0.40 * ttft_s + 0.60 * tot_s, 2)

        return LatencyEvaluationResult(
            ttft_ms=ttft,
            total_latency_ms=total_lat,
            tool_latency_ms=tool_lat,
            ttft_score=ttft_s,
            total_latency_score=tot_s,
            latency_score=composite
        )
