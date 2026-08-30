import json
import os
import re
import time
import uuid
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph
from llm_guard.input_scanners import PromptInjection

from agents.cost_agent import cost_checker_node
from agents.performance_agent import PerformanceAgent
from agents.security_agent import SecurityAgent
from graph.state import ControlPlaneState
from services.audit_logger import AuditLogger


security_agent_instance = SecurityAgent()
performance_agent_instance = PerformanceAgent()
audit_logger = AuditLogger()


def _normalize_action(action: Any) -> str:
    return str(action or "").upper()


def _compute_evaluation_result(expected_action: Any, ground_truth: Any, observed_action: Any) -> str | None:
    expected = _normalize_action(expected_action)
    truth = _normalize_action(ground_truth)
    observed = _normalize_action(observed_action)

    if truth in {"ALLOW", "BLOCK"} and observed in {"ALLOW", "BLOCK"}:
        if truth == "BLOCK" and observed == "BLOCK":
            return "TP"
        if truth == "ALLOW" and observed == "ALLOW":
            return "TN"
        if truth == "ALLOW" and observed == "BLOCK":
            return "FP"
        if truth == "BLOCK" and observed == "ALLOW":
            return "FN"

    if expected and observed == expected:
        return "MATCH"
    if expected and truth and observed != truth:
        return "MISMATCH"
    if truth in {"ESCALATE", "REWRITE"} and observed == truth:
        return "MATCH"
    if truth in {"ESCALATE", "REWRITE"} and observed != truth:
        return "MISMATCH"
    return None


def _persist_audit_record(state: ControlPlaneState, *, forced: bool = False) -> dict[str, Any]:
    request_id = state.get("request_id") or str(uuid.uuid4())
    if state.get("audit_record_id") and not forced:
        return {"request_id": request_id, "audit_record_id": state.get("audit_record_id")}

    record = {
        "request_id": request_id,
        "audit_record_id": state.get("audit_record_id") or f"audit_{request_id}",
        "user_id": state.get("user_id"),
        "use_case": state.get("use_case", "internal_copilot"),
        "user_prompt": state.get("user_prompt", ""),
        "llm_response": state.get("llm_response", ""),
        "source": state.get("source", "internal_api"),
        "destination": state.get("destination", "external_vendor"),
        "trust_boundary_crossed": bool(state.get("trust_boundary_crossed", False)),
        "sensitivity": state.get("sensitivity") or "UNKNOWN",
        "categories": state.get("categories", []) or [],
        "security_score": float(state.get("security_score", 0.0)),
        "security_status": state.get("security_status", "PENDING"),
        "security_decision": state.get("security_decision", "UNKNOWN"),
        "security_findings": state.get("security_findings", []),
        "matched_policies": state.get("matched_policies", []),
        "policy_source": state.get("policy_source", "UNKNOWN"),
        "performance_score": float(state.get("performance_score", 0.0)),
        "performance_status": state.get("performance_status", "PENDING"),
        "cost_score": float(state.get("cost_score", 0.0)),
        "cost_status": state.get("cost_status", "PENDING"),
        "estimated_cost": float(state.get("estimated_cost", 0.0)),
        "unified_risk_score": float(state.get("unified_risk_score", 0.0)),
        "final_action": state.get("final_action", "PENDING"),
        "preflight_risk_score": float(state.get("preflight_risk_score", 0.0)),
        "expected_action": state.get("expected_action"),
        "ground_truth": state.get("ground_truth"),
        "evaluation_result": state.get("evaluation_result") or _compute_evaluation_result(
            state.get("expected_action"),
            state.get("ground_truth"),
            state.get("final_action"),
        ),
        "latency_ms": float(state.get("latency_ms", 0.0)),
    }

    persisted = audit_logger.log_request(record)
    return {
        "request_id": persisted.get("request_id", request_id),
        "audit_record_id": persisted.get("audit_record_id") or record["audit_record_id"],
    }


class CorrectedPromptInjection(PromptInjection):
    def __init__(self, threshold=0.5):
        super().__init__(threshold=threshold)

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        if prompt.strip() == "":
            return prompt, True, 0.0

        explicit_injection = re.search(
            r"(?:ignore|disregard|override|forget)\s+(?:all\s+)?"
            r"(?:previous|prior|earlier)\s+instructions|"
            r"reveal\s+(?:the\s+)?(?:hidden\s+)?system\s+prompt",
            prompt,
            re.IGNORECASE,
        )

        if explicit_injection:
            return prompt, False, 1.0

        try:
            _, is_valid = super().scan(prompt)
            return prompt, is_valid, 0.0 if is_valid else 1.0
        except Exception as e:
            print(f"[Prompt Injection Scanner] Error: {e}")
            return prompt, True, 0.0


injection_scanner = CorrectedPromptInjection(threshold=0.5)


def default_target_llm(prompt: str) -> tuple[str, bool]:
    return "Target LLM is not configured.", True


def default_preflight_scan(prompt: str) -> tuple[str, bool, float]:
    return injection_scanner.scan(prompt)


def preflight_node(state: ControlPlaneState) -> dict[str, Any]:
    request_id = state.get("request_id") or str(uuid.uuid4())
    scanner = state.get("preflight_scanner", default_preflight_scan)
    scan_result = scanner(state.get("user_prompt", ""))
    if len(scan_result) == 2:
        sanitized_prompt, is_valid = scan_result
        risk_score = 0.0
    else:
        sanitized_prompt, is_valid, risk_score = scan_result
    patterns = security_agent_instance.pattern_scanner(state.get("user_prompt", ""))
    sensitivity = security_agent_instance._deterministic_sensitivity(
        state.get("user_prompt", ""), patterns["matches"]
    )

    reason = ""
    blocked = not is_valid
    if blocked:
        reason = "Prompt Injection / Malicious Payload Detected"
    elif sensitivity == "Highly Restricted":
        blocked = True
        risk_score = 100.0
        reason = "Highly Restricted data detected before target LLM processing."
    elif sensitivity == "Confidential" and state.get("destination", "external_vendor") == "external_vendor":
        blocked = True
        risk_score = 100.0
        reason = "Confidential data cannot be sent to an external LLM."

    audit_payload = {
        "request_id": request_id,
        "user_id": state.get("user_id"),
        "use_case": state.get("use_case", "internal_copilot"),
        "user_prompt": sanitized_prompt,
        "llm_response": state.get("llm_response", ""),
        "source": state.get("source", "internal_api"),
        "destination": state.get("destination", "external_vendor"),
        "trust_boundary_crossed": bool(state.get("destination", "external_vendor") in {"external_vendor", "public_api"}),
        "sensitivity": sensitivity or "UNKNOWN",
        "categories": sorted(set(patterns.get("matches", []))),
        "security_score": 100.0 if blocked else 0.0,
        "security_status": "FAIL" if blocked else "PASS",
        "security_decision": "BLOCK" if blocked else "ALLOW",
        "security_findings": [
            {"type": "Preflight", "reason": reason, "confidence": 1.0, "context": {"pattern_matches": patterns.get("matches", []), "sensitivity": sensitivity or "UNKNOWN"}}
        ] if blocked else [],
        "matched_policies": [{"policy_id": "PREFLIGHT", "decision": "BLOCK", "reason": reason}] if blocked else [],
        "policy_source": "PREFLIGHT",
        "performance_score": float(state.get("performance_score", 0.0)),
        "performance_status": state.get("performance_status", "PENDING"),
        "cost_score": float(state.get("cost_score", 0.0)),
        "cost_status": state.get("cost_status", "PENDING"),
        "estimated_cost": float(state.get("estimated_cost", 0.0)),
        "unified_risk_score": 100.0 if blocked else float(state.get("unified_risk_score", 0.0)),
        "final_action": "BLOCK" if blocked else state.get("final_action", "PENDING"),
        "preflight_risk_score": risk_score,
        "expected_action": state.get("expected_action"),
        "ground_truth": state.get("ground_truth"),
        "evaluation_result": _compute_evaluation_result(state.get("expected_action"), state.get("ground_truth"), "BLOCK" if blocked else state.get("final_action")),
        "latency_ms": float(state.get("latency_ms", 0.0)),
    }
    if blocked:
        persisted = audit_logger.log_request(audit_payload)
        audit_payload["audit_record_id"] = persisted.get("audit_record_id") or f"audit_{request_id}"

    return {
        "request_id": request_id,
        "audit_record_id": audit_payload.get("audit_record_id"),
        "user_prompt": sanitized_prompt,
        "preflight_risk_score": risk_score,
        "preflight_blocked": blocked,
        "preflight_reason": reason,
        "preflight_findings": patterns["findings"],
        "final_action": "BLOCK" if blocked else state.get("final_action", "PENDING"),
        "audit_log": {
            "preflight_risk_score": risk_score,
            "action": "BLOCK" if blocked else "PENDING",
            "reason": reason,
        } if blocked else state.get("audit_log", {}),
    }


def route_after_preflight(state: ControlPlaneState) -> str:
    return "final_response" if state.get("preflight_blocked") else "target_llm"


def target_llm_node(state: ControlPlaneState) -> dict[str, Any]:

    if state.get("evaluation_mode") == "evaluate_existing":
        return {
            "llm_failed": False,
            "final_action": "PENDING"
        }
     
    if state.get("llm_response") and "target_llm" not in state:
        return {"llm_failed": False, "final_action": "PENDING"}

    target_llm: Callable[[str], tuple[str, bool]] = state.get("target_llm", default_target_llm)
    try:
        response = target_llm(state.get("user_prompt", ""))
        llm_response, llm_failed = response if isinstance(response, tuple) else (str(response), False)
    except Exception as error:
        llm_response, llm_failed = f"Target LLM failed: {error}", True

    return {
        "llm_response": llm_response,
        "llm_failed": llm_failed,
        "final_action": "BLOCK" if llm_failed else "PENDING",
    }


def route_after_target(state: ControlPlaneState) -> str:
    return "final_response" if state.get("llm_failed") else "agent_fanout"


def agent_fanout_node(state: ControlPlaneState) -> dict[str, Any]:
    return {}


def performance_agent_node(state: ControlPlaneState) -> dict[str, Any]:
    return performance_agent_instance.run_evaluation(
        state.get("user_prompt", ""), state.get("llm_response", ""), state.get("source_documents", [])
    )


def security_agent_node(state: ControlPlaneState) -> dict[str, Any]:
    result = security_agent_instance.scan_output(
        llm_response=state.get("llm_response", ""),
        user_prompt=state.get("user_prompt", ""),
        source=state.get("source", "internal_api"),
        dest=state.get("destination", "external_vendor"),
    )
    findings = result.get("security_findings", [])
    context = findings[0].get("context", {}) if findings else {}
    return {
        "sensitivity": context.get("sensitivity", "UNKNOWN"),
        "categories": sorted(set(context.get("categories", []) or [])),
        "security_score": result["security_score"],
        "security_status": result["security_status"],
        "security_decision": result["security_decision"],
        "security_findings": result["security_findings"],
        "matched_policies": result["matched_policies"],
        "policy_source": result["policy_source"],
    }


def cost_agent_node(state: ControlPlaneState) -> dict[str, Any]:
    result = cost_checker_node(state)
    cost = result["cost_agent"]
    return {
        "cost_agent": cost,
        "cost_score": cost["score"],
        "cost_status": cost["status"],
        "estimated_cost": cost["total_cost_usd"]
    }


def decision_layer_node(state: ControlPlaneState) -> dict[str, Any]:
    security_score = float(state.get("security_score", 0.0))
    performance_score = float(state.get("performance_score", 0.0))
    cost_score = float(state.get("cost_score", 0.0))
    unified_risk = round(
        0.4 * security_score + 0.4 * (100.0 - performance_score) + 0.2 * (100.0 - cost_score),
        2,
    )

    security_decision = state.get("security_decision", "UNKNOWN")
    if security_decision == "BLOCK":
        final_action = "BLOCK"
    elif security_decision == "REDACT":
        final_action = "REWRITE"
    elif security_decision in {"REQUIRE_APPROVAL", "FLAG"}:
        final_action = "ESCALATE"
    elif state.get("performance_status") != "PASS" or state.get("cost_status") == "CRITICAL":
        final_action = "ESCALATE"
    else:
        final_action = "ALLOW"

    audit_log = {
        "user_id": state.get("user_id"),
        "use_case": state.get("use_case", "internal_copilot"),
        "preflight_risk_score": state.get("preflight_risk_score", 0.0),
        "security": {"score": security_score, "status": state.get("security_status"), "decision": security_decision},
        "performance": {"score": performance_score, "status": state.get("performance_status")},
        "cost": {"score": cost_score, "status": state.get("cost_status")},
        "unified_risk_score": unified_risk,
        "action": final_action,
    }
    audit_file = os.path.join(os.path.dirname(__file__), "..", "data", "audit_log.jsonl")
    os.makedirs(os.path.dirname(audit_file), exist_ok=True)
    with open(audit_file, "a", encoding="utf-8") as file:
        file.write(json.dumps(audit_log) + "\n")

    persisted = _persist_audit_record({
        **state,
        "request_id": state.get("request_id") or str(uuid.uuid4()),
        "audit_record_id": state.get("audit_record_id"),
        "sensitivity": state.get("sensitivity") or "UNKNOWN",
        "categories": state.get("categories", []) or [],
        "security_score": security_score,
        "security_status": state.get("security_status", "PENDING"),
        "security_decision": security_decision,
        "security_findings": state.get("security_findings", []),
        "matched_policies": state.get("matched_policies", []),
        "policy_source": state.get("policy_source", "UNKNOWN"),
        "performance_score": performance_score,
        "performance_status": state.get("performance_status", "PENDING"),
        "cost_score": cost_score,
        "cost_status": state.get("cost_status", "PENDING"),
        "estimated_cost": float(state.get("estimated_cost", 0.0)),
        "unified_risk_score": unified_risk,
        "final_action": final_action,
        "preflight_risk_score": float(state.get("preflight_risk_score", 0.0)),
        "expected_action": state.get("expected_action"),
        "ground_truth": state.get("ground_truth"),
        "evaluation_result": state.get("evaluation_result") or _compute_evaluation_result(
            state.get("expected_action"), state.get("ground_truth"), final_action
        ),
        "latency_ms": float(state.get("latency_ms", 0.0)),
    })

    return {
        "request_id": persisted.get("request_id"),
        "audit_record_id": persisted.get("audit_record_id"),
        "unified_risk_score": unified_risk,
        "final_action": final_action,
        "audit_log": audit_log,
        "evaluation_result": state.get("evaluation_result") or _compute_evaluation_result(
            state.get("expected_action"), state.get("ground_truth"), final_action
        ),
    }


def final_response_node(state: ControlPlaneState) -> dict[str, Any]:
    action = state.get("final_action", "BLOCK")
    if action == "BLOCK":
        message = state.get("preflight_reason") or "Response blocked by ControlPlane policy."
    elif action == "REWRITE":
        message = "Response requires rewriting before it can be released."
    elif action == "ESCALATE":
        message = "Response requires human review."
    else:
        message = state.get("llm_response", "")
    return {"final_response": message}


builder = StateGraph(ControlPlaneState)
builder.add_node("preflight", preflight_node)
builder.add_node("target_llm", target_llm_node)
builder.add_node("agent_fanout", agent_fanout_node)
builder.add_node("performance_agent", performance_agent_node)
builder.add_node("security_agent", security_agent_node)
builder.add_node("cost_agent", cost_agent_node)
builder.add_node("decision_layer", decision_layer_node)
builder.add_node("final_response", final_response_node)

builder.add_edge(START, "preflight")
builder.add_conditional_edges("preflight", route_after_preflight)
builder.add_conditional_edges("target_llm", route_after_target)
builder.add_edge("agent_fanout", "security_agent")
builder.add_edge("agent_fanout", "cost_agent")
builder.add_edge("agent_fanout", "performance_agent")
builder.add_edge("performance_agent", "decision_layer")
builder.add_edge("security_agent", "decision_layer")
builder.add_edge("cost_agent", "decision_layer")
builder.add_edge("decision_layer", "final_response")
builder.add_edge("final_response", END)

controlplane_graph = builder.compile()
