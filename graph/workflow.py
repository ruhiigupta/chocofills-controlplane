import json
import os
import re
from typing import Any, Callable
from unittest import result

from langgraph.graph import END, START, StateGraph
from llm_guard.input_scanners import PromptInjection

from agents.cost_agent import cost_checker_node
from agents.performance_agent import PerformanceAgent
from agents.security_agent import SecurityAgent
from graph.state import ControlPlaneState


security_agent_instance = SecurityAgent()
performance_agent_instance = PerformanceAgent()


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

    return {
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
    return {
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

    return {"unified_risk_score": unified_risk, "final_action": final_action, "audit_log": audit_log}


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
