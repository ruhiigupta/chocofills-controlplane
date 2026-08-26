import unittest
from unittest.mock import patch

from agents.security_agent import SecurityAgent
from graph.workflow import controlplane_graph


class SecurityPipelineTests(unittest.TestCase):
    def make_agent(self):
        agent = SecurityAgent.__new__(SecurityAgent)
        agent.pii_patterns = {
            "API Key / Secret Token": r"(?i)(api[_-]?key|secret[_-]?key|bearer|token)[\s=:]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
            "Password": r"(?i)password[\s=:]+['\"]?[^\s'\"]{8,}['\"]?",
            "Email Address": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "Phone Number": r"\b\+?[1-9]\d{1,14}\b",
        }
        agent.decision_priority = {
            "BLOCK": 4,
            "REQUIRE_APPROVAL": 3,
            "REDACT": 2,
            "FLAG": 1,
            "ALLOW": 0,
        }
        agent.policy_database = [
            {
                "id": "POL-001",
                "condition": lambda ctx: "API Key / Secret Token" in ctx.get("pattern_matches", []),
                "decision": "BLOCK",
                "reason": "Hardcoded API Keys or Secrets are strictly forbidden.",
            },
            {
                "id": "POL-002",
                "condition": lambda ctx: ctx.get("sensitivity") in {"Confidential", "Highly Restricted"}
                and ctx.get("trust_boundary_crossed", False),
                "decision": "BLOCK",
                "reason": "Confidential or Highly Restricted data cannot cross trust boundaries.",
            },
            {
                "id": "POL-004",
                "condition": lambda ctx: ctx.get("sensitivity") == "Internal"
                and bool(set(ctx.get("pattern_matches", [])) & {"Email Address", "Phone Number"})
                and ctx.get("trust_boundary_crossed", False),
                "decision": "REQUIRE_APPROVAL",
                "reason": "Internal PII requires redaction or human approval before crossing an external boundary.",
            },
            {
                "id": "POL-005",
                "condition": lambda ctx: ctx.get("sensitivity") == "Highly Restricted",
                "decision": "BLOCK",
                "reason": "Highly Restricted data must not be processed by an LLM.",
            },
            {
                "id": "POL-003",
                "condition": lambda ctx: ctx.get("sensitivity") == "Public"
                and not ctx.get("pattern_matches"),
                "decision": "ALLOW",
                "reason": "Public data without detected PII or secrets can flow freely.",
            },
            {
                "id": "POL-006",
                "condition": lambda ctx: ctx.get("sensitivity") == "Internal"
                and not ctx.get("pattern_matches")
                and ctx.get("destination") in agent.approved_external_destinations,
                "decision": "ALLOW",
                "reason": "Internal data without protected findings may flow to an approved external LLM.",
            },
        ]
        agent.approved_external_destinations = {"external_vendor"}
        agent.retriever = None
        agent.api_key = "dummy_key_for_testing"
        agent.trust_database = {
            "internal_systems": ["db", "backend", "internal_api"],
            "external_systems": ["user_browser", "public_api", "external_vendor"],
        }
        return agent

    def test_email_and_phone_findings_are_preserved(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.85}):
            result = agent.scan_output(
                "alice.johnson@example.com +1-202-555-0147",
                "show contact details",
            )

        types = {finding["type"] for finding in result["security_findings"][0]["context"]["pattern_findings"]}
        self.assertEqual(types, {"Email Address", "Phone Number"})
        self.assertEqual(result["policy_source"], "DETERMINISTIC")
        self.assertEqual(result["security_decision"], "REQUIRE_APPROVAL")

    def test_public_classification_does_not_allow_pii(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Public", "confidence": 0.70}):
            result = agent.scan_output("+1-202-555-0147", "show phone")

        self.assertEqual(result["security_findings"][0]["context"]["sensitivity"], "Public")
        self.assertEqual(result["security_decision"], "FLAG")

    def test_secret_is_deterministically_blocked(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Confidential", "confidence": 0.9}), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output(
                "API_KEY=sk-test-1234567890abcdef1234567890abcdef",
                "inspect configuration",
            )

        self.assertEqual(result["security_decision"], "BLOCK")
        self.assertEqual(result["policy_source"], "DETERMINISTIC")
        self.assertIn("POL-001", {policy["policy_id"] for policy in result["matched_policies"]})

    def test_secret_skips_classifier_and_fallback(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output("API_KEY=sk-test-1234567890abcdef1234567890abcdef", "inspect config")

        self.assertEqual(result["security_decision"], "BLOCK")
        self.assertEqual(result["policy_source"], "DETERMINISTIC")

    def test_secret_in_user_prompt_skips_external_checks(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier") as classifier_mock, patch.object(agent, "policy_rag") as rag_mock:
            result = agent.scan_output(
                "The configuration contains an API key.",
                "Please analyze this configuration: API_KEY=sk-test-1234567890abcdef1234567890abcdef",
            )

        self.assertEqual(result["security_decision"], "BLOCK")
        classifier_mock.assert_not_called()
        rag_mock.assert_not_called()

    def test_password_in_user_prompt_skips_external_checks(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier") as classifier_mock, patch.object(agent, "policy_rag") as rag_mock:
            result = agent.scan_output(
                "I understand.",
                "Remember this password: password=SuperSecretPassword123",
            )

        self.assertEqual(result["security_decision"], "BLOCK")
        classifier_mock.assert_not_called()
        rag_mock.assert_not_called()

    def test_user_prompt_pii_is_sanitized_for_classifier(self):
        agent = self.make_agent()
        captured = {}

        def capture_classifier(prompt, response):
            captured["prompt"] = prompt
            captured["response"] = response
            return {"sensitivity": "Internal", "confidence": 0.8, "categories": []}

        with patch.object(agent, "context_classifier", side_effect=capture_classifier):
            agent.scan_output(
                "Alice Johnson is the project contact.",
                "Summarize this contact: alice.johnson@example.com",
            )

        self.assertIn("[Email Address]", captured["prompt"])
        self.assertNotIn("alice.johnson@example.com", captured["prompt"])
        self.assertNotIn("alice.johnson@example.com", captured["response"])

    def test_generic_source_code_wording_is_not_confidential(self):
        agent = self.make_agent()
        with patch.object(
            agent,
            "context_classifier",
            return_value={"sensitivity": "Public", "confidence": 0.9, "categories": []},
        ):
            result = agent.scan_output("Explain what source code means.", "A source code definition.")

        self.assertNotEqual(result["security_findings"][0]["context"]["sensitivity"], "Confidential")

    def test_generic_terraform_wording_is_not_confidential(self):
        agent = self.make_agent()
        with patch.object(
            agent,
            "context_classifier",
            return_value={"sensitivity": "Public", "confidence": 0.9, "categories": []},
        ):
            result = agent.scan_output("Explain what Terraform is.", "Terraform is an infrastructure tool.")

        self.assertNotEqual(result["security_findings"][0]["context"]["sensitivity"], "Confidential")

    def test_strong_confidential_external_signal_blocks(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier") as classifier_mock:
            result = agent.scan_output(
                "CONFIDENTIAL: unreleased product roadmap for 2027.",
                "summarize roadmap",
                source="internal_api",
                dest="external_vendor",
            )

        self.assertEqual(result["security_findings"][0]["context"]["sensitivity"], "Confidential")
        self.assertEqual(result["security_decision"], "BLOCK")
        classifier_mock.assert_not_called()

    def test_confidential_external_skips_raw_classifier(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output("CONFIDENTIAL unreleased roadmap details", "summarize roadmap")

        self.assertEqual(result["security_decision"], "BLOCK")
        self.assertEqual(result["security_findings"][0]["context"]["sensitivity"], "Confidential")

    def test_public_content_can_be_allowed(self):
        agent = self.make_agent()
        with patch.object(
            agent, "context_classifier", return_value={"sensitivity": "Public", "confidence": 0.99}
        ):
            result = agent.scan_output("Published product documentation", "summarize this")

        self.assertEqual(result["security_decision"], "ALLOW")
        self.assertEqual(result["policy_source"], "DETERMINISTIC")

    def test_internal_non_pii_is_deterministically_allowed_without_fallback(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output(
                "Internal operational memo with no protected findings.",
                "Summarize this memo.",
                source="internal_api",
                dest="external_vendor",
            )

        self.assertEqual(result["security_decision"], "ALLOW")
        self.assertEqual(result["policy_source"], "DETERMINISTIC")

    def test_internal_email_requires_approval_without_reclassification(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}):
            result = agent.scan_output(
                "alice.johnson@example.com",
                "Show the employee contact.",
                source="internal_api",
                dest="external_vendor",
            )

        context = result["security_findings"][0]["context"]
        self.assertEqual(context["sensitivity"], "Internal")
        self.assertEqual(result["security_decision"], "REQUIRE_APPROVAL")

    def test_internal_phone_requires_approval(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}):
            result = agent.scan_output(
                "+1-202-555-0147",
                "Show the employee contact.",
                source="internal_api",
                dest="external_vendor",
            )

        self.assertEqual(result["security_decision"], "REQUIRE_APPROVAL")

    def test_fallback_allow_cannot_override_internal_pii_approval(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not override deterministic approval")
        ):
            result = agent.scan_output("alice.johnson@example.com", "Show contact", source="internal_api", dest="external_vendor")

        self.assertEqual(result["security_decision"], "REQUIRE_APPROVAL")

    def test_classifier_receives_sanitized_pii(self):
        agent = self.make_agent()
        classifier_inputs = []

        def capture_classifier(prompt, response):
            classifier_inputs.extend([prompt, response])
            return {"sensitivity": "Internal", "confidence": 0.8}

        with patch.object(agent, "context_classifier", side_effect=capture_classifier):
            agent.scan_output("Contact alice.johnson@example.com", "show contact")

        self.assertTrue(all("alice.johnson@example.com" not in value for value in classifier_inputs))

    def test_detected_secret_cannot_be_downgraded_by_classifier(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Public", "confidence": 1.0}):
            result = agent.scan_output("API_KEY=sk-test-1234567890abcdef1234567890abcdef", "inspect config")

        context = result["security_findings"][0]["context"]
        self.assertEqual(context["sensitivity"], "Highly Restricted")
        self.assertEqual(result["security_decision"], "BLOCK")

    def test_flow_boundary_is_deterministic(self):
        agent = self.make_agent()
        self.assertTrue(agent.info_flow_analyzer("internal_api", "external_vendor")["trust_boundary_crossed"])
        self.assertFalse(agent.info_flow_analyzer("internal_api", "internal_api")["trust_boundary_crossed"])

    def test_multiple_policy_priority_keeps_block(self):
        agent = self.make_agent()
        result = agent.policy_engine(
            {
                "pattern_matches": ["API Key / Secret Token"],
                "sensitivity": "Highly Restricted",
                "trust_boundary_crossed": True,
            }
        )
        self.assertEqual(result["decision"], "BLOCK")
        self.assertEqual(len(result["matched_policies"]), 3)

    def test_fallback_without_policy_fails_safe(self):
        agent = self.make_agent()
        result = agent.policy_rag({"sensitivity": "Internal"})
        self.assertEqual(result["decision"], "FLAG")
        self.assertNotEqual(result["decision"], "ALLOW")
    def test_password_is_highly_restricted(self):
        agent = self.make_agent()

        result = agent.scan_output(
            "password=SuperSecretPassword123456",
            "inspect configuration",
        )

        self.assertEqual(
            result["security_decision"],
            "BLOCK",
        )


    def test_confidential_external_flow_is_blocked(self):
        agent = self.make_agent()

        result = agent.scan_output(
            "CONFIDENTIAL unreleased roadmap for Q4",
            "summarize roadmap",
            source="internal_api",
            dest="external_vendor",
        )

        self.assertEqual(
            result["security_decision"],
            "BLOCK",
        )


    def test_confidential_internal_flow_is_not_blocked_by_boundary_policy(self):
        agent = self.make_agent()

        with patch.object(
            agent,
            "context_classifier",
            return_value={
                "sensitivity": "Confidential",
                "confidence": 0.95,
                "categories": [],
            },
        ):
            result = agent.scan_output(
                "CONFIDENTIAL unreleased roadmap",
                "summarize roadmap",
                source="internal_api",
                dest="internal_api",
            )

        self.assertNotEqual(
            result["security_decision"],
            "BLOCK",
        )
    def test_graph_preserves_policy_decision(self):
        state = {
            "user_id": "test",
            "use_case": "internal_copilot",
            "source": "internal_api",
            "destination": "external_vendor",
            "user_prompt": "test",
            "system_prompt": None,
            "source_documents": [],
            "llm_response": "safe response",
            "model_name": "test",
            "preflight_risk_score": 0.0,
            "performance_score": 0.0,
            "performance_status": "PENDING",
            "factual_findings": [],
            "relevance_findings": [],
            "security_score": 0.0,
            "security_status": "PENDING",
            "security_decision": "PENDING",
            "security_findings": [],
            "matched_policies": [],
            "policy_source": "PENDING",
            "cost_score": 0.0,
            "cost_status": "PENDING",
            "input_tokens": 1,
            "output_tokens": 1,
            "estimated_cost": 0.0,
            "ttft_latency_ms": 0.0,
            "unified_risk_score": 0.0,
            "final_action": "ALLOW",
            "audit_log": {},
        }
        security_result = {
            "security_score": 100.0,
            "security_status": "FAIL",
            "security_decision": "BLOCK",
            "security_findings": [],
            "matched_policies": [{"policy_id": "POL-001", "decision": "BLOCK"}],
            "policy_source": "DETERMINISTIC",
        }
        with patch("graph.workflow.security_agent_instance.scan_output", return_value=security_result):
            result = controlplane_graph.invoke(state)

        self.assertEqual(result["security_decision"], "BLOCK")
        self.assertEqual(result["final_action"], "BLOCK")


if __name__ == "__main__":
    unittest.main()
