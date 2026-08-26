import asyncio
import unittest
from unittest.mock import patch

from agents.security_agent import SecurityAgent


class SecurityMatrixTests(unittest.TestCase):
    def make_agent(self):
        agent = SecurityAgent.__new__(SecurityAgent)
        agent.pii_patterns = {
            "API Key / Secret Token": r"(?i)(api[_-]?key|secret[_-]?key|bearer|token)[\s=:]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
            "Password": r"(?i)password[\s=:]+['\"]?[^\s'\"]{8,}['\"]?",
            "Credit Card Number": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN / Tax ID": r"\b\d{3}-\d{2}-\d{4}\b",
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
        agent.approved_external_destinations = {"external_vendor"}
        agent.policy_database = [
            {
                "id": "POL-001",
                "condition": lambda ctx: bool(set(ctx.get("pattern_matches", [])) & {"API Key / Secret Token", "Password"}),
                "decision": "BLOCK",
                "reason": "Hardcoded API Keys or Secrets are strictly forbidden.",
            },
            {
                "id": "POL-005",
                "condition": lambda ctx: ctx.get("sensitivity") == "Highly Restricted",
                "decision": "BLOCK",
                "reason": "Highly Restricted data must not be processed by an LLM.",
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
                "id": "POL-003",
                "condition": lambda ctx: ctx.get("sensitivity") == "Public" and not ctx.get("pattern_matches"),
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
        agent.retriever = None
        agent.api_key = "dummy_key_for_testing"
        agent.trust_database = {
            "internal_systems": ["db", "backend", "internal_api"],
            "external_systems": ["user_browser", "public_api", "external_vendor"],
        }
        return agent

    def context(self, result):
        return result["security_findings"][0]["context"]

    def assert_deterministic(self, result, decision, policy_id):
        self.assertEqual(result["security_decision"], decision)
        self.assertEqual(result["policy_source"], "DETERMINISTIC")
        self.assertIn(policy_id, {policy["policy_id"] for policy in result["matched_policies"]})

    def test_public_clean_content_allows_deterministically(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Public", "confidence": 0.99}):
            result = agent.scan_output("Published product documentation", "Summarize this.", dest="external_vendor")

        self.assert_deterministic(result, "ALLOW", "POL-003")
        self.assertEqual(self.context(result)["pattern_matches"], [])
        self.assertTrue(self.context(result)["trust_boundary_crossed"])

    def test_internal_clean_content_allows_without_fallback(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output("Internal operational memo with no protected findings.", "Summarize this.", dest="external_vendor")

        self.assert_deterministic(result, "ALLOW", "POL-006")
        self.assertEqual(self.context(result)["sensitivity"], "Internal")
        self.assertTrue(self.context(result)["trust_boundary_crossed"])

    def test_internal_email_external_requires_approval(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}):
            result = agent.scan_output("alice.johnson@example.com", "Show contact.", dest="external_vendor")

        self.assert_deterministic(result, "REQUIRE_APPROVAL", "POL-004")
        self.assertEqual(self.context(result)["sensitivity"], "Internal")
        self.assertIn("Email Address", self.context(result)["pattern_matches"])
        self.assertTrue(self.context(result)["trust_boundary_crossed"])

    def test_internal_phone_external_requires_approval(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}):
            result = agent.scan_output("+1-202-555-0147", "Show contact.", dest="external_vendor")

        self.assert_deterministic(result, "REQUIRE_APPROVAL", "POL-004")
        self.assertEqual(self.context(result)["sensitivity"], "Internal")
        self.assertIn("Phone Number", self.context(result)["pattern_matches"])

    def test_api_key_is_blocked_and_skips_external_checks(self):
        self.assert_restricted_skips_checks("API_KEY=sk-test-1234567890abcdef1234567890abcdef", "API Key / Secret Token")

    def test_password_is_blocked_and_skips_external_checks(self):
        self.assert_restricted_skips_checks("password=SuperSecretPassword123456", "Password")

    def test_credit_card_is_blocked_and_skips_external_checks(self):
        self.assert_restricted_skips_checks("4111 1111 1111 1111", "Credit Card Number")

    def test_ssn_is_blocked_and_skips_external_checks(self):
        self.assert_restricted_skips_checks("123-45-6789", "SSN / Tax ID")

    def assert_restricted_skips_checks(self, text, category):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output(text, "Inspect this.", dest="external_vendor")

        self.assert_deterministic(result, "BLOCK", "POL-005")
        self.assertIn(category, self.context(result)["pattern_matches"])
        self.assertEqual(self.context(result)["sensitivity"], "Highly Restricted")

    def test_confidential_external_content_blocks_without_external_classifier(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output(
                "CONFIDENTIAL: unreleased product roadmap for 2027.",
                "Summarize this.",
                source="internal_api",
                dest="external_vendor",
            )

        self.assert_deterministic(result, "BLOCK", "POL-002")
        self.assertEqual(self.context(result)["sensitivity"], "Confidential")
        self.assertTrue(self.context(result)["trust_boundary_crossed"])

    def test_generic_source_code_is_not_automatically_confidential(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Public", "confidence": 0.9}):
            result = agent.scan_output("Explain what source code means.", "A general definition.", dest="external_vendor")

        self.assertNotEqual(self.context(result)["sensitivity"], "Confidential")

    def test_proprietary_algorithm_external_content_blocks(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")):
            result = agent.scan_output(
                "CONFIDENTIAL proprietary algorithm design.",
                "Summarize this.",
                source="internal_api",
                dest="external_vendor",
            )

        self.assert_deterministic(result, "BLOCK", "POL-002")
        self.assertEqual(self.context(result)["sensitivity"], "Confidential")

    def test_internal_pii_to_internal_destination_does_not_trigger_boundary_rule(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", return_value={"sensitivity": "Internal", "confidence": 0.9}):
            result = agent.scan_output("alice.johnson@example.com", "Show contact.", dest="internal_api")

        self.assertNotEqual(result["security_decision"], "REQUIRE_APPROVAL")
        self.assertFalse(self.context(result)["trust_boundary_crossed"])

    def test_multiple_violations_with_api_key_block(self):
        agent = self.make_agent()
        with patch.object(agent, "context_classifier", side_effect=AssertionError("classifier must not run")), patch.object(
            agent, "policy_rag", side_effect=AssertionError("fallback must not run")
        ):
            result = agent.scan_output(
                "API_KEY=sk-test-1234567890abcdef1234567890abcdef alice.johnson@example.com",
                "Review this.",
                dest="external_vendor",
            )

        self.assert_deterministic(result, "BLOCK", "POL-001")
        self.assertIn("Email Address", self.context(result)["pattern_matches"])
        self.assertIn("API Key / Secret Token", self.context(result)["pattern_matches"])

    def test_ambiguous_fallback_fails_safe(self):
        agent = self.make_agent()
        result = agent.scan_output("This content has no applicable deterministic policy.", "Evaluate it.", dest="unknown_destination")

        self.assertEqual(result["security_decision"], "FLAG")
        self.assertEqual(result["policy_source"], "LLM_FALLBACK")

    def test_preflight_secret_blocks_before_target_llm(self):
        self.assert_preflight_blocks("API_KEY=sk-test-1234567890abcdef1234567890abcdef")

    def test_preflight_password_blocks_before_target_llm(self):
        self.assert_preflight_blocks("password=SuperSecretPassword123456")

    def test_preflight_confidential_blocks_before_target_llm(self):
        self.assert_preflight_blocks("CONFIDENTIAL unreleased product roadmap")

    def assert_preflight_blocks(self, prompt):
        import app.main as main

        async def invoke():
            return await main.chat_endpoint(user_id="matrix", use_case="internal_copilot", prompt=prompt, file=None)

        with patch.object(main.injection_scanner, "scan", return_value=(prompt, True, 0.0)), patch.object(
            main, "call_target_llm", side_effect=AssertionError("target LLM must not run")
        ):
            result = asyncio.run(invoke())

        self.assertEqual(result["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main()
