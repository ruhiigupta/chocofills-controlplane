import json
import os
import subprocess
import sys
import tempfile

import pytest

from agents.security_agent import SecurityAgent


REAL_LLM_TEST = bool(os.getenv("OPENROUTER_API_KEY"))
TRUFFLEHOG_BIN = os.getenv("TRUFFLEHOG_PATH", r"C:\trufflehog\trufflehog.exe")


def _mock_secret_fixture() -> str:
    return "GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890"


def test_trufflehog_scan_detects_realistic_secret_fixture():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    result = agent._scan_with_trufflehog(_mock_secret_fixture())

    assert result["found"] is True
    assert result["categories"]
    assert result["reason"] in {
        "A secret or credential was detected in the supplied context.",
        "A credential-like value was detected by deterministic fallback.",
    }
    assert "ghp_" not in result["reason"]


def test_context_classifier_secret_gate_overrides_llm():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    result = agent.context_classifier(
        "The deployment configuration includes a GitHub token.",
        "Token: GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890",
    )

    assert result["sensitivity"] == "Highly Restricted"
    assert result["confidence"] == 1.0
    assert "ghp_" not in json.dumps(result)
    assert "Credential" in result["categories"] or result["categories"]


def test_no_secret_context_allows_llm_when_available():
    if not REAL_LLM_TEST:
        pytest.skip("OpenRouter key not configured")

    agent = SecurityAgent()
    result = agent.context_classifier(
        "What is machine learning?",
        "Machine learning is a field of AI.",
    )

    assert result["sensitivity"] in {"Public", "Internal", "Confidential"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["categories"], list)
    assert isinstance(result["reason"], str) and result["reason"].strip()


def test_normal_security_discussion_is_not_highly_restricted():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    result = agent.context_classifier(
        "How should API keys be stored securely?",
        "Use environment variables and rotate keys routinely.",
    )

    assert result["sensitivity"] != "Highly Restricted"

def test_security_concepts_without_actual_values_are_not_highly_restricted():
    agent = SecurityAgent()
    agent._scan_with_trufflehog = lambda text: {
        "found": False,
        "categories": [],
        "reason": "safe",
    }

    cases = [
        (
            "What is an API key?",
            "An API key authenticates API requests.",
        ),
        (
            "How should API keys be stored securely?",
            "Store them in a secret manager.",
        ),
        (
            "What is a password?",
            "A password is used for authentication.",
        ),
        (
            "API keys should never be committed to Git.",
            "This is a security best practice.",
        ),
        (
            "What does Authorization: Bearer mean?",
            "It is an HTTP authentication scheme.",
        ),
        (
            "Passwords should be stored securely.",
            "Use a secret manager.",
        ),
    ]

    for prompt, response in cases:
        result = agent.context_classifier(prompt, response)

        assert result["sensitivity"] != "Highly Restricted", (
            f"False positive for prompt={prompt!r}, response={response!r}: "
            f"{result}"
        )
        
def test_actual_credentials_are_highly_restricted():
    agent = SecurityAgent()

    cases = [
        (
            "api_key=sk_test_12345678901234567890",
            "Okay.",
        ),
        (
            "password=P@ssw0rd123",
            "Okay.",
        ),
        (
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef1234567890",
            "Okay.",
        ),
    ]

    for prompt, response in cases:
        result = agent.context_classifier(prompt, response)

        assert result["sensitivity"] == "Highly Restricted", (
            f"Credential was not detected: {result}"
        )
        assert result["confidence"] == 1.0
        assert result["categories"]
        
def test_trufflehog_executable_missing_returns_safe_status():
    agent = SecurityAgent()
    agent.trufflehog_path = r"C:\does\not\exist\trufflehog.exe"

    result = agent._scan_with_trufflehog("This text has no secret at all.")

    assert result["found"] is False
    assert result["categories"] == []


def test_trufflehog_timeout_is_handled_safely():
    agent = SecurityAgent()
    agent.trufflehog_path = TRUFFLEHOG_BIN

    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    original_run = subprocess.run

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0][0], timeout=kwargs.get("timeout", 15))

    try:
        subprocess.run = fake_run
        result = agent._scan_with_trufflehog("some text")
    finally:
        subprocess.run = original_run

    assert result["found"] is False
    assert result["categories"] == []


def test_malformed_trufflehog_output_is_ignored_safely():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    original_run = subprocess.run

    def fake_run(*args, **kwargs):
        class _Completed:
            stdout = "not-json\n"
            stderr = ""
        return _Completed()

    try:
        subprocess.run = fake_run
        result = agent._scan_with_trufflehog("a short phrase")
    finally:
        subprocess.run = original_run

    assert result["found"] is False
    assert result["categories"] == []


def test_context_classifier_validates_llm_output_fields():
    agent = SecurityAgent()
    agent._scan_with_trufflehog = lambda text: {"found": False, "categories": [], "reason": "safe"}

    invalid_cases = [
        {"sensitivity": "SECRET"},
        {"sensitivity": "Public", "confidence": -1},
        {"sensitivity": "Public", "confidence": "high"},
        {"sensitivity": "Public", "categories": "bad"},
        {"sensitivity": "Public", "reason": ""},
    ]

    for output in invalid_cases:
        agent._call_llm_json = lambda system_prompt, user_prompt, value=output: value
        result = agent.context_classifier("Prompt", "Response")
        assert "sensitivity" in result
        assert "confidence" in result
        assert "categories" in result
        assert "reason" in result
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["categories"], list)
        assert isinstance(result["reason"], str) and result["reason"].strip()


def test_secrets_in_response_are_flags_too():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    result = agent.context_classifier(
        "Summarize this.",
        "The server token is GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890.",
    )

    assert result["sensitivity"] == "Highly Restricted"
    assert "ghp_" not in json.dumps(result)


def test_multiple_secrets_are_detected():
    if not os.path.exists(TRUFFLEHOG_BIN):
        pytest.skip("TruffleHog CLI not installed")

    agent = SecurityAgent()
    result = agent.context_classifier(
        "A key and a second token.",
        "TOKEN=ghp_abcdefghijklmnopqrstuvwxyz1234567890 and GITHUB_PAT=github_pat_1234567890abcdefghijklmnop",
    )

    assert result["sensitivity"] == "Highly Restricted"
    assert result["confidence"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])

    print("\n========== ALL CONTEXT CLASSIFIER TESTS COMPLETED ==========")