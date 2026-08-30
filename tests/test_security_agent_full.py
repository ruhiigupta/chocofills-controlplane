from unittest.mock import patch
from agents.security_agent import SecurityAgent


def make_agent():
    a = SecurityAgent()
    return a


def matches(a, text):
    return a.pattern_scanner(text)["matches"]


def check(name, actual, expected):
    ok = actual == expected
    print(f"{'PASS' if ok else 'FAIL'} | {name}")
    if not ok:
        print(f"       expected: {expected}")
        print(f"       actual:   {actual}")
    return ok


def main():
    a = make_agent()

    passed = 0
    failed = 0

    # ============================================================
    # 1. API KEY / SECRET TESTS
    # ============================================================

    tests = [
        ("API question",
         "What is an API key?",
         []),

        ("API educational question",
         "What is an API key used for?",
         []),

        ("API security advice",
         "How should API keys be stored securely?",
         []),

        ("api_key assignment",
         "api_key=sk_test_12345678901234567890",
         ["API Key / Secret Token"]),

        ("api-key assignment",
         "api-key: sk_test_12345678901234567890",
         ["API Key / Secret Token"]),

        ("uppercase API_KEY",
         "API_KEY = abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("normal API key disclosure",
         "API key: abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("API key in sentence",
         "Here is our API key: abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("secret key disclosure",
         "secret key: abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("secret_key assignment",
         "secret_key=abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("bearer token",
         "Bearer abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("authorization bearer",
         "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef1234567890",
         ["API Key / Secret Token"]),

        ("token assignment",
         "token=abcdefghijklmnopqrstuvwxyz123456",
         ["API Key / Secret Token"]),

        ("token expired discussion",
         "This token is expired.",
         []),

        ("API rotation advice",
         "The API key should be rotated regularly.",
         []),

        ("secret management advice",
         "secret key management best practices",
         []),
    ]

    # ============================================================
    # 2. PASSWORD TESTS
    # ============================================================

    tests += [
        ("password question",
         "What is a password?",
         []),

        ("password policy question",
         "What is a password policy?",
         []),

        ("password policy statement",
         "password policy requires 12 characters",
         []),

        ("actual password with is",
         "My password is P@ssw0rd123",
         ["Password"]),

        ("password equals",
         "password=P@ssw0rd123",
         ["Password"]),

        ("password colon",
         "password: P@ssw0rd123",
         ["Password"]),

        ("quoted password",
         'password="P@ssw0rd123"',
         ["Password"]),

        ("short password",
         "password=abc",
         []),
    ]

    # ============================================================
    # 3. EMAIL TESTS
    # ============================================================

    tests += [
        ("email",
         "test@example.com",
         ["Email Address"]),

        ("email in sentence",
         "My email is test@example.com",
         ["Email Address"]),

        ("multiple emails",
         "Contact a@test.com or b@example.org",
         ["Email Address"]),

        ("fake email",
         "not-an-email",
         []),
    ]

    # ============================================================
    # 4. PHONE TESTS
    # ============================================================

    tests += [
        ("Indian phone",
         "+919876543210",
         ["Phone Number"]),

        ("Indian phone sentence",
         "My phone number is +919876543210",
         ["Phone Number"]),

        ("10 digit phone",
         "9876543210",
         ["Phone Number"]),

        ("US phone",
         "415-555-2671",
         ["Phone Number"]),

        ("US phone with country code",
         "+1 415 555 2671",
         ["Phone Number"]),

        ("phone sentence",
         "Call me at 9876543210",
         ["Phone Number"]),

        ("plain 12",
         "12",
         []),

        ("year",
         "2026",
         []),

        ("plain order ID",
         "Order ID: 123456789",
         []),
    ]

    # ============================================================
    # 5. CREDIT CARD TESTS
    # ============================================================

    tests += [
        ("Visa",
         "4111 1111 1111 1111",
         ["Credit Card Number"]),

        ("Visa without spaces",
         "4111111111111111",
         ["Credit Card Number"]),

        ("Visa with hyphens",
         "4111-1111-1111-1111",
         ["Credit Card Number"]),

        ("short card-like number",
         "123456789012",
         []),
    ]

    # ============================================================
    # 6. SSN / TAX ID TESTS
    # ============================================================

    tests += [
        ("SSN",
         "123-45-6789",
         ["SSN / Tax ID"]),

        ("invalid SSN format",
         "123456789",
         []),
    ]

    # ============================================================
    # 7. GENERAL KNOWLEDGE / FALSE POSITIVE TESTS
    # ============================================================

    tests += [
        ("machine learning",
         "What is machine learning?",
         []),

        ("Python",
         "How does Python work?",
         []),

        ("AI",
         "Explain artificial intelligence.",
         []),

        ("hello",
         "Hello world",
         []),

        ("date",
         "The year is 2026",
         []),

        ("order ID",
         "Order ID: 123456789",
         []),

        ("random number",
         "The answer is 123456789",
         []),
    ]

    # ============================================================
    # RUN PATTERN TESTS
    # ============================================================

    print("\n========== PATTERN SCANNER ==========\n")

    for name, text, expected in tests:
        actual = matches(a, text)

        if check(name, actual, expected):
            passed += 1
        else:
            failed += 1

    # ============================================================
    # 8. DETERMINISTIC SENSITIVITY
    # ============================================================

    print("\n========== DETERMINISTIC SENSITIVITY ==========\n")

    deterministic_tests = [
        (
            "API secret",
            "api_key=abcdefghijklmnopqrstuvwxyz123456",
            ["API Key / Secret Token"],
            "Highly Restricted",
        ),
        (
            "Password",
            "My password is P@ssw0rd123",
            ["Password"],
            "Highly Restricted",
        ),
        (
            "Credit card",
            "4111 1111 1111 1111",
            ["Credit Card Number"],
            "Highly Restricted",
        ),
        (
            "SSN",
            "123-45-6789",
            ["SSN / Tax ID"],
            "Highly Restricted",
        ),
        (
            "Confidential document",
            "This document is confidential.",
            [],
            "Confidential",
        ),
        (
            "Unreleased roadmap",
            "Our unreleased product roadmap is Q4.",
            [],
            "Confidential",
        ),
        (
            "Business strategy",
            "Our business strategy is changing.",
            [],
            "Confidential",
        ),
        (
            "System architecture",
            "Our system architecture uses Kubernetes.",
            [],
            "Confidential",
        ),
        (
            "Machine learning",
            "What is machine learning?",
            [],
            None,
        ),
        (
            "API educational question",
            "What is an API key?",
            [],
            "Internal",
        ),
    ]

    for name, text, pattern_matches, expected in deterministic_tests:
        actual = a._deterministic_sensitivity(text, pattern_matches)

        if check(name, actual, expected):
            passed += 1
        else:
            failed += 1

    # ============================================================
    # 9. INFORMATION FLOW
    # ============================================================

    print("\n========== INFORMATION FLOW ==========\n")

    flow_tests = [
        ("internal -> external",
         "internal_api",
         "external_vendor",
         True),

        ("db -> public",
         "db",
         "public_api",
         True),

        ("backend -> browser",
         "backend",
         "user_browser",
         True),

        ("internal -> internal",
         "backend",
         "db",
         False),

        ("external -> external",
         "user_browser",
         "external_vendor",
         False),

        ("unknown -> external",
         "unknown",
         "external_vendor",
         False),
    ]

    for name, source, dest, expected in flow_tests:
        result = a.info_flow_analyzer(source, dest)
        actual = result["trust_boundary_crossed"]

        if check(name, actual, expected):
            passed += 1
        else:
            failed += 1

    # ============================================================
    # 10. POLICY ENGINE
    # ============================================================

    print("\n========== POLICY ENGINE ==========\n")

    policy_tests = [
        (
            "API key -> BLOCK",
            {
                "pattern_matches": ["API Key / Secret Token"],
                "sensitivity": "Highly Restricted",
                "trust_boundary_crossed": True,
            },
            "BLOCK",
        ),
        (
            "password -> BLOCK",
            {
                "pattern_matches": ["Password"],
                "sensitivity": "Highly Restricted",
                "trust_boundary_crossed": True,
            },
            "BLOCK",
        ),
        (
            "confidential external -> BLOCK",
            {
                "pattern_matches": [],
                "sensitivity": "Confidential",
                "trust_boundary_crossed": True,
            },
            "BLOCK",
        ),
        (
            "internal email external -> approval",
            {
                "pattern_matches": ["Email Address"],
                "sensitivity": "Internal",
                "trust_boundary_crossed": True,
            },
            "REQUIRE_APPROVAL",
        ),
        (
            "internal phone external -> approval",
            {
                "pattern_matches": ["Phone Number"],
                "sensitivity": "Internal",
                "trust_boundary_crossed": True,
            },
            "REQUIRE_APPROVAL",
        ),
        (
            "public no PII -> ALLOW",
            {
                "pattern_matches": [],
                "sensitivity": "Public",
                "trust_boundary_crossed": False,
            },
            "ALLOW",
        ),
        (
            "nothing applicable -> NO_MATCH",
            {
                "pattern_matches": [],
                "sensitivity": "Unknown",
                "trust_boundary_crossed": False,
            },
            "NO_MATCH",
        ),
    ]

    for name, context, expected in policy_tests:
        result = a.policy_engine(context)
        actual = result["decision"]

        if check(name, actual, expected):
            passed += 1
        else:
            failed += 1

    # ============================================================
    # 11. FALLBACK SAFETY
    # ============================================================

    print("\n========== FALLBACK SAFETY ==========\n")

    # No RAG -> FLAG
    with patch.object(a, "retriever", None):
        result = a.policy_rag({
            "sensitivity": "Unknown",
            "flow_status": "REVIEW_REQUIRED",
        })

        if check(
            "RAG unavailable -> FLAG",
            result["decision"],
            "FLAG",
        ):
            passed += 1
        else:
            failed += 1

    # Invalid LLM decision -> FLA

    # ============================================================
    # 12. CONTEXT CLASSIFIER FALLBACK
    # ============================================================

    print("\n========== CONTEXT CLASSIFIER ==========\n")

    with patch.object(
        a,
        "_call_llm_json",
        return_value={},
    ):
        result = a.context_classifier(
            "What is machine learning?",
            "Machine learning is AI.",
        )

        if check(
            "empty LLM result -> Internal fallback",
            result["sensitivity"],
            "Internal",
        ):
            passed += 1
        else:
            failed += 1

    with patch.object(
        a,
        "_call_llm_json",
        return_value={
            "sensitivity": "Public",
            "confidence": 0.99,
            "categories": [],
            "reason": "General knowledge",
        },
    ):
        result = a.context_classifier(
            "What is machine learning?",
            "Machine learning is AI.",
        )

        if check(
            "valid LLM classification -> Public",
            result["sensitivity"],
            "Public",
        ):
            passed += 1
        else:
            failed += 1

    # ============================================================
    # 13. LLM MALFORMED OUTPUT
    # ============================================================

    print("\n========== MALFORMED LLM OUTPUT ==========\n")

    malformed_cases = [
        {"decision": "SAFE"},
        {"decision": "INVALID"},
        {"decision": None},
        {},
    ]

    for i, fake_output in enumerate(malformed_cases, 1):
        with patch.object(
            a,
            "_call_llm_json",
            return_value=fake_output,
        ):
            # Directly test the validation logic by making a minimal
            # fake retriever.
            class FakeRetriever:
                def invoke(self, query):
                    class Doc:
                        page_content = "A policy exists."
                    return [Doc()]

            a.retriever = FakeRetriever()

            result = a.policy_rag({
                "sensitivity": "Unknown",
                "flow_status": "REVIEW_REQUIRED",
            })

            expected = "FLAG"

            if check(
                f"malformed LLM output #{i} -> FLAG",
                result["decision"],
                expected,
            ):
                passed += 1
            else:
                failed += 1

    # ============================================================
    # 14. FINAL SUMMARY
    # ============================================================

    print("\n========================================")
    print(f"PASSED: {passed}")
    print(f"FAILED: {failed}")
    print(f"TOTAL:  {passed + failed}")
    print("========================================")

    if failed:
        print("\nThere are still failures. FIX THESE before moving on.")
        raise SystemExit(1)

    print("\nALL TESTS PASSED.")


if __name__ == "__main__":
    main()