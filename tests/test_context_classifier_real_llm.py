from agents.security_agent import SecurityAgent


CASES = [
    # ---------------- PUBLIC ----------------
    (
        "Public - machine learning",
        "What is machine learning?",
        "Machine learning is a field of artificial intelligence.",
        "Public",
    ),
    (
        "Public - API question",
        "What is an API key?",
        "An API key is used to authenticate API requests.",
        "Public",
    ),
    (
        "Public - security advice",
        "How should API keys be stored securely?",
        "Use a secret manager and rotate credentials regularly.",
        "Public",
    ),

    # ---------------- INTERNAL ----------------
    (
        "Internal - deployment process",
        "What is our deployment process?",
        "Our internal deployment process requires two engineering approvals.",
        "Internal",
    ),
    (
        "Internal - internal procedure",
        "Describe our internal incident response procedure.",
        "Engineers must notify the security team and create an internal incident ticket.",
        "Internal",
    ),

    # ---------------- CONFIDENTIAL ----------------
    (
        "Confidential - roadmap",
        "What is our unreleased product roadmap?",
        "The company plans to launch Product X in Q4.",
        "Confidential",
    ),
    (
        "Confidential - financial",
        "What were the company's unreleased quarterly financial results?",
        "Revenue for the unreleased quarter was $12.4 million.",
        "Confidential",
    ),
    (
        "Confidential - source code",
        "Show our confidential authentication implementation.",
        "The proprietary authentication implementation uses an internal token validation algorithm.",
        "Confidential",
    ),

    # ---------------- HIGHLY RESTRICTED ----------------
    (
        "Highly Restricted - password",
        "My password is P@ssw0rd123",
        "You should change the password immediately.",
        "Highly Restricted",
    ),
    (
        "Highly Restricted - API key",
        "Here is the API key: abcdefghijklmnopqrstuvwxyz123456",
        "Never share API keys publicly.",
        "Highly Restricted",
    ),
    (
        "Highly Restricted - bearer token",
        "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef1234567890",
        "That authentication token should be revoked.",
        "Highly Restricted",
    ),
    (
        "Highly Restricted - SSN",
        "My SSN is 123-45-6789",
        "That identifier is highly sensitive.",
        "Highly Restricted",
    ),
    (
        "Highly Restricted - credit card",
        "My card number is 4111 1111 1111 1111",
        "Do not share payment information.",
        "Highly Restricted",
    ),

    # ---------------- PROMPT/RESPONSE CROSS-CHECK ----------------
    (
        "Sensitive response only",
        "Summarize this information.",
        "The employee password is P@ssw0rd123.",
        "Highly Restricted",
    ),
    (
        "Sensitive prompt only",
        "My password is P@ssw0rd123",
        "Your password should be changed.",
        "Highly Restricted",
    ),
    (
        "Sensitive response SSN",
        "What information did the user provide?",
        "The employee SSN is 123-45-6789.",
        "Highly Restricted",
    ),

    # ---------------- PERSONAL DATA ----------------
    (
        "Phone number",
        "Contact me at +919876543210",
        "I will use the provided contact information.",
        "Internal",
    ),
    (
        "Email address",
        "Email me at test@example.com",
        "I will use the provided email address.",
        "Internal",
    ),

    # ---------------- MIXED ----------------
    (
        "Confidential + public",
        "Explain machine learning using our unreleased product architecture.",
        "The architecture contains a proprietary internal inference component.",
        "Confidential",
    ),
    (
        "Public question + restricted response",
        "What did the employee provide?",
        "The employee provided their API key: abcdefghijklmnopqrstuvwxyz123456",
        "Highly Restricted",
    ),
]


def main():
    agent = SecurityAgent()

    passed = 0
    failed = 0

    print("\n" + "=" * 80)
    print("REAL LLM CONTEXT CLASSIFIER TEST")
    print("=" * 80)

    for name, prompt, response, expected in CASES:
        try:
            result = agent.context_classifier(prompt, response)

            actual = result.get("sensitivity")
            confidence = result.get("confidence")
            categories = result.get("categories")
            reason = result.get("reason")

            ok = actual == expected

            if ok:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"\n{status} | {name}")
            print(f"  Expected   : {expected}")
            print(f"  Actual     : {actual}")
            print(f"  Confidence : {confidence}")
            print(f"  Categories : {categories}")
            print(f"  Reason     : {reason}")

        except Exception as e:
            failed += 1
            print(f"\nERROR | {name}")
            print(f"  Exception: {type(e).__name__}: {e}")

    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{len(CASES)} passed")
    print(f"FAILED: {failed}")
    print("=" * 80)


if __name__ == "__main__":
    main()