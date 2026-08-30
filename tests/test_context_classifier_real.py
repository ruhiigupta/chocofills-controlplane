from agents.security_agent import SecurityAgent


def run_tests():
    a = SecurityAgent()

    cases = [
        # =========================
        # PUBLIC
        # =========================
        (
            "Public - machine learning",
            "What is machine learning?",
            "Machine learning is a field of artificial intelligence.",
            "Public",
        ),
        (
            "Public - API key definition",
            "What is an API key?",
            "An API key is a credential used to authenticate API requests.",
            "Public",
        ),
        (
            "Public - password definition",
            "What is a password?",
            "A password is information used to authenticate a user.",
            "Public",
        ),
        (
            "Public - encryption",
            "What is encryption?",
            "Encryption transforms readable data into ciphertext.",
            "Public",
        ),
        (
            "Public - API security advice",
            "How should API keys be stored securely?",
            "Use a secret manager and rotate credentials regularly.",
            "Public",
        ),
        (
            "Public - password security advice",
            "How should passwords be protected?",
            "Use hashing, salting, MFA, and secure credential storage.",
            "Public",
        ),
        (
            "Public - credit card security discussion",
            "Explain credit card security.",
            "Use tokenization and avoid storing raw card numbers.",
            "Public",
        ),

        # =========================
        # INTERNAL
        # =========================
        (
            "Internal - deployment process",
            "Our internal deployment process requires two engineering approvals.",
            "This is an internal operational procedure.",
            "Internal",
        ),
        (
            "Internal - internal procedure",
            "Our engineering team follows an internal deployment checklist.",
            "The checklist requires approval from the engineering lead.",
            "Internal",
        ),
        (
            "Internal - operational information",
            "Our internal support team handles incidents through the internal escalation process.",
            "This procedure is not publicly documented.",
            "Internal",
        ),

        # =========================
        # CONFIDENTIAL
        # =========================
        (
            "Confidential - roadmap",
            "Our unreleased product roadmap includes Product X launching in Q4.",
            "The launch plan has not been publicly announced.",
            "Confidential",
        ),
        (
            "Confidential - financial",
            "Our confidential financial report shows revenue of $12 million.",
            "These figures have not been publicly released.",
            "Confidential",
        ),
        (
            "Confidential - proprietary algorithm",
            "Our proprietary ranking algorithm uses a private scoring method.",
            "The implementation is confidential company information.",
            "Confidential",
        ),
        (
            "Confidential - source code",
            "Here is our confidential internal source code for the ranking service.",
            "The code is proprietary and not publicly available.",
            "Confidential",
        ),

        # =========================
        # HIGHLY RESTRICTED
        # =========================
        (
            "Highly Restricted - password",
            "My password is P@ssw0rd123",
            "You should change it immediately.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - password assignment",
            "password=P@ssw0rd123",
            "Credential supplied by the user.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - API key",
            "Here is my API key: abcdefghijklmnopqrstuvwxyz123456",
            "Do not share API keys publicly.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - bearer token",
            "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9abcdef1234567890",
            "Do not expose authentication tokens.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - SSN",
            "My SSN is 123-45-6789",
            "That is sensitive personal information.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - credit card",
            "My card number is 4111 1111 1111 1111",
            "Do not share payment information.",
            "Highly Restricted",
        ),
        (
            "Highly Restricted - token",
            "token=abcdefghijklmnopqrstuvwxyz123456",
            "Do not expose credentials.",
            "Highly Restricted",
        ),

        # =========================
        # SENSITIVE DATA ONLY IN RESPONSE
        # =========================
        (
            "Response only - SSN",
            "Summarize this information.",
            "The employee SSN is 123-45-6789.",
            "Highly Restricted",
        ),
        (
            "Response only - password",
            "What information did the user provide?",
            "The user's password is P@ssw0rd123.",
            "Highly Restricted",
        ),
        (
            "Response only - API key",
            "Summarize the credentials.",
            "The API key is abcdefghijklmnopqrstuvwxyz123456.",
            "Highly Restricted",
        ),

        # =========================
        # MIXED SENSITIVITY
        # =========================
        (
            "Mixed - public + password",
            "What is machine learning? My password is P@ssw0rd123",
            "Machine learning is a field of AI.",
            "Highly Restricted",
        ),
        (
            "Mixed - internal + password",
            "Our deployment process requires two approvals.",
            "The deployment password is P@ssw0rd123.",
            "Highly Restricted",
        ),
        (
            "Mixed - confidential + SSN",
            "Our unreleased roadmap contains Product X.",
            "The employee SSN is 123-45-6789.",
            "Highly Restricted",
        ),
        (
            "Mixed - confidential + API key",
            "Our unreleased product uses a private API.",
            "The API key is abcdefghijklmnopqrstuvwxyz123456.",
            "Highly Restricted",
        ),

        # =========================
        # PERSONAL INFORMATION
        # =========================
        (
            "Personal - email",
            "My email is test@example.com",
            "Thanks.",
            "Internal",
        ),
        (
            "Personal - phone",
            "Call me at +919876543210",
            "Okay.",
            "Internal",
        ),

        # =========================
        # WORDS ABOUT SECRETS WITHOUT ACTUAL SECRET
        # =========================
        (
            "No secret - password word",
            "Please explain password policies.",
            "Organizations should enforce strong passwords.",
            "Public",
        ),
        (
            "No secret - API key word",
            "How do API keys work?",
            "They authenticate requests to an API.",
            "Public",
        ),
        (
            "No secret - SSN word",
            "What is an SSN?",
            "An SSN is a government-issued identifier.",
            "Public",
        ),
        (
            "No secret - credit card word",
            "What is a credit card number?",
            "It is an identifier associated with a payment card.",
            "Public",
        ),
    ]

    passed = 0
    failed = 0

    print("\n" + "=" * 80)
    print("REAL LLM CONTEXT CLASSIFIER TEST")
    print("=" * 80)

    for i, (name, prompt, response, expected) in enumerate(cases, 1):

        try:
            result = a.context_classifier(prompt, response)

            actual = result.get("sensitivity")
            confidence = result.get("confidence")
            categories = result.get("categories")
            reason = result.get("reason")

            if actual == expected:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1

            print(f"\n[{status}] {i}. {name}")
            print(f"Expected    : {expected}")
            print(f"Actual      : {actual}")
            print(f"Confidence  : {confidence}")
            print(f"Categories  : {categories}")
            print(f"Reason      : {reason}")

        except Exception as e:
            failed += 1
            print(f"\n[ERROR] {i}. {name}")
            print(f"Exception   : {e}")

    total = len(cases)

    print("\n" + "=" * 80)
    print(f"RESULT: {passed}/{total} PASSED | {failed}/{total} FAILED")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()