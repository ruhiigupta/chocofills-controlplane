import os
import re
import json
import shutil
import subprocess
import tempfile
from typing import Dict, Any
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

class SecurityAgent:
    def __init__(self):
        # OpenRouter Configuration
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.trufflehog_path = os.getenv("TRUFFLEHOG_PATH", r"C:\trufflehog\trufflehog.exe")
        if self.api_key:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key,
            )
        else:
            self.client = None
        # Using a highly-capable, free open source model on OpenRouter
        self.model = "nvidia/nemotron-3.5-lightning"

        # Initialize RAG for Policy Fallback
        self.retriever = None
        policy_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "policies", "corporate_ai_policy.txt")
        try:
            if os.path.exists(policy_path):
                loader = TextLoader(policy_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                splits = text_splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectorstore = FAISS.from_documents(splits, embeddings)
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 2})
                print(f"[SecurityAgent] Initialized Policy RAG with {len(splits)} chunks.")
            else:
                print(f"[SecurityAgent] Warning: Policy file not found. RAG disabled.")
        except Exception as e:
            print(f"[SecurityAgent] Failed to initialize RAG: {e}")

        # Pattern Scanner definitions
        self.pii_patterns = {
            "API Key / Secret Token": r'(?i)(api[\s_-]?key|secret[\s_-]?key|bearer|token)[\s=:]+[\'"]?([a-zA-Z0-9_\-]{20,})[\'"]?',
            "Password": r"(?i)\bpassword\b(?:[\s=:]+|[\s]+is[\s]+)['\"]?[^\s'\"]{8,}['\"]?",
            "Credit Card Number": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN / Tax ID": r"\b\d{3}-\d{2}-\d{4}\b",
            "Email Address": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "Phone Number": r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
        }
        
        # Policy Database
        self.policy_database = [
            {"id": "POL-001", "condition": lambda ctx: bool(set(ctx.get("pattern_matches", [])) & {"API Key / Secret Token", "Password"}), "decision": "BLOCK", "reason": "Hardcoded API Keys or Secrets are strictly forbidden."},
            {"id": "POL-005", "condition": lambda ctx: ctx.get("sensitivity") == "Highly Restricted", "decision": "BLOCK", "reason": "Highly Restricted data must not be processed by an LLM."},
            {"id": "POL-002", "condition": lambda ctx: ctx.get("sensitivity") in {"Confidential", "Highly Restricted"} and ctx.get("trust_boundary_crossed", False), "decision": "BLOCK", "reason": "Confidential or Highly Restricted data cannot cross trust boundaries."},
            {"id": "POL-004", "condition": lambda ctx: ctx.get("sensitivity") == "Internal" and bool(set(ctx.get("pattern_matches", [])) & {"Email Address", "Phone Number"}) and ctx.get("trust_boundary_crossed", False), "decision": "REQUIRE_APPROVAL", "reason": "Internal PII requires redaction or human approval before crossing an external boundary."},
            {"id": "POL-003", "condition": lambda ctx: ctx.get("sensitivity") == "Public" and not ctx.get("pattern_matches"), "decision": "ALLOW", "reason": "Public data without detected PII or secrets can flow freely."},
            {"id": "POL-006", "condition": lambda ctx: ctx.get("sensitivity") == "Internal" and not ctx.get("pattern_matches") and ctx.get("destination") in self.approved_external_destinations, "decision": "ALLOW", "reason": "Internal data without protected findings may flow to an approved external LLM."}
        ]

        self.approved_external_destinations = {"external_vendor"}
        self.decision_priority = {
            "BLOCK": 4,
            "REQUIRE_APPROVAL": 3,
            "REDACT": 2,
            "FLAG": 1,
            "ALLOW": 0,
        }

        # Mock Trust Database for Information Flow Analyzer
        self.trust_database = {
            "internal_systems": ["db", "backend", "internal_api"],
            "external_systems": ["user_browser", "public_api", "external_vendor"]
        }
        self.trufflehog_path = (
            os.getenv("TRUFFLEHOG_PATH")
            or shutil.which("trufflehog")
            or (
                r"C:\trufflehog\trufflehog.exe"
                if os.path.exists(r"C:\trufflehog\trufflehog.exe")
                else None
            )
        )

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Helper to call OpenRouter LLM expecting JSON output."""
        try:
            if not self.api_key or self.client is None:
                print("[SecurityAgent] OPENROUTER_API_KEY is not configured; skipping real LLM call.")
                return {}

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            # Sometimes models return markdown blocks even with json_object format
            content = response.choices[0].message.content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            return json.loads(content)
        except Exception as e:
            print(f"[SecurityAgent LLM Error] {e}")
            return {}

    def pattern_scanner(self, text: str) -> Dict[str, Any]:
        """Layer 1: Deterministic Pattern Scanning."""

        matches_found = []
        findings = []

        # Store spans of high-priority sensitive matches so lower-priority
        # detectors do not classify substrings inside them.
        protected_spans = []

        # Detect highly sensitive credentials first.
        priority_patterns = [
            "API Key / Secret Token",
            "Password",
            "Credit Card Number",
            "SSN / Tax ID",
        ]

        for entity_type in priority_patterns:
            if entity_type not in self.pii_patterns:
                continue

            pattern = self.pii_patterns[entity_type]

            for match in re.finditer(pattern, text):
                matches_found.append(entity_type)

                findings.append({
                    "type": entity_type,
                    "matched": True,
                    "confidence": 1.0 if entity_type == "API Key / Secret Token" else 0.98,
                })

                protected_spans.append(match.span())

                # One finding per entity type is enough for the current API.
                break

        # Detect remaining PII types, but ignore matches that occur inside
        # already-detected sensitive values.
        for entity_type in self.pii_patterns:
            if entity_type in priority_patterns:
                continue

            pattern = self.pii_patterns[entity_type]

            for match in re.finditer(pattern, text):
                start, end = match.span()

                overlaps_protected = any(
                    start < protected_end and end > protected_start
                    for protected_start, protected_end in protected_spans
                )

                if overlaps_protected:
                    continue

                matches_found.append(entity_type)

                findings.append({
                    "type": entity_type,
                    "matched": True,
                    "confidence": 0.98,
                })

                break

        return {
            "type": "PatternScanner",
            "confidence_score": max(
                (finding["confidence"] for finding in findings),
                default=0.0
            ),
            "matches": matches_found,
            "findings": findings,
        }

    def _detect_obvious_secret_patterns(self, text: str) -> list[str]:
        """Fall back to deterministic credential detection when TruffleHog misses obvious tokens."""
        if not isinstance(text, str):
            return []

        cleaned = text.strip()
        if not cleaned:
            return []

        direct_token_patterns = [
            r"(?i)\bgh[pousr]_[A-Za-z0-9_]{20,}\b",
            r"(?i)\bgithub_pat_[A-Za-z0-9_]{20,}\b",
            r"(?i)\bxox[bapors]-[A-Za-z0-9-]{10,}\b",
            r"(?i)\bAKIA[0-9A-Z]{16}\b",
            r"(?i)\bASIA[0-9A-Z]{16}\b",
            r"(?i)\bAIza[0-9A-Za-z\-_]{20,}\b",
            r"(?i)\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b",
            r"(?i)-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        ]

        for pattern in direct_token_patterns:
            if re.search(pattern, cleaned):
                return ["Credential"]

        assignment_patterns = [
            r"(?i)(?:^|[\s;,(\[{])(?:api[_ -]?key|secret[_ -]?key|access[_ -]?key[_ -]?id|secret[_ -]?access[_ -]?key|token|password|private[_ -]?key)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=@:%#]{8,}['\"]?(?=\s|$|[;,)\]}.])",
            r"(?i)(?:^|[\s;,(\[{])(?:authorization)\s*:\s*bearer\s+[A-Za-z0-9._~+\-/=]{20,}(?=\s|$|[;,)\]}.])",
        ]

        for pattern in assignment_patterns:
            if re.search(pattern, cleaned):
                return ["Credential"]

        # Do not classify generic mentions like "What is an API key?" or "How should passwords be stored?"
        # A value must be present in an actual credential assignment or token-bearing format.
        return []

    def _redact_sensitive_values(self, text: str) -> str:
        """Strip obvious credential material from any returned reason text."""
        if not isinstance(text, str):
            return ""

        redacted = re.sub(
            r"(?i)(gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|xox[bapors]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{20,}|sk_(?:live|test)_[A-Za-z0-9]{16,})",
            "[REDACTED_CREDENTIAL]",
            text,
        )
        redacted = re.sub(
            r"(?i)((?:api[_ -]?key|secret[_ -]?key|access[_ -]?key[_ -]?id|secret[_ -]?access[_ -]?key|token|password|private[_ -]?key)[\s:=]+)['\"]?[A-Za-z0-9_\-./=+]{12,}['\"]?",
            r"\1[REDACTED_CREDENTIAL]",
            redacted,
        )
        return redacted.strip() or "Sensitive content detected; credential material has been redacted."

    def _scan_with_trufflehog(self, text: str) -> Dict[str, Any]:
        """Scan text with TruffleHog and return only safe metadata."""

        if not isinstance(text, str) or not text.strip():
            return {
                "available": True,
                "found": False,
                "categories": [],
                "reason": "No content to scan.",
            }

        cleaned = text.strip()

        scanner_path = (
            self.trufflehog_path
            or os.getenv("TRUFFLEHOG_PATH")
            or shutil.which("trufflehog")
        )

        if not scanner_path or not os.path.isfile(scanner_path):
            print("[SecurityAgent] TruffleHog unavailable.")

            # Deterministic fallback for obvious credential formats.
            fallback = self._detect_obvious_secret_patterns(cleaned)

            if fallback:
                return {
                    "available": False,
                    "found": True,
                    "categories": fallback,
                    "reason": "A credential-like value was detected by deterministic fallback.",
                }

            return {
                "available": False,
                "found": False,
                "categories": [],
                "reason": "TruffleHog unavailable; continuing with LLM classification.",
            }

        temp_handle, temp_path = tempfile.mkstemp(
            prefix="security_agent_th_",
            suffix=".txt",
        )
        os.close(temp_handle)

        try:
            with open(
                temp_path,
                "w",
                encoding="utf-8",
                errors="surrogateescape",
            ) as file_handle:
                file_handle.write(cleaned)

            completed = subprocess.run(
                [
                    scanner_path,
                    "filesystem",
                    "--json",
                    "--results=verified,unverified,unknown",
                    temp_path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )

            stdout = completed.stdout or ""
            stderr = completed.stderr or ""

            # Exit code 1 means the scanner itself failed.
            if completed.returncode == 1:
                print(
                    f"[SecurityAgent] TruffleHog scan failed: "
                    f"{self._redact_sensitive_values(stderr)}"
                )

                fallback = self._detect_obvious_secret_patterns(cleaned)

                if fallback:
                    return {
                        "available": True,
                        "found": True,
                        "categories": fallback,
                        "reason": "A credential-like value was detected by deterministic fallback.",
                    }

                return {
                    "available": True,
                    "found": False,
                    "categories": [],
                    "reason": "TruffleHog scan failed; continuing with LLM classification.",
                }

            findings = []

            for line in stdout.splitlines():
                line = line.strip()

                if not line.startswith("{"):
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Actual TruffleHog finding
                if (
                    result.get("DetectorName")
                    or result.get("DetectorType")
                ):
                    findings.append(result)

            if findings:
                categories = []

                for finding in findings:
                    detector = finding.get("DetectorName")

                    if isinstance(detector, str) and detector.strip():
                        categories.append(detector.strip())

                categories = list(dict.fromkeys(categories))

                return {
                    "available": True,
                    "found": True,
                    "categories": categories or ["credential"],
                    "reason": (
                        "TruffleHog detected credential or secret material "
                        "in the supplied context."
                    ),
                }

            # IMPORTANT:
            # Scanner ran successfully but detected nothing.
            fallback = self._detect_obvious_secret_patterns(cleaned)

            if fallback:
                return {
                    "available": True,
                    "found": True,
                    "categories": fallback,
                    "reason": (
                        "A credential-like value was detected by "
                        "deterministic fallback."
                    ),
                }

            return {
                "available": True,
                "found": False,
                "categories": [],
                "reason": "TruffleHog completed successfully with no detected secrets.",
            }

        except subprocess.TimeoutExpired:
            print("[SecurityAgent] TruffleHog timed out.")

            fallback = self._detect_obvious_secret_patterns(cleaned)

            if fallback:
                return {
                    "available": True,
                    "found": True,
                    "categories": fallback,
                    "reason": "A credential-like value was detected by deterministic fallback.",
                }

            return {
                "available": True,
                "found": False,
                "categories": [],
                "reason": "TruffleHog timed out; continuing with LLM classification.",
            }

        except Exception as e:
            print(f"[SecurityAgent] TruffleHog error: {e}")

            fallback = self._detect_obvious_secret_patterns(cleaned)

            if fallback:
                return {
                    "available": True,
                    "found": True,
                    "categories": fallback,
                    "reason": "A credential-like value was detected by deterministic fallback.",
                }

            return {
                "available": False,
                "found": False,
                "categories": [],
                "reason": "TruffleHog could not complete; continuing with LLM classification.",
            }

        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    def context_classifier(self, prompt: str, response: str) -> Dict[str, Any]:
        """Layer 2: LLM Context Classifier."""

        if not isinstance(prompt, str):
            prompt = ""
        if not isinstance(response, str):
            response = ""

        combined_text = "\n".join(part.strip() for part in (prompt, response) if part and part.strip())
        if combined_text:
            secret_result = self._scan_with_trufflehog(combined_text)
        else:
            secret_result = {"found": False, "categories": [], "reason": "TruffleHog unavailable; continuing with LLM classification."}

        if secret_result.get("found"):
            categories = secret_result.get("categories") or ["Credential"]
            return {
                "type": "ContextClassifier",
                "sensitivity": "Highly Restricted",
                "confidence": 1.0,
                "categories": categories,
                "reason": secret_result.get("reason") or "A secret or credential was detected in the supplied context.",
            }

        sys_prompt = """You are an enterprise data sensitivity classification engine.

    Your task is to classify the INFORMATION contained in the user's prompt and the LLM response.

    IMPORTANT:
    Classify the actual information/data being transmitted, NOT merely the topic being discussed.

    Use exactly one sensitivity level:
    - Public
    - Internal
    - Confidential
    - Highly Restricted

    CLASSIFICATION RULES:

    1. PUBLIC
    Information intentionally public or generally available to anyone.
    Examples:
    - General knowledge
    - Educational information
    - Public documentation
    - Publicly available facts
    - General questions about passwords, APIs, security, encryption, etc.
    when NO actual sensitive data is provided.

    2. INTERNAL
    Non-public organizational information that is not confidential or highly restricted.
    Examples:
    - Internal processes
    - Internal procedures
    - Internal project information
    - Non-public operational information
    - Internal discussions without confidential/restricted data

    3. CONFIDENTIAL
    Sensitive non-public organizational information whose disclosure could harm the organization.
    Examples:
    - Financial reports
    - Non-public financial information
    - Unreleased product plans
    - Proprietary algorithms
    - Confidential source code
    - Non-public business strategy
    - Confidential technical documentation

    4. HIGHLY RESTRICTED
    Actual sensitive data requiring the highest protection.
    Examples:
    - Passwords
    - API keys
    - Authentication tokens
    - Bearer tokens
    - Private keys
    - SSNs / government identification numbers
    - Credit/debit card numbers
    - Protected health information
    - Other actual credentials or authentication-sensitive identifiers

    CRITICAL DISTINCTIONS:

    - A question ABOUT a sensitive topic is NOT itself sensitive.
    Example: "What is an API key?" -> Public.

    - General security advice is NOT automatically Internal or Confidential.
    Example: "How should API keys be stored?" -> Public.

    - Actual sensitive data MUST be classified according to that data.
    Example: "My password is P@ssw0rd123" -> Highly Restricted.

    - Do not infer sensitive information that is not explicitly present.

    - If multiple sensitivity levels are present, choose the HIGHEST applicable level.

    - Evaluate BOTH the user prompt AND the LLM response.

    - If either contains sensitive information, use the highest applicable level.

    EMAIL AND PHONE NUMBER RULE:

    Ordinary personal contact information is NOT Highly Restricted by itself.

    - A normal email address such as:
    "test@example.com"
    -> Internal

    - A normal phone number such as:
    "+919876543210"
    -> Internal

    Do NOT classify an email address or phone number as Highly Restricted
    merely because it is personal information.

    Only classify contact information as Highly Restricted if it is part of
    another explicitly Highly Restricted dataset or credential context.

    Examples:

    "Email me at test@example.com"
    -> Internal

    "Call me at +919876543210"
    -> Internal

    "My password is P@ssw0rd123"
    -> Highly Restricted

    "My SSN is 123-45-6789"
    -> Highly Restricted

    - When uncertain between two levels, choose the more restrictive level,
    EXCEPT for ordinary email addresses and phone numbers. Those remain
    Internal unless another rule explicitly makes the surrounding data
    Highly Restricted.

    CONFIDENCE:
    Return a number between 0.0 and 1.0.

    CATEGORIES:
    Return concise categories describing sensitive information actually detected.
    If no sensitive category is present, return [].

    REASON:
    Give a brief explanation based only on information present.

    Return ONLY valid JSON:

    {
        "sensitivity": "Public|Internal|Confidential|Highly Restricted",
        "confidence": 0.0,
        "categories": [],
        "reason": "brief explanation"
    }
    """

        user_prompt = (
            f"Prompt: {prompt}\n"
            f"Response: {response}\n"
            "Classify the sensitivity of the complete context."
        )

        try:
            llm_out = self._call_llm_json(sys_prompt, user_prompt)

            # Empty / failed LLM output
            if not isinstance(llm_out, dict) or not llm_out:
                return {
                    "type": "ContextClassifier",
                    "sensitivity": "Internal",
                    "confidence": 0.5,
                    "categories": [],
                    "reason": "LLM classification unavailable; using safe fallback.",
                }

            # Validate sensitivity
            valid_sensitivities = {
                "Public",
                "Internal",
                "Confidential",
                "Highly Restricted",
            }

            sensitivity = llm_out.get("sensitivity", "Internal")

            if sensitivity not in valid_sensitivities:
                sensitivity = "Internal"

            # Validate confidence
            confidence = llm_out.get("confidence", 0.5)

            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not 0.0 <= confidence <= 1.0
            ):
                confidence = 0.5

            # Validate categories
            categories = llm_out.get("categories", [])

            if not isinstance(categories, list):
                categories = []

            # Validate reason
            reason = llm_out.get("reason", "Classified by context classifier")

            if not isinstance(reason, str) or not reason.strip():
                reason = "Classified by context classifier"

            reason = self._redact_sensitive_values(reason)

            return {
                "type": "ContextClassifier",
                "sensitivity": sensitivity,
                "confidence": confidence,
                "categories": categories,
                "reason": reason,
            }

        except Exception as e:
            print(f"[SecurityAgent Context Classifier Error] {e}")

            return {
                "type": "ContextClassifier",
                "sensitivity": "Internal",
                "confidence": 0.5,
                "categories": [],
                "reason": "LLM classification failed; using safe fallback.",
            }

    def _deterministic_sensitivity(self, text: str, pattern_matches: list[str]) -> str | None:
        restricted = {"API Key / Secret Token", "Password", "Credit Card Number", "SSN / Tax ID"}
        if restricted & set(pattern_matches):
            return "Highly Restricted"

        confidential_terms = (
            # Explicit confidentiality
            "confidential",
            "strictly confidential",
            "private",
            "non-public",
            "not for public distribution",
            "internal use only",
            "do not share",
            "do not distribute",

            # Business / strategy
            "proprietary business strategy",
            "business strategy",
            "internal strategy",
            "strategic plan",
            "business plan",
            "go-to-market strategy",
            "competitive strategy",
            "pricing strategy",
            "sales strategy",
            "marketing strategy",

            # Product / roadmap
            "unreleased roadmap",
            "product roadmap",
            "internal roadmap",
            "unreleased product",
            "unannounced product",
            "product launch plan",
            "launch plan",
            "product strategy",

            # Financial
            "financial report",
            "financial statement",
            "financial forecast",
            "revenue forecast",
            "earnings forecast",
            "budget",
            "internal budget",
            "profit margin",
            "cost structure",
            "pricing information",

            # Intellectual property / technology
            "proprietary algorithm",
            "proprietary technology",
            "proprietary model",
            "trade secret",
            "internal source code",
            "proprietary source code",
            "internal architecture",
            "system architecture",
            "technical design",
            "internal documentation",

            # Corporate / operational
            "internal policy",
            "internal procedure",
            "internal process",
            "internal documentation",
            "employee information",
            "customer information",
            "vendor information",
            "supplier information",
            "contract terms",
            "business agreement",

            # Legal / strategic
            "merger plan",
            "acquisition plan",
            "acquisition target",
            "legal strategy",
            "litigation strategy",
            "settlement terms",
        )
        lower_text = text.lower()

        # Generic/conceptual discussion should NOT be classified as Confidential.
        # Confidentiality requires evidence that actual non-public organizational
        # information is being disclosed.
        generic_discussion_patterns = (
            "explain financial forecasting",
            "what is financial forecasting",
            "how does financial forecasting work",
            "explain financial reports",
            "what is a financial report",
            "explain business strategy",
            "what is a business strategy",
            "explain product roadmaps",
            "what is a product roadmap",
            "explain proprietary algorithms",
            "what is a proprietary algorithm",
            "explain system architecture",
            "what is system architecture",
            "explain internal policies",
            "what is an internal policy",
            "explain api keys",
            "what is an api key",
            "explain password policies",
            "what are password policies",
        )

        if any(pattern in lower_text for pattern in generic_discussion_patterns):
            return "Internal"

        if any(term in lower_text for term in confidential_terms):
            return "Confidential"

    def _sanitize_for_classifier(self, text: str, pattern_matches: list[str]) -> str:
        sanitized = text
        for entity_type, pattern in self.pii_patterns.items():
            if entity_type in pattern_matches:
                sanitized = re.sub(pattern, f"[{entity_type}]", sanitized)
        return sanitized

    def info_flow_analyzer(self, source: str, dest: str) -> Dict[str, Any]:
        """Layer 3: Information Flow Analyzer."""
        crossed = (source in self.trust_database["internal_systems"] and 
                   dest in self.trust_database["external_systems"])
        
        return {
            "type": "InfoFlowAnalyzer",
            "trust_boundary_crossed": crossed,
            "status": "AUTHORIZED" if not crossed else "REVIEW_REQUIRED"
        }

    def policy_engine(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 4: Deterministic Policy Engine."""
        matched_policies = []
        for policy in self.policy_database:
            try:
                if policy["condition"](context):
                    matched_policies.append({
                        "decision": policy["decision"],
                        "reason": policy["reason"],
                        "policy_id": policy["id"],
                        "confidence_score": 1.0
                    })
            except Exception as e:
                print(f"[PolicyEngine Error] {e}")

        if not matched_policies:
            return {"decision": "NO_MATCH", "matched_policies": []}

        selected_match = max(
            matched_policies,
            key=lambda result: self.decision_priority.get(result["decision"], -1),
        )
        selected = dict(selected_match)
        selected["score"] = {
            "BLOCK": 100.0,
            "REQUIRE_APPROVAL": 70.0,
            "REDACT": 60.0,
            "FLAG": 50.0,
            "ALLOW": 0.0,
        }[selected["decision"]]
        selected["matched_policies"] = matched_policies
        selected["policy_source"] = "DETERMINISTIC"
        return selected

    def policy_rag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 5: Fallback LLM Policy Evaluation (Policy RAG)."""
        # Retrieve relevant policies
        retrieved_policies = ""
        if self.retriever:
            # Create a search query based on the context
            query = f"""
            Data Sensitivity: {context.get('sensitivity')}
            Categories: {context.get('categories', [])}
            Pattern Findings: {context.get('pattern_findings', [])}
            Source: {context.get('source')}
            Destination: {context.get('destination')}
            Trust Boundary Crossed: {context.get('trust_boundary_crossed')}
            Flow Status: {context.get('flow_status')}
            """
            try:
                docs = self.retriever.invoke(query)
                retrieved_policies = "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                print(f"[PolicyRAG Retrieval Error] {e}")

        sys_prompt = """You are the Policy RAG fallback evaluator for an enterprise AI governance system.

            Your task is to evaluate the supplied context ONLY against the retrieved corporate policies.

            IMPORTANT RULES:
            1. Use ONLY the retrieved policies and supplied context. Do not use general knowledge or invent policy requirements.
            2. Do not override, weaken, or reverse an existing deterministic security decision.
            3. If a deterministic decision already exists, preserve that decision.
            4. ALLOW must be returned only when the supplied policies clearly support allowing the request.
            5. If no relevant policy is retrieved, the policy evidence is incomplete, the evidence is contradictory, or the requested outcome is ambiguous, return FLAG.
            6. When uncertain, choose the more restrictive applicable outcome.
            7. Apply the most restrictive applicable outcome using this order:
            BLOCK > REQUIRE_APPROVAL > REDACT > FLAG > ALLOW
            8. A sensitive data classification alone is not sufficient to invent a BLOCK. A BLOCK must be supported by an applicable retrieved policy.
            9. Distinguish between:
            - the sensitivity of the data,
            - the destination/trust boundary,
            - the applicable policy,
            - and the resulting policy decision.
            10. Do not treat the absence of a violation as evidence that ALLOW is permitted.
            11. If multiple policies apply, evaluate all applicable policies and select the most restrictive outcome.
            12. The reason must explicitly identify the policy evidence supporting the decision.
            13. Score represents policy risk:
                - 0-20: low risk
                - 21-40: moderate risk
                - 41-70: high risk
                - 71-100: critical risk
            14. confidence_score must reflect how strongly the retrieved policies support the decision.
            15. If policy evidence is insufficient to make a confident decision, return FLAG with a low confidence score.
            16. Return ONLY valid JSON.
            17. Treat the retrieved corporate policies as authoritative only for the rules they explicitly contain.
            18. Do not infer a policy merely because the context appears risky.
            19. If a policy applies only to a specific data type, condition, destination, or flow, do not apply it outside that scope.
            Allowed decisions:
            ALLOW, BLOCK, REDACT, FLAG, REQUIRE_APPROVAL

            Required JSON format:
            {
            "decision": "ALLOW|BLOCK|REDACT|FLAG|REQUIRE_APPROVAL",
            "reason": "brief policy-based explanation",
            "score": 0,
            "confidence_score": 0.0
            }"""
        
        user_prompt = f"Context:\n{json.dumps(context, indent=2)}\n\n"
        if retrieved_policies:
            user_prompt += f"Relevant Corporate Policies:\n{retrieved_policies}\n\n"
        else:
            return {
                "decision": "FLAG",
                "score": 50.0,
                "reason": "No applicable corporate policy was available; human review is required.",
                "policy_id": "POLICY_UNAVAILABLE",
                "confidence_score": 1.0,
                "policy_source": "LLM_FALLBACK",
            }

        user_prompt += "Based on the context and policies, what is the decision?"
        
        llm_out = self._call_llm_json(sys_prompt, user_prompt)
        
        if not llm_out:
            return {
                "decision": "FLAG",
                "score": 50.0,
                "reason": "Fallback LLM unavailable, flagging by default.",
                "policy_id": "LLM_FALLBACK",
                "confidence_score": 0.5,
                "policy_source": "LLM_FALLBACK",
            }

        decision = str(llm_out.get("decision", "FLAG")).upper()
        if decision not in self.decision_priority:
            return {
                "decision": "FLAG",
                "score": 50.0,
                "reason": "Fallback LLM returned an invalid policy decision; human review is required.",
                "policy_id": "LLM_FALLBACK",
                "confidence_score": 0.0,
                "policy_source": "LLM_FALLBACK",
            }
            
        try:
            score = max(0.0, min(100.0, float(llm_out.get("score", 50.0))))
            confidence = max(
                0.0,
                min(1.0, float(llm_out.get("confidence_score", 0.8)))
            )
        except (TypeError, ValueError):
            return {
                "decision": "FLAG",
                "score": 50.0,
                "reason": "Fallback LLM returned an invalid score; human review is required.",
                "policy_id": "LLM_FALLBACK",
                "confidence_score": 0.0,
                "policy_source": "LLM_FALLBACK",
            }

        return {
            "decision": decision,
            "score": score,
            "reason": llm_out.get("reason", "Evaluated by LLM Fallback"),
            "policy_id": "LLM_FALLBACK",
            "confidence_score": confidence,
            "policy_source": "LLM_FALLBACK",
        }

    def scan_output(self, llm_response: str, user_prompt: str = "", source: str = "internal_api", dest: str = "external_vendor") -> Dict[str, Any]:
        """The main entrypoint that merges all inputs and evaluates policies."""
        # Hard constraints must be established before any external classifier call.
        combined_text = f"{user_prompt}\n{llm_response}"
        pattern_results = self.pattern_scanner(combined_text)
        flow_results = self.info_flow_analyzer(source, dest)
        deterministic_sensitivity = self._deterministic_sensitivity(
            combined_text, pattern_results["matches"]
        )

        if deterministic_sensitivity == "Highly Restricted":
            context_results = {
                "sensitivity": "Highly Restricted",
                "confidence": 1.0,
                "categories": pattern_results["matches"],
                "reason": "Deterministic restricted-data pattern detected.",
            }
        elif deterministic_sensitivity == "Confidential" and flow_results["trust_boundary_crossed"]:
            context_results = {
                "sensitivity": "Confidential",
                "confidence": 1.0,
                "categories": pattern_results["matches"],
                "reason": "Deterministic confidential-data signal crossed an external boundary.",
            }
        else:
            sanitized_prompt = self._sanitize_for_classifier(user_prompt, pattern_results["matches"])
            sanitized_response = self._sanitize_for_classifier(llm_response, pattern_results["matches"])
            context_results = self.context_classifier(sanitized_prompt, sanitized_response)
        
        # 2. Merger
        if deterministic_sensitivity:
            classified_sensitivity = deterministic_sensitivity
            sensitivity_confidence = 1.0
        else:
            classified_sensitivity = context_results.get("sensitivity", "Internal")
            sensitivity_confidence = context_results.get("confidence", 0.5)

        merged_context = {
            "pattern_matches": pattern_results["matches"],
            "pattern_findings": pattern_results["findings"],
            "pattern_confidence": pattern_results["confidence_score"],
            "sensitivity": classified_sensitivity,
            "sensitivity_confidence": sensitivity_confidence,
            "categories": sorted(set(pattern_results["matches"]) | set(context_results.get("categories", []))),
            "source": source,
            "destination": dest,
            "flow": flow_results,
            "trust_boundary_crossed": flow_results["trust_boundary_crossed"],
            "flow_status": flow_results["status"]
        }
        
        # 3. Policy Engine Evaluation
        decision_result = self.policy_engine(merged_context)
        
        # 4. Fallback LLM Evaluation if No Match
        if decision_result.get("decision") == "NO_MATCH":
            decision_result = self.policy_rag(merged_context)

        policy_source = decision_result.get("policy_source", "DETERMINISTIC")
            
        # Format the output
        final_score = float(decision_result.get("score", 0.0))
        decision = decision_result.get("decision", "UNKNOWN")

        if decision == "BLOCK":
            final_score = max(final_score, 80.0)
            status = "FAIL"

        elif decision in ("REQUIRE_APPROVAL", "REDACT", "FLAG"):
            final_score = max(final_score, 40.0)
            status = "FLAG"

        elif decision == "ALLOW":
            status = "PASS" if final_score < 40.0 else "FLAG"

        else:
            status = "FLAG"
            
        return {
            "security_score": final_score,
            "security_status": status,
            "security_findings": [
                {
                    "policy_id": decision_result.get("policy_id", "UNKNOWN"),
                    "decision": decision_result.get("decision", "UNKNOWN"),
                    "reason": decision_result.get("reason", ""),
                    "confidence": decision_result.get("confidence_score", 0.0),
                    "context": merged_context
                }
            ],
            "security_decision": decision_result.get("decision", "UNKNOWN"),
            "matched_policies": decision_result.get("matched_policies", []),
            "policy_source": policy_source,
        }