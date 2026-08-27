import os
import re
import json
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
        self.api_key = os.getenv("OPENROUTER_API_KEY", "dummy_key_for_testing")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
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
            "API Key / Secret Token": r"(?i)(api[_-]?key|secret[_-]?key|bearer|token)[\s=:]+['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
            "Password": r"(?i)password[\s=:]+['\"]?[^\s'\"]{8,}['\"]?",
            "Credit Card Number": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN / Tax ID": r"\b\d{3}-\d{2}-\d{4}\b",
            "Email Address": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "Phone Number": r"\b\+?[1-9]\d{1,14}\b"
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

    def _call_llm_json(self, system_prompt: str, user_prompt: str) -> dict:
        """Helper to call OpenRouter LLM expecting JSON output."""
        try:
            if self.api_key == "dummy_key_for_testing":
                return {}
                
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"}
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
        for entity_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                matches_found.append(entity_type)
        findings = [
            {
                "type": entity_type,
                "matched": True,
                "confidence": 1.0 if entity_type == "API Key / Secret Token" else 0.98,
            }
            for entity_type in matches_found
        ]
        return {
            "type": "PatternScanner",
            "confidence_score": max((finding["confidence"] for finding in findings), default=0.0),
            "matches": matches_found,
            "findings": findings,
        }

    def context_classifier(self, prompt: str, response: str) -> Dict[str, Any]:
        """Layer 2: LLM Context Classifier."""
        sys_prompt = """You are an enterprise data sensitivity classification engine.

    Your task is to classify the INFORMATION contained in the user's prompt and the LLM response.

    IMPORTANT:
    Classify the sensitivity of the actual information/data being transmitted, NOT the topic being discussed, the user's intent, or the fact that the request is related to security or an organization.

    Use exactly one sensitivity level:
    - Public
    - Internal
    - Confidential
    - Highly Restricted

    CLASSIFICATION RULES:

    1. PUBLIC
    Information that is intentionally public or generally available to anyone.
    Examples:
    - General knowledge
    - Public educational information
    - Public documentation
    - Publicly available facts
    - General questions such as "What is machine learning?"
    - Questions about passwords, APIs, security, etc. when NO actual secret or sensitive data is provided

    2. INTERNAL
    Non-public organizational information that is not confidential or highly restricted.
    Examples:
    - Internal processes
    - Internal procedures
    - Internal project information
    - Non-public operational information
    - Internal discussions that do not contain confidential or highly restricted data

    3. CONFIDENTIAL
    Sensitive non-public organizational information whose disclosure could harm the organization.
    Examples:
    - Financial reports or non-public financial information
    - Unreleased product plans or roadmaps
    - Proprietary algorithms
    - Confidential source code
    - Non-public business strategy
    - Confidential technical documentation

    4. HIGHLY RESTRICTED
    Information that must receive the highest level of protection.
    Examples:
    - Passwords
    - API keys
    - Authentication tokens
    - Access tokens
    - Bearer tokens
    - Private keys
    - Secrets or credentials
    - SSNs / government identification numbers
    - Credit/debit card numbers
    - Protected health information (PHI)
    - Other regulated or authentication-sensitive identifiers

    CRITICAL DISTINCTIONS:

    - A QUESTION ABOUT sensitive information is NOT itself sensitive.
    Example: "What is an API key?" -> Public.

    - A REQUEST TO DISCUSS password policies is NOT itself a password.
    Example: "How should a company improve its password policy?" -> Public/Internal depending on context.

    - ACTUAL sensitive data appearing in the prompt or response MUST be classified according to that data.
    Example: "My password is P@ssw0rd123" -> Highly Restricted.

    - The presence of words such as "password", "API key", "SSN", "security", or "confidential" does NOT automatically determine the classification. Determine whether actual sensitive information is present.

    - Do not infer sensitive information that is not explicitly present.

    - Do not downgrade information merely because it appears inside an otherwise harmless question.

    - If multiple sensitivity levels are present, choose the HIGHEST applicable level.

    - Evaluate both the USER PROMPT and LLM RESPONSE.

    - If either contains sensitive information, classify according to the highest applicable sensitivity level across both fields.

    - Do not let harmless content in one field downgrade sensitive content in the other.

    - If actual secrets or regulated identifiers are detected, always classify as Highly Restricted.

    - Email addresses and phone numbers are personal information and must NOT automatically be classified as Public merely because they are not secrets.

    - When uncertain between two levels, choose the more restrictive level.

    CONFIDENCE:
    Return a confidence between 0.0 and 1.0 representing how certain you are about the classification.

    CATEGORIES:
    Return concise categories describing the sensitive information actually detected.
    Examples:
    ["Password"]
    ["API Key"]
    ["Financial Information"]
    ["Source Code"]
    ["General Knowledge"]
    If no sensitive category is present, return [].

    REASON:
    Give a brief explanation based only on the information present in the prompt and response.

    Return ONLY valid JSON in exactly this structure:
    {
        "sensitivity": "Public|Internal|Confidential|Highly Restricted",
        "confidence": 0.0,
        "categories": [],
        "reason": "brief explanation"
    }"""
        user_prompt = f"Prompt: {prompt}\nResponse: {response}\nClassify the sensitivity of the response context."
        
        llm_out = self._call_llm_json(sys_prompt, user_prompt)
        
        if not llm_out:
            return {"sensitivity": "Internal", "confidence": 0.5}
            
        return {
            "type": "ContextClassifier",
            "sensitivity": llm_out.get("sensitivity", "Internal"),
            "confidence": llm_out.get("confidence", 0.5),
            "categories": llm_out.get("categories", []),
            "reason": llm_out.get("reason", "Classified by context classifier"),
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
        if any(term in text.lower() for term in confidential_terms):
            return "Confidential"
        return None

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