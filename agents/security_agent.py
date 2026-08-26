import os
import re
import json
from concurrent.futures import ThreadPoolExecutor
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
        self.model = "google/gemma-2-9b-it:free"

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
            "Credit Card Number": r"\b(?:\d[ -]*?){13,16}\b",
            "SSN / Tax ID": r"\b\d{3}-\d{2}-\d{4}\b",
            "Email Address": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "Phone Number": r"\b\+?[1-9]\d{1,14}\b"
        }
        
        # Policy Database
        self.policy_database = [
            {"id": "POL-001", "condition": lambda ctx: "API Key / Secret Token" in ctx.get("pattern_matches", []), "decision": "BLOCK", "reason": "Hardcoded API Keys or Secrets are strictly forbidden."},
            {"id": "POL-002", "condition": lambda ctx: ctx.get("sensitivity") == "Highly Restricted" and ctx.get("trust_boundary_crossed", False), "decision": "BLOCK", "reason": "Highly Restricted data cannot cross trust boundaries."},
            {"id": "POL-003", "condition": lambda ctx: ctx.get("sensitivity") == "Public", "decision": "ALLOW", "reason": "Public data can flow freely."}
        ]

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
        sys_prompt = "You are a data sensitivity classifier. Output ONLY JSON with 'sensitivity' (choose from: Public, Internal, Confidential, Highly Restricted) and 'confidence' (0.0 to 1.0). No extra text."
        user_prompt = f"Prompt: {prompt}\nResponse: {response}\nClassify the sensitivity of the response context."
        
        llm_out = self._call_llm_json(sys_prompt, user_prompt)
        
        if not llm_out:
            return {"sensitivity": "Internal", "confidence": 0.5}
            
        return {
            "type": "ContextClassifier",
            "sensitivity": llm_out.get("sensitivity", "Internal"),
            "confidence": llm_out.get("confidence", 0.5)
        }

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
        selected["score"] = 100.0 if selected["decision"] == "BLOCK" else 0.0
        selected["matched_policies"] = matched_policies
        return selected

    def policy_rag(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Layer 5: Fallback LLM Policy Evaluation (Policy RAG)."""
        # Retrieve relevant policies
        retrieved_policies = ""
        if self.retriever:
            # Create a search query based on the context
            query = f"Data Sensitivity: {context.get('sensitivity')} | Flow: {context.get('flow_status')}"
            try:
                docs = self.retriever.invoke(query)
                retrieved_policies = "\n\n".join([d.page_content for d in docs])
            except Exception as e:
                print(f"[PolicyRAG Retrieval Error] {e}")

        sys_prompt = "You are a policy engine fallback. You are given an action context and relevant corporate policies. Decide if the action is ALLOW, BLOCK, REDACT, FLAG, or REQUIRE_APPROVAL. Output ONLY JSON with 'decision', 'reason', 'score' (0 to 100, 100 is block), and 'confidence_score' (0.0 to 1.0). No extra text."
        
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
            }

        user_prompt += "Based on the context and policies, what is the decision?"
        
        llm_out = self._call_llm_json(sys_prompt, user_prompt)
        
        if not llm_out:
            return {
                "decision": "FLAG",
                "score": 50.0,
                "reason": "Fallback LLM unavailable, flagging by default.",
                "policy_id": "LLM_FALLBACK",
                "confidence_score": 0.5
            }
            
        return {
            "decision": llm_out.get("decision", "FLAG"),
            "score": float(llm_out.get("score", 50.0)),
            "reason": llm_out.get("reason", "Evaluated by LLM Fallback"),
            "policy_id": "LLM_FALLBACK",
            "confidence_score": float(llm_out.get("confidence_score", 0.8))
        }

    def scan_output(self, llm_response: str, user_prompt: str = "", source: str = "internal_api", dest: str = "external_vendor") -> Dict[str, Any]:
        """The main entrypoint that merges all inputs and evaluates policies."""
        # 1. Run parallel components
        # These checks are independent; keep the slower classifier off the scanner's critical path.
        with ThreadPoolExecutor(max_workers=3) as executor:
            pattern_future = executor.submit(self.pattern_scanner, llm_response)
            context_future = executor.submit(self.context_classifier, user_prompt, llm_response)
            flow_future = executor.submit(self.info_flow_analyzer, source, dest)
            pattern_results = pattern_future.result()
            context_results = context_future.result()
            flow_results = flow_future.result()
        
        # 2. Merger
        merged_context = {
            "pattern_matches": pattern_results["matches"],
            "pattern_findings": pattern_results["findings"],
            "pattern_confidence": pattern_results["confidence_score"],
            "sensitivity": context_results["sensitivity"],
            "sensitivity_confidence": context_results.get("confidence", 0.5),
            "flow": flow_results,
            "trust_boundary_crossed": flow_results["trust_boundary_crossed"],
            "flow_status": flow_results["status"]
        }
        
        # 3. Policy Engine Evaluation
        decision_result = self.policy_engine(merged_context)
        
        # 4. Fallback LLM Evaluation if No Match
        if decision_result.get("decision") == "NO_MATCH":
            decision_result = self.policy_rag(merged_context)
            
        # Format the output
        final_score = decision_result.get("score", 0.0)
        if decision_result.get("decision") == "BLOCK":
            final_score = max(final_score, 80.0)
            
        if final_score >= 80.0:
            status = "FAIL"
        elif final_score >= 40.0:
            status = "FLAG"
        else:
            status = "PASS"
            
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
        }