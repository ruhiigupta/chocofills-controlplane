import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
import os

# Define schemas for structured output
class ClaimList(BaseModel):
    claims: List[str] = Field(description="A list of atomic, verifiable claims extracted from the response.")

class FactualityEvaluation(BaseModel):
    is_supported: bool = Field(description="True if the claim is supported by the evidence, False otherwise.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of why the claim is or isn't supported.")

class RelevanceEvaluation(BaseModel):
    is_relevant: bool = Field(description="True if the claim answers or relates to the user prompt.")
    reasoning: str = Field(description="Brief explanation.")

class PerformanceAgent:
    def __init__(self):
        # We use instructor to patch the OpenAI client for structured JSON outputs
        # Ensure OPENAI_API_KEY is set in the environment
        self.client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key")))
        self.eval_model = "gpt-4o-mini" # Using a fast model for evaluation

    def extract_claims(self, text: str) -> List[str]:
        """Decomposes the model response into verifiable claims."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=ClaimList,
                messages=[
                    {"role": "system", "content":"""
                     Task: Break the given paragraph/sentence into a list of atomic, independently verifiable factual claims.

                     Context: The paragraph/sentence was generated in response to a user's prompt. The extracted claims will be independently checked for factual correctness and relevance. Therefore, each claim must be self-contained and preserve the meaning of the original response.

                     Rules:
                     1. Express a factual assertion.
                     2. Be independently verifiable.
                     3. Contain enough context to stand alone.
                     4. Preserve the original meaning.
                     5. Be as atomic as possible.
                     6. Do not introduce information that is not present in the original response.

                     Output: Return only the extracted claims. If there are no verifiable factual claims, return an empty list.
                     """ },
                    {"role": "user", "content": text}
                ]
            )
            return result.claims
        except Exception as e:
            print(f"[Performance Agent] Claim extraction failed: {e}")
            return [text] # Fallback to evaluating the whole text

    def mock_external_rag_retrieval(self, claim: str) -> str:
        """
        Placeholder for the external RAG system.
        In reality, this would hit the bi-encoder/BM25 vector search API.
        """
        # Simulating returning some evidence
        return f"Simulated evidence document for: {claim}"

    def evaluate_factuality(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Checks if a claim is supported by the retrieved evidence."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=FactualityEvaluation,
                messages=[
                    {"role": "system", "content":"""
                    Task: Determine whether the given claim is supported by the provided evidence.

                    Context: You are evaluating a claim using the evidence provided to you. The evidence is the basis for deciding whether the claim is factually supported or not.

                    Rules:
                    1. Compare the claim directly against the provided evidence.
                    2. Mark the claim as supported only when the evidence sufficiently establishes it.
                    3. Do not assume missing information is true.
                    4. Do not use outside knowledge to fill gaps.
                    5. Distinguish between evidence that supports, contradicts, or does not establish the claim.
                    6. Preserve uncertainty when the evidence is incomplete or ambiguous.
                    7. Assign a confidence score between 0.0 and 1.0.
                    8. Explain your decision using the provided evidence.

                    Output: Return whether the claim is supported (true/false), the confidence score, and a brief evidence-based reasoning.
                    """},
                    {"role": "user", "content": f"Claim: {claim}\n\nEvidence: {evidence}"}
                ]
            )
            return result.model_dump()
        except Exception as e:
            return {"is_supported": False, "confidence": 0.0, "reasoning": "Evaluation failed"}

    def evaluate_relevance(self, user_prompt: str, claim: str) -> Dict[str, Any]:
        """Checks if a claim is relevant to the original user prompt."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=RelevanceEvaluation,
                messages=[
                    {"role": "system", "content": """
                    Task: Determine whether the given claim is relevant to the user's prompt.

                    Context: You are evaluating whether a factual claim is relevant to the user's request and helps address what the user asked.

                    Rules:
                    1. Compare the claim directly with the user's prompt.
                    2. Mark the claim as relevant if it directly answers the question or provides necessary information to answer it.
                    3. Do not mark a claim as relevant merely because it shares a topic or keywords with the prompt.
                    4. Consider whether the claim contributes meaningfully to answering the user's request.
                    5. Evaluate relevance independently of factual correctness.

                    Output: Return whether the claim is relevant (true/false) and provide a brief explanation.
                    """},
                    {"role": "user", "content": f"User Prompt: {user_prompt}\n\nClaim: {claim}"}
                ]
            )
            return result.model_dump()
        except Exception as e:
            return {"is_relevant": False, "reasoning": "Evaluation failed"}

    def run_evaluation(self, user_prompt: str, llm_response: str) -> Dict[str, Any]:
        """Main orchestrator for the Performance Agent node."""
        
        claims = self.extract_claims(llm_response)
        
        factual_findings = []
        relevance_findings = []
        
        total_claims = len(claims)
        supported_claims = 0
        relevant_claims = 0
        
        for claim in claims:
            # 1. External RAG Retrieval
            evidence = self.mock_external_rag_retrieval(claim)
            
            # 2. Factuality Check
            fact_eval = self.evaluate_factuality(claim, evidence)
            factual_findings.append({
                "claim": claim,
                "evidence_used": evidence,
                **fact_eval
            })
            if fact_eval.get("is_supported"):
                supported_claims += 1
                
            # 3. Relevance Check
            rel_eval = self.evaluate_relevance(user_prompt, claim)
            relevance_findings.append({
                "claim": claim,
                **rel_eval
            })
            if rel_eval.get("is_relevant"):
                relevant_claims += 1

        # Calculate final aggregated scores (0-100 scale)
        factuality_score = (supported_claims / total_claims * 100) if total_claims > 0 else 100.0
        relevance_score = (relevant_claims / total_claims * 100) if total_claims > 0 else 100.0
        
        # Performance Score = weighted average of Factuality and Relevance
        performance_score = (0.7 * factuality_score) + (0.3 * relevance_score)
        
        status = "PASS" if performance_score >= 80 else "NEEDS_REVIEW"
        
        return {
            "performance_score": performance_score,
            "performance_status": status,
            "factual_findings": factual_findings,
            "relevance_findings": relevance_findings
        }

