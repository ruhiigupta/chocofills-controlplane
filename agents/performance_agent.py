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
                    {"role": "system", "content": "You are a claim extraction agent. Break down the following text into atomic, verifiable factual claims."},
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
                    {"role": "system", "content": "You are a fact-checking evaluator. Given a claim and evidence, determine if the evidence supports the claim."},
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
                    {"role": "system", "content": "Determine if the extracted claim is relevant and helpful in answering the user's prompt."},
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

