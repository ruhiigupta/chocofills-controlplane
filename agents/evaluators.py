import os
import time
from typing import List
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from schemas.core import Claim, FactualityMetric, RelevanceMetric, EvidenceChunk
from interfaces.retriever import IRetriever

class ClaimListWrapper(BaseModel):
    claims: List[Claim] = Field(description="A list of atomic, verifiable claims extracted from the response.")

class BaseEvaluator:
    def __init__(self, model: str = "google/gemini-3.5-flash-lite"):
        # Support OpenRouter if the key is provided
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key:
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key,
            )
            self.eval_model = "google/gemini-3.5-flash-lite" # Fast model on openrouter
        else:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "dummy-key"))
            self.eval_model = "gpt-4o-mini"
            
        # We use instructor to patch the OpenAI client for guaranteed structured JSON outputs
        self.client = instructor.from_openai(client, mode=instructor.Mode.JSON)

class ClaimExtractionAgent(BaseEvaluator):
    def extract(self, text: str) -> List[Claim]:
        """Decomposes the model response into verifiable, atomic claims."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=ClaimListWrapper,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are a precise claim extraction agent. Break down the text into atomic, standalone, verifiable factual claims. Give each claim a unique ID (e.g., 'claim_1')."},
                    {"role": "user", "content": text}
                ]
            )
            return result.claims
        except Exception as e:
            print(f"[ClaimExtractionAgent] Failed: {e}")
            return []

class FactualityAgent(BaseEvaluator):
    def evaluate(self, claim: Claim, evidence: List[EvidenceChunk]) -> FactualityMetric:
        """Evaluates a claim against retrieved evidence, returning granular numerical scores."""
        evidence_text = "\n\n".join([f"[{chunk.id}] (Score: {chunk.relevance_score}): {chunk.text}" for chunk in evidence])
        
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=FactualityMetric,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "You are a strict factuality judge. Evaluate the claim against the provided evidence. Return granular scores between 0.0 and 1.0 for entailment, evidence coverage, and evidence quality."},
                    {"role": "user", "content": f"Claim ID: {claim.id}\nClaim: {claim.text}\n\nEvidence:\n{evidence_text}"}
                ]
            )
            # Ensure the claim_id matches
            result.claim_id = claim.id
            return result
        except Exception as e:
            print(f"[FactualityAgent] Failed for {claim.id}: {e}")
            return FactualityMetric(claim_id=claim.id, entailment_score=0.0, evidence_coverage=0.0, evidence_quality=0.0, confidence=0.0, reasoning="Evaluation failed", evidence_used=[])

class RelevanceAgent(BaseEvaluator):
    def evaluate(self, user_prompt: str, claim: Claim) -> RelevanceMetric:
        """Evaluates how relevant a claim is to the original user prompt."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=RelevanceMetric,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content": "Evaluate how relevant and helpful this specific claim is in addressing the user's prompt. Output a score from 0.0 (irrelevant) to 1.0 (highly relevant)."},
                    {"role": "user", "content": f"User Prompt: {user_prompt}\n\nClaim: {claim.text}"}
                ]
            )
            result.claim_id = claim.id
            return result
        except Exception as e:
            print(f"[RelevanceAgent] Failed for {claim.id}: {e}")
            return RelevanceMetric(claim_id=claim.id, relevance_score=0.0, reasoning="Evaluation failed")

class IndependentCriticAgent(BaseEvaluator):
    def evaluate(
        self, 
        user_prompt: str, 
        claim: Claim, 
        evidence: List[EvidenceChunk], 
        fact_metric: FactualityMetric, 
        rel_metric: RelevanceMetric
    ) -> "CriticMetric":
        """
        Independently verifies the evaluations. Does NOT just copy scores.
        Looks at the raw data + the previous evaluators' reasoning to form an independent judgement.
        """
        from schemas.core import CriticMetric # Import here to avoid circular dependencies if any
        
        evidence_text = "\n\n".join([f"[{chunk.id}]: {chunk.text}" for chunk in evidence])
        
        prompt_context = f"""
You are an Independent Critic in a multi-agent system.
Your job is to independently verify the Factuality and Relevance of a claim, and then compare your findings against the original evaluators.

[ORIGINAL DATA]
User Prompt: {user_prompt}
Claim ({claim.id}): {claim.text}
Evidence:
{evidence_text}

[ORIGINAL EVALUATOR FINDINGS]
Factuality Evaluator -> Entailment Score: {fact_metric.entailment_score}, Reasoning: {fact_metric.reasoning}
Relevance Evaluator -> Relevance Score: {rel_metric.relevance_score}, Reasoning: {rel_metric.reasoning}

Determine if you agree or disagree, provide your OWN independent scores, and flag if there is a severe contradiction (hallucination).
"""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=CriticMetric,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt_context}
                ]
            )
            result.claim_id = claim.id
            return result
        except Exception as e:
            print(f"[IndependentCriticAgent] Failed for {claim.id}: {e}")
            from schemas.core import CriticMetric
            return CriticMetric(
                claim_id=claim.id, agrees_with_factuality=False, agrees_with_relevance=False,
                critic_factuality_score=0.0, critic_relevance_score=0.0,
                contradiction_flag=True, confidence=0.0, reasoning=f"Critic failed: {e}"
            )

# --- Mock Retriever Implementation ---

class MockVectorDBRetriever(IRetriever):
    def retrieve(self, query: str, top_k: int = 3) -> List[EvidenceChunk]:
        """A mock implementation of the IRetriever protocol."""
        # In a real app, this would hit Pinecone/Qdrant
        return [
            EvidenceChunk(
                id=f"doc_{time.time_ns()}",
                text=f"Simulated retrieved context regarding: {query}",
                relevance_score=0.85,
                source_metadata={"source": "mock_db", "url": "http://internal-wiki.local/mock"}
            )
        ]

