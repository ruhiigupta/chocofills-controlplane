import os
import time
from dotenv import load_dotenv
load_dotenv()
from schemas.core import EvidenceChunk, EvaluationTrace
from agents.evaluators import (
    ClaimExtractionAgent, 
    FactualityAgent, 
    RelevanceAgent, 
    IndependentCriticAgent
)
from agents.policy_engine import PolicyEngine

# Initialize the agents directly
claim_extractor = ClaimExtractionAgent()
factuality_agent = FactualityAgent()
relevance_agent = RelevanceAgent()
critic_agent = IndependentCriticAgent()
policy_engine = PolicyEngine()

# Custom retriever to control evidence per test case
class TestCaseRetriever:
    def __init__(self):
        self.mock_db = {}
    
    def set_mock_db(self, db):
        self.mock_db = db

    def retrieve(self, query: str, top_k: int = 3):
        for key, evidence_text in self.mock_db.items():
            if key.lower() in query.lower():
                return [
                    EvidenceChunk(
                        id=f"doc_{time.time_ns()}",
                        text=evidence_text,
                        relevance_score=0.95
                    )
                ]
        return [
            EvidenceChunk(
                id=f"doc_{time.time_ns()}",
                text="The database contains no relevant information regarding this topic.",
                relevance_score=0.1
            )
        ]

custom_retriever = TestCaseRetriever()

test_cases = [
    {
        "name": "1. Clearly factual + relevant response",
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "db": {"paris": "Paris is the capital and most populous city of France.", "france": "Paris is the capital and most populous city of France."}
    },
    {
        "name": "2. Factual response containing one false claim",
        "prompt": "Who wrote Hamlet and when was it published?",
        "response": "Hamlet was written by William Shakespeare and published in 1995.",
        "db": {
            "shakespeare": "Hamlet is a tragedy written by William Shakespeare.", 
            "1995": "Hamlet was published around 1603, not 1995.",
            "published": "Hamlet was published around 1603."
        }
    },
    {
        "name": "3. Relevant but unsupported/hallucinated claim",
        "prompt": "Does the new Acme X100 phone have a headphone jack?",
        "response": "Yes, the Acme X100 phone comes with a 3.5mm headphone jack.",
        "db": {"acme": "The Acme X100 phone features a bezel-less display and USB-C charging. It removed all legacy ports including the headphone jack."}
    },
    {
        "name": "4. Factually correct but irrelevant response",
        "prompt": "How do I reset my password?",
        "response": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "db": {"eiffel": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."}
    },
    {
        "name": "5. Poor/insufficient RAG evidence",
        "prompt": "What are the side effects of XYZ medication?",
        "response": "XYZ medication can cause mild drowsiness and headaches.",
        "db": {} 
    }
]

print("Starting Sequential Evaluation Pipeline Tests...\n")
if not os.getenv("OPENROUTER_API_KEY"):
    print("WARNING: OPENROUTER_API_KEY is missing.\n")

for tc in test_cases:
    print(f"{'='*80}")
    print(f"TEST: {tc['name']}")
    print(f"Prompt: {tc['prompt']}")
    print(f"Response: {tc['response']}\n")
    
    custom_retriever.set_mock_db(tc['db'])
    
    # Manually construct the EvaluationTrace
    trace = EvaluationTrace(
        user_prompt=tc["prompt"],
        llm_response=tc["response"]
    )
    
    start_time = time.time()
    try:
        # 1. Extract Claims
        trace.claims = claim_extractor.extract(trace.llm_response)
        
        # 2. Retrieve Evidence
        for claim in trace.claims:
            trace.retrieved_evidence[claim.id] = custom_retriever.retrieve(claim.text)
            
        # 3. Evaluate Factuality & Relevance
        for claim in trace.claims:
            evidence = trace.retrieved_evidence.get(claim.id, [])
            fact_metric = factuality_agent.evaluate(claim, evidence)
            rel_metric = relevance_agent.evaluate(trace.user_prompt, claim)
            trace.factuality_metrics.append(fact_metric)
            trace.relevance_metrics.append(rel_metric)
            
        # 4. Independent Critic Verification
        fact_map = {m.claim_id: m for m in trace.factuality_metrics}
        rel_map = {m.claim_id: m for m in trace.relevance_metrics}
        
        for claim in trace.claims:
            evidence = trace.retrieved_evidence.get(claim.id, [])
            fact_metric = fact_map.get(claim.id)
            rel_metric = rel_map.get(claim.id)
            if fact_metric and rel_metric:
                critic_metric = critic_agent.evaluate(trace.user_prompt, claim, evidence, fact_metric, rel_metric)
                trace.critic_metrics.append(critic_metric)
                
        # 5. Policy Engine Decision
        trace.policy_decision = policy_engine.evaluate_trace(trace)
        
        latency = time.time() - start_time
        
        # Print Results
        print("--- EXTRACTED CLAIMS ---")
        if trace.claims:
            for c in trace.claims:
                print(f" [{c.id}] {c.text}")
        else:
            print(" No claims extracted.")
            
        print("\n--- RETRIEVED EVIDENCE ---")
        if trace.retrieved_evidence:
            for claim_id, chunks in trace.retrieved_evidence.items():
                print(f" Claim {claim_id}: {[chunk.text for chunk in chunks]}")
        else:
            print(" No evidence retrieved.")
            
        print("\n--- FACTUALITY SCORES ---")
        for m in trace.factuality_metrics:
            print(f" [{m.claim_id}] Entailment: {m.entailment_score} | Coverage: {m.evidence_coverage} | Reason: {m.reasoning}")
            
        print("\n--- RELEVANCE SCORES ---")
        for m in trace.relevance_metrics:
            print(f" [{m.claim_id}] Relevance: {m.relevance_score} | Reason: {m.reasoning}")
            
        print("\n--- CRITIC SCORES ---")
        for m in trace.critic_metrics:
            print(f" [{m.claim_id}] Agrees Fact: {m.agrees_with_factuality} | Agrees Rel: {m.agrees_with_relevance} | Contradict: {m.contradiction_flag}\n   Reason: {m.reasoning}")
            
        print("\n--- POLICY DECISION ---")
        if trace.policy_decision:
            print(f" Action: {trace.policy_decision.final_action}")
            print(f" Reason: {trace.policy_decision.reasoning}")
            print(f" Flagged Claims: {trace.policy_decision.flagged_claims}")
            
        print(f"\nExecution Time: {latency:.2f}s")
        
    except Exception as e:
        print(f"\n[!] CRITICAL ERROR during execution:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    print(f"{'='*80}\n")

