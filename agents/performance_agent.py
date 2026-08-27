import json
import os
import re
from typing import List, Dict, Any

from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

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
        # Use the configured OpenRouter provider for structured evaluator outputs.
        self.client = instructor.from_openai(
            OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", "dummy-key"),
            )
            , mode=instructor.Mode.JSON
        )
        self.eval_model = "google/gemini-3.5-flash-lite"
        self.retriever = None
        self.web_retriever = None
        self.performance_rag_relevance_threshold = float(
            os.getenv("PERFORMANCE_RAG_RELEVANCE_THRESHOLD", "0.2")
        )
        self._initialize_local_retriever()
        self._initialize_web_retriever()

    def _initialize_local_retriever(self):
        """Initialize a local source-document retriever for performance evidence."""
        source_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data",
            "performance_sources",
        )

        self.retriever = None

        try:
            if not os.path.isdir(source_dir):
                print("[PerformanceAgent] Performance source directory not found; RAG disabled.")
                return None

            docs = []
            for filename in sorted(os.listdir(source_dir)):
                file_path = os.path.join(source_dir, filename)
                if not os.path.isfile(file_path):
                    continue
                try:
                    loader = TextLoader(file_path)
                    docs.extend(loader.load())
                except Exception as exc:
                    print(f"[PerformanceAgent] Failed to load document {file_path}: {exc}")

            if not docs:
                print("[PerformanceAgent] No performance source documents available; RAG disabled.")
                return None

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(splits, embeddings)
            self.vectorstore = vectorstore
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            print(f"[PerformanceAgent] Initialized performance RAG with {len(splits)} chunks.")
            return self.retriever
        except Exception as exc:
            print(f"[PerformanceAgent] Failed to initialize performance RAG: {exc}")
            self.retriever = None
            return None

    def _initialize_web_retriever(self):
        """Initialize a Tavily web retriever only when an API key is configured."""
        self.web_retriever = None

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            print("[PerformanceAgent] TAVILY_API_KEY not configured; web retrieval disabled.")
            return None

        try:
            from langchain_tavily import TavilySearch

            self.web_retriever = TavilySearch(
                tavily_api_key=api_key,
                max_results=5,
                include_answer=True,
                include_raw_content=True,
            )
            print("[PerformanceAgent] Initialized Tavily web retriever.")
            return self.web_retriever
        except Exception as exc:
            print(f"[PerformanceAgent] Failed to initialize Tavily web retriever: {exc}")
            self.web_retriever = None
            return None

    def extract_claims(self, text: str) -> List[str]:
        """Decomposes the model response into verifiable claims."""
        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=ClaimList,
                max_tokens=2000,
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

    def _coerce_evidence_text(self, document: Any) -> str:
        """Normalize document-like retrieval results into plain evidence text."""
        if document is None:
            return ""

        if isinstance(document, str):
            stripped = document.strip()
            return stripped if stripped else ""

        if isinstance(document, dict):
            for key in ("page_content", "content", "text", "snippet", "raw_content"):
                value = document.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        page_content = getattr(document, "page_content", None)
        if isinstance(page_content, str) and page_content.strip():
            return page_content.strip()

        text = getattr(document, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        return ""

    def _keyword_overlap_score(self, claim: str, text: str) -> float:
        """Lightweight fallback relevance heuristic when vector relevance scores are unavailable."""
        claim_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", claim or "")
            if len(token) > 2
        }
        content_tokens = {
            token.lower()
            for token in re.findall(r"[A-Za-z0-9]+", text or "")
            if len(token) > 2
        }

        if not claim_tokens:
            return 0.0
        if not content_tokens:
            return 0.0

        overlap = claim_tokens & content_tokens
        if not overlap:
            return 0.0

        union = claim_tokens | content_tokens
        return len(overlap) / len(union) if union else 0.0

    def _collect_local_evidence(self, claim: str):
        """Query the local source-document retriever and return (documents, is_sufficient)."""
        if not getattr(self, "retriever", None):
            return [], False

        scored_relevant_docs = []
        try:
            if hasattr(self, "vectorstore") and hasattr(self.vectorstore, "similarity_search_with_relevance_scores"):
                scored = self.vectorstore.similarity_search_with_relevance_scores(
                    claim,
                    k=3,
                )
                if scored:
                    scored_relevant_docs = [
                        doc for doc, score in scored
                        if isinstance(score, (int, float)) and float(score) >= self.performance_rag_relevance_threshold
                    ]
                    if scored_relevant_docs:
                        return scored_relevant_docs, True
        except Exception as exc:
            print(f"[PerformanceAgent] Local relevance scoring unavailable or failed: {exc}")

        try:
            docs = self.retriever.invoke(claim)
        except Exception as exc:
            print(f"[Performance RAG Retrieval Error] Local retrieval failed: {exc}")
            return [], False

        if not docs:
            return [], False

        relevant_docs = []
        for doc in docs:
            content = self._coerce_evidence_text(doc)
            if not content:
                continue
            score = self._keyword_overlap_score(claim, content)
            if score > 0.0:
                relevant_docs.append(doc)

        if relevant_docs:
            return relevant_docs, True

        return docs, False

    def _format_local_evidence(self, docs: List[Any]) -> str:
        """Format source-document evidence with source labels."""
        if not docs:
            return "NO_EVIDENCE"

        evidence_parts = []
        for doc in docs:
            content = self._coerce_evidence_text(doc)
            if not content:
                continue

            source_name = "unknown"
            metadata = getattr(doc, "metadata", None) or {}
            source_value = metadata.get("source") or metadata.get("file_path")
            if source_value:
                source_name = os.path.basename(str(source_value))
            elif isinstance(doc, dict):
                source_value = (doc.get("metadata", {}) or {}).get("source") or (doc.get("metadata", {}) or {}).get("file_path")
                if source_value:
                    source_name = os.path.basename(str(source_value))

            evidence_parts.append(f"[SOURCE: performance_sources/{source_name}]\n{content.strip()}")

        if not evidence_parts:
            return "NO_EVIDENCE"

        return "\n\n==========\n\n".join(evidence_parts)

    def _fetch_web_evidence(self, claim: str) -> str:
        """Return evidence from Tavily only when there is a configured web retriever."""
        if not getattr(self, "web_retriever", None):
            return "NO_EVIDENCE"

        try:
            results = self.web_retriever.invoke(claim)
        except Exception as exc:
            print(f"[Performance RAG Retrieval Error] Web retrieval failed: {exc}")
            return "NO_EVIDENCE"

        if results is None:
            return "NO_EVIDENCE"

        payload = results
        if isinstance(results, dict):
            if "results" in results and isinstance(results["results"], list):
                payload = results["results"]
            elif "data" in results and isinstance(results["data"], list):
                payload = results["data"]
            elif "answer" in results and not results.get("results"):
                payload = [{"title": "Web answer", "url": "", "content": str(results.get("answer", ""))}]
        elif not isinstance(results, list):
            payload = [results]

        if not isinstance(payload, list):
            return "NO_EVIDENCE"

        evidence_parts = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            title = item.get("title") or item.get("name") or "Web source"
            url = item.get("url") or item.get("link") or ""
            content = (
                item.get("content")
                or item.get("snippet")
                or item.get("raw_content")
                or item.get("summary")
                or ""
            )

            if not content and url:
                continue
            if not content:
                continue

            formatted = f"[WEB SOURCE: {title}]"
            if url:
                formatted += f"\nURL: {url}"
            formatted += f"\nContent:\n{content.strip()}"
            evidence_parts.append(formatted)

        if not evidence_parts:
            return "NO_EVIDENCE"

        return "\n\n==========\n\n".join(evidence_parts)

    def retrieve_evidence(self, claim: str) -> str:
        """Retrieve evidence in priority order: local source documents first, then web fallback."""
        if not claim or not claim.strip():
            return "NO_EVIDENCE"

        local_docs, is_local_sufficient = self._collect_local_evidence(claim)
        if is_local_sufficient:
            return self._format_local_evidence(local_docs)

        web_evidence = self._fetch_web_evidence(claim)
        if web_evidence != "NO_EVIDENCE":
            return web_evidence

        return "NO_EVIDENCE"

    def mock_external_rag_retrieval(self, claim: str) -> str:
        """Backward-compatible wrapper for legacy callers that still use the old name."""
        return self.retrieve_evidence(claim)

    def evaluate_factuality(self, claim: str, evidence: str) -> Dict[str, Any]:
        """Checks if a claim is supported by the retrieved evidence."""
        if evidence == "NO_EVIDENCE":
            return {
                "is_supported": False,
                "confidence": 0.0,
                "reasoning": "Factuality could not be established because no relevant evidence was available."
            }

        try:
            result = self.client.chat.completions.create(
                model=self.eval_model,
                response_model=FactualityEvaluation,
                max_tokens=2000,
                messages=[
                    {"role": "system", "content":"""
You are a factuality evaluator in an enterprise AI governance system.

Your task is to determine whether a claim is supported by the evidence provided.

The evidence may come from:
1. Internal source documents
2. External web sources

Rules:
1. Evaluate the claim only against the provided evidence.
2. Do not use outside knowledge to fill gaps.
3. Do not assume a claim is true merely because it sounds plausible.
4. Mark supported=true only when the evidence sufficiently establishes the claim.
5. Mark supported=false when the evidence contradicts the claim.
6. Mark supported=false when the evidence is insufficient to establish the claim.
7. Preserve uncertainty when evidence is incomplete, ambiguous, or conflicting.
8. Do not treat keyword overlap as factual support.
9. Do not treat the existence of a retrieved document as proof that the claim is true.
10. Evaluate the meaning of the evidence in relation to the claim.
11. If the evidence is NO_EVIDENCE, the claim cannot be considered supported.
12. Give a confidence score between 0.0 and 1.0.
13. Provide brief reasoning based only on the supplied evidence.
14. Do not fabricate citations, sources, or facts.

Return only the structured output required by the response schema.
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
                max_tokens=2000,
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
            # 1. Evidence Retrieval (real retrieval only; no fabrication)
            evidence = self.retrieve_evidence(claim)

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

