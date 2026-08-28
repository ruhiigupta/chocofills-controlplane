import json
import os
import re
import traceback
from typing import List, Dict, Any

from matplotlib import text

from pydantic import BaseModel, Field
import instructor
from openai import OpenAI
from google import genai
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.client = genai.Client(api_key=api_key)
        self.eval_model = "gemini-3.5-flash-lite"
        
        self.web_retriever = None
        self.performance_rag_keyword_threshold = float(
            os.getenv("PERFORMANCE_RAG_KEYWORD_THRESHOLD", "0.05")
        )
        self.performance_rag_relevance_threshold = float(
            os.getenv("PERFORMANCE_RAG_RELEVANCE_THRESHOLD", "0.3")
        )
        # self._initialize_local_retriever()
        self._initialize_web_retriever()


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
                search_depth="advanced",
                topic="general",
                include_answer=False,
                include_raw_content=True,
            )

            print("[PerformanceAgent] Initialized Tavily web retriever.")
            return self.web_retriever

        except Exception as exc:
            print(
                f"[PerformanceAgent] Failed to initialize Tavily web retriever: {exc}"
            )
            self.web_retriever = None
            return None

    def extract_claims(self, text: str) -> List[str]:
        """Decomposes the model response into verifiable claims."""
        try:
            prompt = """
            Task: Break the given paragraph/sentence into a list of atomic,
            independently verifiable factual claims.

            Rules:
            1. Express a factual assertion.
            2. Be independently verifiable.
            3. Contain enough context to stand alone.
            4. Preserve the original meaning.
            5. Be as atomic as possible.
            6. Do not introduce information that is not present in the original response.

            Return only the extracted claims. If there are no verifiable factual
            claims, return an empty list.

            Text:
            """ + text

            result = self.client.models.generate_content(
                model=self.eval_model,
                contents=prompt,
                config={
                    "max_output_tokens": 2000,
                    "response_mime_type": "application/json",
                    "response_schema": ClaimList,
                },
            )

            claims = ClaimList.model_validate_json(result.text)

            return claims.claims

        except Exception as e:
            import traceback
            print("[Performance Agent] Claim extraction failed:")
            traceback.print_exc()
            return [text]

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

    def _collect_local_evidence(self, claim: str, uploaded_vectorstore=None):
        """Retrieve relevant evidence from the uploaded document."""

        if uploaded_vectorstore is not None:
            try:
                scored_docs = uploaded_vectorstore.similarity_search_with_relevance_scores(
                    claim,
                    k=3
                )

                relevant_docs = [
                    doc
                    for doc, score in scored_docs
                    if isinstance(score, (int, float))
                    and float(score) >= self.performance_rag_relevance_threshold
                ]

                if relevant_docs:
                    print(
                        f"[Performance RAG] Found {len(relevant_docs)} "
                        f"relevant chunks in uploaded source."
                    )
                    return relevant_docs, True

            except Exception as exc:
                print(
                    f"[Performance RAG] Uploaded document retrieval failed: {exc}"
                )
            return [], False
        # Uploaded document exists, but no relevant chunks were found.
         # Let retrieve_evidence() fall back to web.
        return [], False

    def _build_uploaded_vectorstore(
        self,
        source_documents: List[Dict[str, Any]] = None
    ):
        """Build a temporary vector store from documents uploaded in this request."""

        if not source_documents:
            return None

        uploaded_docs = []

        for source in source_documents:
            content = source.get("content", "")
            filename = source.get("filename", "uploaded_file")

            if content and content.strip():
                uploaded_docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": filename}
                    )
                )

        if not uploaded_docs:
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        splits = text_splitter.split_documents(uploaded_docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(
            splits,
            embeddings
        )

        print(
            f"[Performance RAG] Built uploaded-document vector DB "
            f"with {len(splits)} chunks."
        )

        return vectorstore  

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

            evidence_parts.append(f"[SOURCE: uploaded/{source_name}]\n{content.strip()}")

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

    def calculate_factuality_score(
        self,
        claim: str,
        uploaded_vectorstore
    ) -> tuple[float, List[Any]]:

        if not claim or not claim.strip() or uploaded_vectorstore is None:
            return 0.0, []

        try:
            scored_docs = uploaded_vectorstore.similarity_search_with_relevance_scores(
                claim,
                k=3
            )

            if not scored_docs:
                return 0.0, []

            scores = [float(score) for _, score in scored_docs]

            highest_score = max(scores)

            top_docs = [doc for doc, _ in scored_docs]

            return highest_score, top_docs

        except Exception:
            return 0.0, []


    def retrieve_evidence(self, claim: str, uploaded_vectorstore=None) -> str:
        """
        Retrieve factual evidence in priority order:

        1. Local/source documents
        2. Web search
        3. NO_EVIDENCE

        Retrieval only provides evidence. It does not decide factuality.
        """
        if not claim or not claim.strip():
            return "NO_EVIDENCE"

        # ---------------------------------------------------------
        # STEP 1: LOCAL SOURCE DOCUMENTS
        # ---------------------------------------------------------
        local_docs, local_available = self._collect_local_evidence(claim,uploaded_vectorstore)

        if local_available and local_docs:
            local_evidence = self._format_local_evidence(local_docs)

            if local_evidence != "NO_EVIDENCE":
                return local_evidence

        # ---------------------------------------------------------
        # STEP 2: WEB SEARCH FALLBACK
        # ---------------------------------------------------------
        web_evidence = self._fetch_web_evidence(claim)

        if web_evidence != "NO_EVIDENCE":
            return web_evidence

        # ---------------------------------------------------------
        # STEP 3: NO EVIDENCE
        # ---------------------------------------------------------
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
            prompt = f"""
        You are a factuality evaluator in an enterprise AI governance system.

        Your task is to determine whether a claim is supported by the evidence provided.

        Rules:
        1. Evaluate the claim only against the provided evidence.
        2. Do not use outside knowledge to fill gaps.
        3. Do not assume a claim is true merely because it sounds plausible.
        4. Mark supported=true only when the evidence sufficiently establishes the claim.
        5. Mark supported=false when the evidence contradicts the claim.
        6. Mark supported=false when the evidence is insufficient.
        7. Preserve uncertainty when evidence is incomplete, ambiguous, or conflicting.
        8. Do not treat keyword overlap as factual support.
        9. Evaluate the meaning of the evidence in relation to the claim.
        10. Give confidence between 0.0 and 1.0.
        11. Provide brief reasoning based only on the supplied evidence.
        12. Do not fabricate citations, sources, or facts.

        Claim:
        {claim}

        Evidence:
        {evidence}
        """

            result = self.client.models.generate_content(
                model=self.eval_model,
                contents=prompt,
                config={
                    "max_output_tokens": 2000,
                    "response_mime_type": "application/json",
                    "response_schema": FactualityEvaluation,
                },
            )

            evaluation = FactualityEvaluation.model_validate_json(result.text)

            return evaluation.model_dump()

        except Exception as e:
            print(f"[PerformanceAgent] Factuality evaluation failed: {e}")
            return {
                "is_supported": False,
                "confidence": 0.0,
                "reasoning": "Evaluation failed"
            }

    def calculate_relevance_score(
        self,
        prompt_embedding,
        claim: str
    ) -> float:
        """
        Calculate deterministic semantic similarity between
        the user's prompt and a claim.
        """

        if not claim:
            return 0.0

        #prompt_embedding = self.embeddings.embed_query(user_prompt)
        claim_embedding = self.embeddings.embed_query(claim)

        # cosine similarity
        dot_product = sum(
            a * b
            for a, b in zip(prompt_embedding, claim_embedding)
        )

        prompt_norm = sum(a * a for a in prompt_embedding) ** 0.5
        claim_norm = sum(b * b for b in claim_embedding) ** 0.5

        if prompt_norm == 0 or claim_norm == 0:
            return 0.0

        return dot_product / (prompt_norm * claim_norm)

    def evaluate_relevance(self, user_prompt: str, claim: str) -> Dict[str, Any]:
        """Checks if a claim is relevant to the original user prompt."""

        try:
            prompt = f"""
        Task: Determine whether the given claim is relevant to the user's prompt.

        Rules:
        1. Compare the claim directly with the user's prompt.
        2. Mark the claim as relevant if it directly answers the question or provides necessary information.
        3. Do not mark a claim as relevant merely because it shares a topic or keywords.
        4. Consider whether the claim contributes meaningfully to answering the user's request.
        5. Evaluate relevance independently of factual correctness.

        User Prompt:
        {user_prompt}

        Claim:
        {claim}
        """

            result = self.client.models.generate_content(
                model=self.eval_model,
                contents=prompt,
                config={
                    "max_output_tokens": 2000,
                    "response_mime_type": "application/json",
                    "response_schema": RelevanceEvaluation,
                },
            )

            evaluation = RelevanceEvaluation.model_validate_json(result.text)

            return evaluation.model_dump()

        except Exception as e:
            print(f"[PerformanceAgent] Relevance evaluation failed: {e}")
            return {
                "is_relevant": False,
                "reasoning": "Evaluation failed"
            }

    def run_evaluation(
        self,
        user_prompt: str,
        llm_response: str,
        source_documents: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Main orchestrator for the Performance Agent node."""

        claims = self.extract_claims(llm_response)
        prompt_embedding = self.embeddings.embed_query(user_prompt)
        factual_findings = []
        relevance_findings = []

        total_claims = len(claims)

        factuality_scores = []
        relevance_scores = []

        # Build uploaded-document vector DB ONCE per request
        uploaded_vectorstore = self._build_uploaded_vectorstore(source_documents)

        for claim in claims:

            # ---------------------------------------------------------
            # 1. FACTUALITY
            # ---------------------------------------------------------

            score, top_docs = self.calculate_factuality_score(
                claim,
                uploaded_vectorstore
            )

            if score >= 0.3:
                # Deterministic factuality score
                claim_factuality_score = score

                factual_findings.append({
                    "claim": claim,
                    "evidence_used": self._format_local_evidence(top_docs),
                    "is_supported": True,
                    "confidence": score,
                    "reasoning": "Claim has sufficiently relevant evidence in the uploaded source document."
                })

            else:
                # Weak retrieval → LLM fallback
                evidence = self.retrieve_evidence(
                    claim,
                    uploaded_vectorstore
                )

                fact_eval = self.evaluate_factuality(
                    claim,
                    evidence
                )

                # LLM gives only a binary fallback decision
                claim_factuality_score = (
                    1.0 if fact_eval.get("is_supported") else 0.0
                )

                factual_findings.append({
                    "claim": claim,
                    "evidence_used": evidence,
                    **fact_eval
                })

            factuality_scores.append(claim_factuality_score)

            # ---------------------------------------------------------
            # 2. RELEVANCE
            # ---------------------------------------------------------

            relevance_score = self.calculate_relevance_score(
                prompt_embedding,
                claim
            )

            if relevance_score >= 0.3:
                # Deterministic relevance
                claim_relevance_score = relevance_score

                relevance_findings.append({
                    "claim": claim,
                    "is_relevant": True,
                    "confidence": relevance_score,
                    "reasoning": "Claim has sufficiently high semantic similarity to the user's prompt."
                })

            else:
                # Weak semantic similarity → LLM fallback
                rel_eval = self.evaluate_relevance(
                    user_prompt,
                    claim
                )

                claim_relevance_score = (
                    1.0 if rel_eval.get("is_relevant") else 0.0
                )

                relevance_findings.append({
                    "claim": claim,
                    **rel_eval
                })

            relevance_scores.append(claim_relevance_score)

        # ---------------------------------------------------------
        # 3. FINAL SCORES
        # ---------------------------------------------------------

        factuality_score = (
            sum(factuality_scores) / total_claims * 100
            if total_claims > 0
            else 100.0
        )

        relevance_score = (
            sum(relevance_scores) / total_claims * 100
            if total_claims > 0
            else 100.0
        )

        performance_score = (
            0.65 * factuality_score
            + 0.35 * relevance_score
        )

        if performance_score >= 60:
            status = "PASS"
        elif performance_score >= 40:
            status = "NEEDS_REVIEW"
        elif performance_score >= 25:
            status = "FLAG"
        else:
            status = "BLOCK"

        return {
            "performance_score": performance_score,
            "factuality_score": factuality_score,
            "relevance_score": relevance_score,
            "performance_status": status,
            "factual_findings": factual_findings,
            "relevance_findings": relevance_findings
        }

