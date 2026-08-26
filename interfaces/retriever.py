from typing import List, Protocol
from schemas.core import EvidenceChunk

class IRetriever(Protocol):
    """
    Abstract interface for evidence retrieval.
    This allows swapping between mock RAG, real Vector DBs, or Web Search APIs
    without changing the evaluation logic.
    """
    
    def retrieve(self, query: str, top_k: int = 3) -> List[EvidenceChunk]:
        """
        Retrieve relevant evidence chunks for a given query (claim).
        
        Args:
            query: The search query (usually a factual claim).
            top_k: The number of chunks to return.
            
        Returns:
            A list of EvidenceChunk objects containing the text, score, and metadata.
        """
        ...

