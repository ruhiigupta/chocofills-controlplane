"""
ControlPlane.ai - Tokenizer Service
Deterministic token counting for User Prompts, Model Responses, and Source Documents.
Supports 'tiktoken' BPE when installed, with deterministic fallback tokenizer.
"""

import re
from typing import List, Dict, Any, Optional

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False


class TokenizerService:
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.encoding_name = encoding_name
        self._encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self._encoder = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._encoder = None

    def count_tokens(self, text: Optional[str]) -> int:
        """Deterministically count tokens in a given text string."""
        if not text:
            return 0
        
        if self._encoder is not None:
            return len(self._encoder.encode(text, disallowed_special=()))
        
        # Deterministic Calibrated Heuristic Tokenizer
        # Regex splits on words, numbers, punctuation, and whitespace chunks
        tokens = re.findall(r"\w+|[^\w\s]|\s+", text, re.UNICODE)
        token_count = 0
        for token in tokens:
            if token.isspace():
                # Whitespace tokens: 1 token per 4 spaces or single newline
                token_count += max(1, len(token) // 4)
            elif len(token) > 4:
                # Long words / code identifiers split into sub-tokens (~3.8 chars per sub-token)
                token_count += max(1, int(round(len(token) / 3.8)))
            else:
                token_count += 1
        return max(1, token_count) if text.strip() else 0

    def count_documents_tokens(self, source_documents: Optional[List[Dict[str, Any]]]) -> int:
        """Count total tokens across all source documents in ControlPlaneState."""
        if not source_documents:
            return 0
        
        total_doc_tokens = 0
        for doc in source_documents:
            content = doc.get("content", "")
            total_doc_tokens += self.count_tokens(content)
        return total_doc_tokens


# Global default tokenizer instance
default_tokenizer = TokenizerService()
