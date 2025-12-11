"""
KeywordOnlyStrategy - Pure BM25 keyword search (implements RetrievalStrategy)

This strategy performs retrieval using ONLY keyword/full-text search (BM25).
It delegates to the VectorStore interface's keyword_search() method.

Design Pattern: Strategy Pattern (concrete strategy)
SOLID Principles:
- Single Responsibility: Only handles keyword-based retrieval
- Open/Closed: Can be extended without modification
- Dependency Inversion: Depends on VectorStore interface, not concrete implementation
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from core.interfaces.retrieval_strategy import RetrievalStrategy
from core.interfaces.vector_store import VectorStore


class KeywordOnlyStrategy(RetrievalStrategy):
    """
    Pure BM25 keyword search strategy.
    
    Uses only lexical matching (exact keywords, term frequency) to retrieve documents.
    Best for queries with specific terms, names, or exact phrases.
    
    Example use cases:
    - Specific name queries: "statements by Angela Merkel"
    - Exact term queries: "GDP growth rate 2023"
    - Technical jargon: "quantitative easing policy"
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the keyword-only retrieval strategy.
        
        Args:
            vector_store: VectorStore interface implementation
                         (e.g., WeaviateAdapter)
        """
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must implement VectorStore interface, "
                f"got {type(vector_store)}"
            )
        
        self._vector_store = vector_store
        self._logger = logging.getLogger(__name__)
        self._logger.debug("KeywordOnlyStrategy initialized")
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using pure BM25 keyword search.
        
        Args:
            query_vector: Query embedding vector (not used, but required by interface)
            query_text: Original query text (used for keyword matching)
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents sorted by BM25 score
            
        Raises:
            ValueError: If query_text is empty or invalid
            RuntimeError: If keyword search fails
        """
        # Validation
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty for KeywordOnlyStrategy")
        
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        # Log retrieval attempt
        self._logger.info(
            f"KeywordOnlyStrategy: Retrieving {top_k} documents for query: '{query_text[:50]}...'"
        )
        if filters:
            self._logger.debug(f"Applying filters: {filters}")
        
        try:
            # Delegate to vector store's keyword_search method
            results = self._vector_store.keyword_search(
                query_text=query_text,
                top_k=top_k,
                filters=filters
            )
            
            self._logger.info(
                f"KeywordOnlyStrategy: Retrieved {len(results)} documents"
            )
            
            # Ensure all results have the 'score' field for consistency
            for result in results:
                if "bm25_score" in result and "score" not in result:
                    result["score"] = result["bm25_score"]
            
            return results
            
        except Exception as e:
            self._logger.error(f"Keyword search failed: {e}")
            raise RuntimeError(f"KeywordOnlyStrategy retrieval failed: {e}") from e
    
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "KeywordOnly"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": self.get_strategy_name(),
            "description": "Pure BM25 keyword search using lexical matching",
            "search_type": "keyword",
            "best_for": [
                "Queries with specific names/terms",
                "Exact phrase matching",
                "Technical terminology"
            ],
            "parameters": {
                "uses_vector": False,
                "uses_keywords": True,
                "fusion_method": None
            }
        }