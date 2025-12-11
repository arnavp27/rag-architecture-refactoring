"""
VectorOnlyStrategy - Pure vector similarity search (implements RetrievalStrategy)

This strategy performs retrieval using ONLY vector similarity search.
It delegates to the VectorStore interface's vector_search() method.

Design Pattern: Strategy Pattern (concrete strategy)
SOLID Principles:
- Single Responsibility: Only handles vector-based retrieval
- Open/Closed: Can be extended without modification
- Dependency Inversion: Depends on VectorStore interface, not concrete implementation
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from core.interfaces.retrieval_strategy import RetrievalStrategy
from core.interfaces.vector_store import VectorStore


class VectorOnlyStrategy(RetrievalStrategy):
    """
    Pure vector similarity search strategy.
    
    Uses only semantic similarity (cosine similarity) to retrieve documents.
    Best for queries where semantic meaning is more important than exact keywords.
    
    Example use cases:
    - Conceptual questions: "How does the economy affect daily life?"
    - Paraphrased queries: "Ways to improve fiscal policy"
    - Cross-lingual semantic search
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize the vector-only retrieval strategy.
        
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
        self._logger.debug("VectorOnlyStrategy initialized")
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using pure vector similarity search.
        
        Args:
            query_vector: Query embedding vector (1D numpy array)
            query_text: Original query text (for logging/metadata)
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents sorted by vector similarity score
            
        Raises:
            ValueError: If query_vector is None or invalid
            RuntimeError: If vector search fails
        """
        # Validation
        if query_vector is None:
            raise ValueError("query_vector cannot be None for VectorOnlyStrategy")
        
        if not isinstance(query_vector, np.ndarray):
            raise ValueError(
                f"query_vector must be numpy.ndarray, got {type(query_vector)}"
            )
        
        if query_vector.ndim != 1:
            raise ValueError(
                f"query_vector must be 1D array, got shape {query_vector.shape}"
            )
        
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        # Log retrieval attempt
        self._logger.info(
            f"VectorOnlyStrategy: Retrieving {top_k} documents for query: '{query_text[:50]}...'"
        )
        if filters:
            self._logger.debug(f"Applying filters: {filters}")
        
        try:
            # Delegate to vector store's vector_search method
            results = self._vector_store.vector_search(
                query_vector=query_vector,
                top_k=top_k,
                filters=filters
            )
            
            self._logger.info(
                f"VectorOnlyStrategy: Retrieved {len(results)} documents"
            )
            
            # Ensure all results have the 'score' field for consistency
            for result in results:
                if "vector_score" in result and "score" not in result:
                    result["score"] = result["vector_score"]
            
            return results
            
        except Exception as e:
            self._logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"VectorOnlyStrategy retrieval failed: {e}") from e
    
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "VectorOnly"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": self.get_strategy_name(),
            "description": "Pure vector similarity search using cosine distance",
            "search_type": "vector",
            "best_for": [
                "Semantic/conceptual queries",
                "Paraphrased questions",
                "Cross-lingual search"
            ],
            "parameters": {
                "uses_vector": True,
                "uses_keywords": False,
                "fusion_method": None
            }
        }