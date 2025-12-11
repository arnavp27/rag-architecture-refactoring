"""
HybridStrategy - Combines vector + keyword search with RRF fusion (implements RetrievalStrategy)

This strategy combines both vector similarity and keyword search results using
Reciprocal Rank Fusion (RRF) to produce a balanced ranking.

Ported from: RAG_v2/retrieval/retriever.py - _reciprocal_rank_fusion method

Design Pattern: Strategy Pattern (concrete strategy)
SOLID Principles:
- Single Responsibility: Only handles hybrid retrieval with RRF fusion
- Open/Closed: Can be extended without modification
- Dependency Inversion: Depends on VectorStore interface, not concrete implementation
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from core.interfaces.retrieval_strategy import RetrievalStrategy
from core.interfaces.vector_store import VectorStore


class HybridStrategy(RetrievalStrategy):
    """
    Hybrid retrieval strategy combining vector and keyword search with RRF fusion.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results from:
    1. Vector similarity search (semantic matching)
    2. BM25 keyword search (lexical matching)
    
    Best for queries that benefit from both semantic understanding and exact term matching.
    
    Example use cases:
    - Complex queries: "economic policies affecting small businesses"
    - Mixed queries: "Angela Merkel's stance on climate change"
    - General questions: "What is the government doing about unemployment?"
    
    RRF Formula: score = sum(1 / (k + rank)) for each retrieval method
    where k=60 is a constant that reduces the impact of rank differences
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        rrf_k: int = 60,
        vector_weight: float = 1.0,
        keyword_weight: float = 1.0
    ):
        """
        Initialize the hybrid retrieval strategy.
        
        Args:
            vector_store: VectorStore interface implementation
            rrf_k: RRF constant (default 60, standard value from literature)
            vector_weight: Weight for vector results in RRF (default 1.0)
            keyword_weight: Weight for keyword results in RRF (default 1.0)
        """
        if not isinstance(vector_store, VectorStore):
            raise TypeError(
                f"vector_store must implement VectorStore interface, "
                f"got {type(vector_store)}"
            )
        
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be at least 1, got {rrf_k}")
        
        self._vector_store = vector_store
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._logger = logging.getLogger(__name__)
        
        self._logger.debug(
            f"HybridStrategy initialized with rrf_k={rrf_k}, "
            f"vector_weight={vector_weight}, keyword_weight={keyword_weight}"
        )
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search with RRF fusion.
        
        Steps:
        1. Perform vector search
        2. Perform keyword search
        3. Fuse results using Reciprocal Rank Fusion
        4. Return top_k results sorted by RRF score
        
        Args:
            query_vector: Query embedding vector (for vector search)
            query_text: Original query text (for keyword search)
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents sorted by RRF score
            
        Raises:
            ValueError: If query_vector or query_text is invalid
            RuntimeError: If retrieval fails
        """
        # Validation
        if query_vector is None:
            raise ValueError("query_vector cannot be None for HybridStrategy")
        
        if not isinstance(query_vector, np.ndarray):
            raise ValueError(
                f"query_vector must be numpy.ndarray, got {type(query_vector)}"
            )
        
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty for HybridStrategy")
        
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        # Log retrieval attempt
        self._logger.info(
            f"HybridStrategy: Retrieving {top_k} documents for query: '{query_text[:50]}...'"
        )
        if filters:
            self._logger.debug(f"Applying filters: {filters}")
        
        try:
            # Retrieve more candidates initially to ensure good fusion
            # (retrieve 2x top_k from each method, then fuse and return top_k)
            initial_k = max(top_k * 2, 20)
            
            # Step 1: Vector search
            self._logger.debug(f"Performing vector search (top_k={initial_k})")
            vector_results = self._vector_store.vector_search(
                query_vector=query_vector,
                top_k=initial_k,
                filters=filters
            )
            
            # Step 2: Keyword search
            self._logger.debug(f"Performing keyword search (top_k={initial_k})")
            keyword_results = self._vector_store.keyword_search(
                query_text=query_text,
                top_k=initial_k,
                filters=filters
            )
            
            # Step 3: Reciprocal Rank Fusion
            self._logger.debug(
                f"Fusing {len(vector_results)} vector results and "
                f"{len(keyword_results)} keyword results"
            )
            fused_results = self._reciprocal_rank_fusion(
                vector_results=vector_results,
                keyword_results=keyword_results
            )
            
            # Step 4: Return top_k results
            final_results = fused_results[:top_k]
            
            self._logger.info(
                f"HybridStrategy: Retrieved {len(final_results)} documents "
                f"(fused from {len(vector_results)} vector + {len(keyword_results)} keyword)"
            )
            
            return final_results
            
        except Exception as e:
            self._logger.error(f"Hybrid search failed: {e}")
            raise RuntimeError(f"HybridStrategy retrieval failed: {e}") from e
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword results using Reciprocal Rank Fusion.
        
        This is the core RRF algorithm ported from RAG_v2/retrieval/retriever.py
        
        RRF Formula: score(doc) = sum(weight / (k + rank)) across all retrieval methods
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            
        Returns:
            Fused and sorted results with RRF scores
        """
        try:
            # Dictionary to accumulate RRF scores
            # Key: document identifier (embedding_index or id)
            # Value: {"result": document_dict, "rrf_score": float}
            rrf_scores: Dict[str, Dict[str, Any]] = {}
            
            # Score vector results
            for rank, result in enumerate(vector_results, start=1):
                # Use embedding_index as key (or fall back to id)
                key = result.get("embedding_index") or result.get("id", f"vec_{rank}")
                
                # Initialize entry if not exists
                if key not in rrf_scores:
                    rrf_scores[key] = {
                        "result": result.copy(),
                        "rrf_score": 0.0
                    }
                
                # Add RRF score contribution from vector search
                rrf_contribution = self._vector_weight / (self._rrf_k + rank)
                rrf_scores[key]["rrf_score"] += rrf_contribution
                
                # Store original vector score for debugging
                if "vector_score" not in rrf_scores[key]["result"]:
                    rrf_scores[key]["result"]["vector_score"] = result.get("score", 0.0)
            
            # Score keyword results
            for rank, result in enumerate(keyword_results, start=1):
                key = result.get("embedding_index") or result.get("id", f"kw_{rank}")
                
                # Initialize entry if not exists
                if key not in rrf_scores:
                    rrf_scores[key] = {
                        "result": result.copy(),
                        "rrf_score": 0.0
                    }
                
                # Add RRF score contribution from keyword search
                rrf_contribution = self._keyword_weight / (self._rrf_k + rank)
                rrf_scores[key]["rrf_score"] += rrf_contribution
                
                # Store original keyword score for debugging
                if "bm25_score" not in rrf_scores[key]["result"]:
                    rrf_scores[key]["result"]["bm25_score"] = result.get("score", 0.0)
                
                # Merge vector score if we have it from both searches
                if "vector_score" in result and "vector_score" not in rrf_scores[key]["result"]:
                    rrf_scores[key]["result"]["vector_score"] = result.get("vector_score", 0.0)
            
            # Build final result list with RRF scores
            fused_results = []
            for data in rrf_scores.values():
                result = data["result"]
                result["rrf_score"] = data["rrf_score"]
                result["score"] = data["rrf_score"]  # Use RRF score as primary score
                fused_results.append(result)
            
            # Sort by RRF score (descending - highest first)
            fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
            
            self._logger.debug(
                f"RRF fusion produced {len(fused_results)} unique candidates"
            )
            
            return fused_results
            
        except Exception as e:
            self._logger.error(f"RRF fusion failed: {e}")
            # Fallback: return vector results if fusion fails
            self._logger.warning("Falling back to vector results only")
            return vector_results
    
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        return "Hybrid"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            "name": self.get_strategy_name(),
            "description": "Hybrid search combining vector and keyword with RRF fusion",
            "search_type": "hybrid",
            "best_for": [
                "Complex multi-faceted queries",
                "Queries with both concepts and specific terms",
                "General-purpose retrieval"
            ],
            "parameters": {
                "uses_vector": True,
                "uses_keywords": True,
                "fusion_method": "Reciprocal Rank Fusion (RRF)",
                "rrf_k": self._rrf_k,
                "vector_weight": self._vector_weight,
                "keyword_weight": self._keyword_weight
            }
        }