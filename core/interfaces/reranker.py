"""
Reranker ABC - Abstract interface for document reranking models

This interface defines the contract for all reranking model implementations
(CrossEncoder, ColBERT, LLM-based rerankers, etc.), enabling the system to be
model-agnostic.

Reranking is the second stage of retrieval where we use a more powerful model
to re-score the initially retrieved candidates for better relevance.

Design Pattern: Adapter Pattern (target interface for reranking models)
SOLID Principle: Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple


class Reranker(ABC):
    """
    Abstract interface for reranking retrieved documents.
    
    All reranker implementations MUST implement this interface to ensure
    they can be used interchangeably in the RAG pipeline.
    
    Reranking workflow:
    1. Retrieval stage: Get top-K candidates (e.g., K=100) using fast search
    2. Reranking stage: Use powerful model to re-score and select best (e.g., top-5)
    
    This two-stage approach balances efficiency and quality.
    """
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents based on relevance to the query.
        
        Takes initially retrieved candidates and re-scores them using a more
        sophisticated model (typically a cross-encoder) that can better assess
        query-document relevance.
        
        Args:
            query (str): The search query text
            candidates (List[Dict[str, Any]]): List of candidate documents to rerank
                Each document should have:
                - content (str): The document text
                - score (float): Initial retrieval score
                - metadata (Dict): Document metadata
                - id (str): Document identifier
            top_k (Optional[int]): Return only top K results after reranking
                If None, returns all candidates (reranked)
                
        Returns:
            List[Dict[str, Any]]: Reranked list of documents, sorted by relevance
                Each document has an additional field:
                - rerank_score (float): New relevance score from reranker
                Documents are sorted by rerank_score (descending)
                
        Raises:
            ValueError: If query is empty or candidates list is empty
            RuntimeError: If reranking fails
            
        Example:
            >>> reranker = CrossEncoderReranker()
            >>> 
            >>> # Initial retrieval returned 100 candidates
            >>> candidates = retriever.retrieve(query, top_k=100)
            >>> 
            >>> # Rerank and get top 5 most relevant
            >>> reranked = reranker.rerank(
            ...     query="What causes economic growth?",
            ...     candidates=candidates,
            ...     top_k=5
            ... )
            >>> 
            >>> for doc in reranked:
            ...     print(f"Rerank score: {doc['rerank_score']:.3f}")
            ...     print(f"Content: {doc['content'][:100]}...")
        """
        pass
    
    @abstractmethod
    def get_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Get relevance scores without modifying the candidate list.
        
        This method only computes scores and returns them as a separate list,
        without adding them to the documents or reordering them.
        
        Useful when you want to:
        - Inspect scores separately
        - Combine scores with other signals
        - Implement custom ranking logic
        
        Args:
            query (str): The search query text
            candidates (List[Dict[str, Any]]): List of candidate documents
            
        Returns:
            List[float]: Relevance scores in the same order as input candidates
                Scores are typically in range [0, 1] or [-1, 1]
                Higher scores = more relevant
                
        Raises:
            ValueError: If query is empty or candidates list is empty
            RuntimeError: If scoring fails
            
        Example:
            >>> scores = reranker.get_scores(query, candidates)
            >>> print(scores)
            [0.85, 0.72, 0.45, 0.12, 0.08, ...]
            >>> 
            >>> # Use scores however you want
            >>> for doc, score in zip(candidates, scores):
            ...     doc['custom_score'] = score * 0.7 + doc['original_score'] * 0.3
        """
        pass
    
    def rerank_with_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Rerank documents and return both reranked list and scores separately.
        
        Convenience method that combines rerank() and get_scores() functionality.
        
        Args:
            query (str): The search query text
            candidates (List[Dict[str, Any]]): List of candidate documents
            top_k (Optional[int]): Return only top K results
            
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: 
                - Reranked documents list
                - Corresponding rerank scores
                
        Note:
            This is NOT an abstract method - default implementation uses rerank().
        """
        reranked_docs = self.rerank(query, candidates, top_k)
        scores = [doc.get('rerank_score', 0.0) for doc in reranked_docs]
        return reranked_docs, scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranking model.
        
        Optional method to provide metadata about the model.
        
        Returns:
            Dict[str, Any]: Model metadata (name, type, parameters, etc.)
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return {
            "model_type": self.__class__.__name__
        }