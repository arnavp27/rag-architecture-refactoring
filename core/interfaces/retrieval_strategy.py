"""
RetrievalStrategy ABC - Abstract interface for retrieval algorithms

This is the Strategy Pattern interface that defines the contract for all
retrieval algorithm implementations (VectorOnly, Hybrid, KeywordOnly, etc.).

The Strategy Pattern allows algorithms to be swapped at runtime without
changing the code that uses them. This enables:
- Different retrieval approaches for different use cases
- A/B testing of retrieval methods
- Runtime optimization based on query characteristics

Design Pattern: Strategy Pattern (defines interchangeable algorithms)
SOLID Principles: 
- Open/Closed Principle (OCP): Can add new strategies without modifying existing code
- Dependency Inversion Principle (DIP): High-level code depends on this abstraction
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class RetrievalStrategy(ABC):
    """
    Abstract interface for retrieval algorithms.
    
    This is the Strategy pattern interface. Different retrieval algorithms
    (hybrid, vector-only, keyword-only, etc.) implement this interface,
    allowing them to be used interchangeably.
    
    Key characteristics of strategies:
    1. Stateless: No state between retrieve() calls
    2. Swappable: Can be changed at runtime
    3. Self-contained: Each strategy encapsulates its algorithm
    """
    
    @abstractmethod
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using this strategy's specific algorithm.
        
        Each strategy implements its own retrieval logic:
        - VectorOnlyStrategy: Pure vector similarity search
        - HybridStrategy: Combines vector + keyword with RRF fusion
        - KeywordOnlyStrategy: Pure BM25 keyword search
        
        All strategies accept the same parameters and return the same format,
        making them interchangeable from the caller's perspective.
        
        Args:
            query_vector (np.ndarray): Query embedding vector (1D array)
                Required even for keyword-only strategies (can be None)
            query_text (str): Original query text
                Required even for vector-only strategies (for metadata)
            top_k (int): Number of documents to retrieve (default: 5)
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
                Example: {
                    "theme": ["Economy", "Finance"],
                    "sentiment": "Positive",
                    "politician": "John Doe"
                }
                
        Returns:
            List[Dict[str, Any]]: List of retrieved documents, each containing:
                - content (str): The document text/statement
                - score (float): Relevance score (higher = more relevant)
                - metadata (Dict): Document metadata
                - id (str): Document identifier
                
            Documents are sorted by score (descending - best first)
                
        Raises:
            ValueError: If inputs are invalid (empty query, negative top_k, etc.)
            RuntimeError: If retrieval fails (database error, etc.)
            
        Example:
            >>> # Context class can switch strategies at runtime
            >>> retriever = Retriever(strategy=VectorOnlyStrategy(vector_store))
            >>> 
            >>> # Use the strategy
            >>> results = retriever.retrieve(
            ...     query_vector=query_vec,
            ...     query_text="economic growth",
            ...     top_k=5,
            ...     filters={"theme": ["Economy"]}
            ... )
            >>> 
            >>> # Switch to a different strategy
            >>> retriever.set_strategy(HybridStrategy(vector_store))
            >>> results = retriever.retrieve(...)  # Same interface!
        """
        pass
    
    def get_strategy_name(self) -> str:
        """
        Get the name of this retrieval strategy.
        
        Useful for logging, debugging, and analytics.
        
        Returns:
            str: Name of the strategy
            
        Note:
            This is NOT an abstract method - default implementation returns class name.
        """
        return self.__class__.__name__
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this retrieval strategy.
        
        Optional method to provide metadata about the strategy's
        configuration and characteristics.
        
        Returns:
            Dict[str, Any]: Strategy metadata
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return {
            "strategy_name": self.get_strategy_name(),
            "strategy_type": "retrieval"
        }
    
    def validate_inputs(
        self,
        query_vector: Optional[np.ndarray],
        query_text: str,
        top_k: int
    ) -> None:
        """
        Validate retrieval inputs before processing.
        
        Common validation logic that can be used by all strategies.
        Strategies can override this to add custom validation.
        
        Args:
            query_vector: Query vector (can be None for keyword-only)
            query_text: Query text
            top_k: Number of results
            
        Raises:
            ValueError: If inputs are invalid
            
        Note:
            This is NOT an abstract method - provides default validation.
        """
        if not query_text or not query_text.strip():
            raise ValueError("Query text cannot be empty")
        
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        
        if query_vector is not None:
            if not isinstance(query_vector, np.ndarray):
                raise ValueError("query_vector must be a numpy array")
            
            if query_vector.ndim != 1:
                raise ValueError(
                    f"query_vector must be 1-dimensional, got shape {query_vector.shape}"
                )
            
            if np.any(np.isnan(query_vector)):
                raise ValueError("query_vector contains NaN values")


# Type alias for better code readability
Strategy = RetrievalStrategy