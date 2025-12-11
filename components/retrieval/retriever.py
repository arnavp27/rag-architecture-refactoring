"""
Retriever - Context class for Strategy Pattern

This is the Context class in the Strategy Pattern. It holds a reference to a
RetrievalStrategy and delegates the actual retrieval work to it.

The key benefit: strategies can be swapped at runtime without changing the code
that uses the Retriever.

Design Pattern: Strategy Pattern (context class)
SOLID Principles:
- Single Responsibility: Only manages strategy delegation
- Open/Closed: New strategies can be added without modifying this class
- Dependency Inversion: Depends on RetrievalStrategy interface
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging

from core.interfaces.retrieval_strategy import RetrievalStrategy


class Retriever:
    """
    Context class for retrieval strategies.
    
    This class allows strategies to be swapped at runtime, enabling:
    - Different retrieval approaches for different query types
    - A/B testing of retrieval methods
    - Dynamic optimization based on query characteristics
    
    Example usage:
        # Start with hybrid strategy
        retriever = Retriever(strategy=HybridStrategy(vector_store))
        results = retriever.retrieve(query_vec, query_text, top_k=5)
        
        # Switch to vector-only strategy at runtime
        retriever.set_strategy(VectorOnlyStrategy(vector_store))
        results = retriever.retrieve(query_vec, query_text, top_k=5)
    """
    
    def __init__(self, strategy: RetrievalStrategy):
        """
        Initialize the retriever with a strategy.
        
        Args:
            strategy: Initial retrieval strategy to use
        
        Raises:
            TypeError: If strategy doesn't implement RetrievalStrategy
        """
        if not isinstance(strategy, RetrievalStrategy):
            raise TypeError(
                f"strategy must implement RetrievalStrategy interface, "
                f"got {type(strategy)}"
            )
        
        self._strategy = strategy
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"Retriever initialized with strategy: {strategy.get_strategy_name()}"
        )
    
    def set_strategy(self, strategy: RetrievalStrategy) -> None:
        """
        Change the retrieval strategy at runtime.
        
        This is the key method that enables the Strategy Pattern.
        It allows switching algorithms without affecting client code.
        
        Args:
            strategy: New retrieval strategy to use
        
        Raises:
            TypeError: If strategy doesn't implement RetrievalStrategy
        
        Example:
            retriever = Retriever(VectorOnlyStrategy(store))
            
            # Later, switch to hybrid
            retriever.set_strategy(HybridStrategy(store))
        """
        if not isinstance(strategy, RetrievalStrategy):
            raise TypeError(
                f"strategy must implement RetrievalStrategy interface, "
                f"got {type(strategy)}"
            )
        
        old_strategy = self._strategy.get_strategy_name()
        new_strategy = strategy.get_strategy_name()
        
        self._strategy = strategy
        
        self._logger.info(
            f"Strategy changed: {old_strategy} -> {new_strategy}"
        )
    
    def get_current_strategy(self) -> RetrievalStrategy:
        """
        Get the current retrieval strategy.
        
        Returns:
            The current RetrievalStrategy instance
        """
        return self._strategy
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using the current strategy.
        
        This method simply delegates to the current strategy's retrieve() method.
        The actual retrieval logic is in the strategy implementations.
        
        Args:
            query_vector: Query embedding vector
            query_text: Original query text
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents
            
        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If retrieval fails
        """
        self._logger.debug(
            f"Delegating retrieval to {self._strategy.get_strategy_name()}"
        )
        
        # Delegate to the current strategy
        return self._strategy.retrieve(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            filters=filters
        )
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about the current strategy.
        
        Returns:
            Dictionary with current strategy metadata
        """
        return self._strategy.get_strategy_info()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Retriever(strategy={self._strategy.get_strategy_name()})"