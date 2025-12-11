"""
Query - Domain model representing a user query to the RAG system

This is a pure data class with no business logic, following the
Domain-Driven Design principle of separating data from behavior.

Design Pattern: Data Transfer Object (DTO)
SOLID Principle: Single Responsibility Principle (SRP) - only holds data
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import numpy as np


@dataclass
class Query:
    """
    Represents a user query to the RAG system.
    
    This class encapsulates all information about a query:
    - The query text itself
    - Filters to apply during retrieval
    - Number of results desired
    - The query embedding vector (computed lazily)
    - Conversation history for context
    
    Attributes:
        text (str): The user's question or search query
        filters (Dict[str, Any]): Metadata filters for retrieval
        top_k (int): Number of results to return (default: 5)
        vector (Optional[np.ndarray]): Query embedding vector (computed lazily)
        conversation_history (Optional[List]): Previous conversation turns
    """
    
    text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    vector: Optional[np.ndarray] = None
    conversation_history: Optional[List[Dict[str, str]]] = None
    
    def __post_init__(self):
        """
        Validate query data after initialization.
        
        This ensures that Query objects are always in a valid state.
        Follows the "fail fast" principle.
        """
        # Validate query text
        if not self.text or not self.text.strip():
            raise ValueError("Query text cannot be empty")
        
        # Validate top_k
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        
        if self.top_k > 100:
            raise ValueError(f"top_k too large (max 100), got {self.top_k}")
        
        # Ensure filters is a dictionary
        if self.filters is None:
            self.filters = {}
        
        # Validate vector if provided
        if self.vector is not None:
            if not isinstance(self.vector, np.ndarray):
                raise ValueError("vector must be a numpy array")
            
            if self.vector.ndim != 1:
                raise ValueError(
                    f"vector must be 1-dimensional, got shape {self.vector.shape}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert query to dictionary for serialization.
        
        Useful for logging, debugging, and API responses.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the query
        """
        return {
            "text": self.text,
            "filters": self.filters,
            "top_k": self.top_k,
            "has_vector": self.vector is not None,
            "vector_shape": self.vector.shape if self.vector is not None else None,
            "has_history": self.conversation_history is not None,
            "history_length": len(self.conversation_history) if self.conversation_history else 0
        }
    
    def __str__(self) -> str:
        """String representation for debugging"""
        filters_str = f", filters={list(self.filters.keys())}" if self.filters else ""
        return f"Query(text='{self.text[:50]}...', top_k={self.top_k}{filters_str})"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (
            f"Query(text={self.text!r}, filters={self.filters!r}, "
            f"top_k={self.top_k}, vector={'present' if self.vector is not None else 'None'})"
        )