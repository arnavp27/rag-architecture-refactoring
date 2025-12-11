"""
VectorStore ABC - Abstract interface for vector database operations

This interface defines the contract for all vector database implementations
(Weaviate, Pinecone, Qdrant, ChromaDB, etc.), enabling the system to be
database-agnostic.

Design Pattern: Adapter Pattern (target interface for vector databases)
SOLID Principle: Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np


class VectorStore(ABC):
    """
    Abstract interface for vector database operations.
    
    All vector database implementations MUST implement this interface to ensure
    they can be used interchangeably in retrieval strategies.
    
    Supports three types of search:
    1. Vector search (semantic similarity)
    2. Keyword search (BM25/full-text)
    3. Hybrid search (combination of both)
    """
    
    @abstractmethod
    def vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure vector similarity search.
        
        Uses cosine similarity (or other distance metrics) to find
        documents with embeddings most similar to the query vector.
        
        Args:
            query_vector (np.ndarray): Query embedding vector (1D array)
            top_k (int): Number of top results to return (default: 5)
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
                Example: {"theme": ["Economy"], "sentiment": "Positive"}
                
        Returns:
            List[Dict[str, Any]]: List of retrieved documents, each containing:
                - content (str): The document text
                - score (float): Similarity score (higher = more similar)
                - metadata (Dict): Document metadata
                - id (str): Document identifier
                
        Raises:
            ValueError: If query_vector is invalid (wrong shape, NaN, etc.)
            RuntimeError: If database connection fails
            
        Example:
            >>> vector_store = WeaviateAdapter()
            >>> query_vec = embedder.embed_query("economic growth")
            >>> results = vector_store.vector_search(query_vec, top_k=5)
            >>> for doc in results:
            ...     print(f"Score: {doc['score']:.3f} - {doc['content'][:100]}")
        """
        pass
    
    @abstractmethod
    def keyword_search(
        self,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform pure keyword-based search (BM25 or full-text search).
        
        Uses lexical matching to find documents containing query terms.
        Good for exact phrase matching and named entity retrieval.
        
        Args:
            query_text (str): The query text (not embedded)
            top_k (int): Number of top results to return (default: 5)
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
                
        Returns:
            List[Dict[str, Any]]: List of retrieved documents with same structure
                as vector_search()
                
        Raises:
            ValueError: If query_text is empty
            RuntimeError: If database connection fails
            
        Example:
            >>> results = vector_store.keyword_search("economic policy reform")
            >>> # Finds documents with exact keyword matches
        """
        pass
    
    @abstractmethod
    def hybrid_search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        alpha: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Combines semantic similarity (vector search) with lexical matching
        (keyword search) for improved retrieval quality. The alpha parameter
        controls the balance between the two approaches.
        
        Args:
            query_vector (np.ndarray): Query embedding vector
            query_text (str): Original query text (for keyword search)
            top_k (int): Number of top results to return (default: 5)
            alpha (float): Balance between vector and keyword search (0.0 to 1.0)
                - 0.0: Pure keyword search
                - 0.5: Equal balance (recommended)
                - 1.0: Pure vector search
            filters (Optional[Dict[str, Any]]): Metadata filters to apply
                
        Returns:
            List[Dict[str, Any]]: List of retrieved documents
            
        Raises:
            ValueError: If inputs are invalid or alpha not in [0, 1]
            RuntimeError: If database connection fails
            
        Example:
            >>> results = vector_store.hybrid_search(
            ...     query_vector=query_vec,
            ...     query_text="economic growth",
            ...     alpha=0.7  # Favor vector search slightly
            ... )
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """
        Close the connection to the vector database.
        
        Properly cleanup resources, close connections, and release any
        held resources. Should be called when done using the vector store.
        
        Example:
            >>> vector_store = WeaviateAdapter()
            >>> try:
            ...     results = vector_store.vector_search(query_vec)
            ... finally:
            ...     vector_store.close()
        """
        pass
    
    def is_connected(self) -> bool:
        """
        Check if connection to vector database is active.
        
        Optional method that can be overridden to check connection status.
        
        Returns:
            bool: True if connected, False otherwise
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return True
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector database collection.
        
        Optional method to retrieve metadata about the collection/index.
        
        Returns:
            Dict[str, Any]: Collection metadata (document count, schema, etc.)
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return {
            "store_type": self.__class__.__name__,
            "connected": self.is_connected()
        }


# Type alias for better code readability
VectorDB = VectorStore