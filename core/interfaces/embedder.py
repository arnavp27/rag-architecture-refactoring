"""
Embedder ABC - Abstract interface for text embedding models

This interface defines the contract for all embedding model implementations
(SentenceTransformers, OpenAI, Cohere, etc.), enabling the system to be
model-agnostic.

Design Pattern: Adapter Pattern (target interface for embedding models)
SOLID Principle: Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class Embedder(ABC):
    """
    Abstract interface for text embedding models.
    
    All embedding model implementations MUST implement this interface to ensure
    they can be used interchangeably in the RAG pipeline.
    
    Embeddings convert text into dense vector representations that capture
    semantic meaning, enabling similarity-based retrieval.
    """
    
    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Convert a single text query into an embedding vector.
        
        This is used for embedding the user's query before retrieval.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Embedding vector as 1D numpy array
                Shape: (embedding_dim,)
                Example: array with 384, 768, or 1024 dimensions
                
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If embedding generation fails
            
        Example:
            >>> embedder = SentenceTransformerEmbedder()
            >>> query_vector = embedder.embed_query("What is economic growth?")
            >>> print(query_vector.shape)
            (384,)
            >>> print(query_vector[:5])
            array([0.123, -0.456, 0.789, ...])
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts into embedding vectors (batched for efficiency).
        
        Batching is more efficient than calling embed_query() multiple times
        because it processes multiple texts in parallel.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            np.ndarray: 2D array of embedding vectors
                Shape: (len(texts), embedding_dim)
                Each row is one embedding vector
                
        Raises:
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If embedding generation fails
            
        Example:
            >>> texts = [
            ...     "Economic growth is important",
            ...     "Climate change requires action",
            ...     "Education improves society"
            ... ]
            >>> embeddings = embedder.embed_batch(texts)
            >>> print(embeddings.shape)
            (3, 384)
            >>> # Each row is an embedding for one text
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the embedding dimension (vector size).
        
        This is important for:
        - Validating vector dimensions before vector database operations
        - Allocating appropriate storage
        - Ensuring compatibility between embedder and vector store
        
        Returns:
            int: Dimension of embedding vectors
                Common values: 384, 768, 1024, 1536
                
        Example:
            >>> embedder = SentenceTransformerEmbedder(
            ...     model_name="all-MiniLM-L6-v2"
            ... )
            >>> print(embedder.get_dimension())
            384
        """
        pass
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Alias for embed_batch() for semantic clarity.
        
        When embedding documents (rather than queries), this method
        makes the code more readable. It's functionally identical to
        embed_batch().
        
        Args:
            documents (List[str]): List of documents to embed
            
        Returns:
            np.ndarray: 2D array of embedding vectors
            
        Note:
            This is NOT an abstract method - default implementation uses embed_batch().
        """
        return self.embed_batch(documents)
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Optional method to provide metadata about the model.
        
        Returns:
            dict: Model metadata (name, dimension, etc.)
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return {
            "model_type": self.__class__.__name__,
            "dimension": self.get_dimension()
        }