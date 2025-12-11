"""
SentenceTransformerEmbedder - Adapter for Sentence Transformers

This adapter implements the Embedder interface using the sentence-transformers
library, providing text-to-vector conversion for semantic search.

Design Pattern: Adapter Pattern (adapts SentenceTransformer to Embedder interface)
SOLID Principles:
- Single Responsibility: Only handles text embedding
- Dependency Inversion: Implements Embedder abstraction
"""

import logging
from typing import List, Union
import numpy as np
from core.interfaces import Embedder

# Sentence Transformers imports (external dependency)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class SentenceTransformerEmbedder(Embedder):
    """
    Adapter for Sentence Transformers embedding models.
    
    Implements the Embedder interface using sentence-transformers library.
    Provides efficient text-to-vector conversion for semantic search.
    
    Features:
    - Batch processing for efficiency
    - GPU support (if available)
    - Multiple model options
    - Consistent vector dimensions
    
    Attributes:
        model_name (str): HuggingFace model name
        device (str): Device for inference ('cpu' or 'cuda')
        dimension (int): Embedding vector dimension
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize SentenceTransformer embedder.
        
        Args:
            model_name (str): HuggingFace model name
                Popular options:
                - "all-MiniLM-L6-v2" (384 dim, fast, recommended)
                - "all-mpnet-base-v2" (768 dim, higher quality)
                - "multi-qa-mpnet-base-dot-v1" (768 dim, for Q&A)
            device (str): Device for inference ('cpu' or 'cuda')
            
        Raises:
            ImportError: If sentence-transformers package is not installed
            RuntimeError: If model loading fails
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self._model = SentenceTransformer(model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            
            self.logger.info(
                f"SentenceTransformerEmbedder loaded: {model_name} "
                f"(dimension={self._dimension}, device={device})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load SentenceTransformer model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def embed_query(self, text: str) -> np.ndarray:
        """
        Convert a single text query into an embedding vector.
        
        Args:
            text (str): Input text to embed
            
        Returns:
            np.ndarray: Embedding vector (1D array)
            
        Raises:
            ValueError: If text is empty
            RuntimeError: If embedding generation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Generate embedding
            embedding = self._model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalize for cosine similarity
            )
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Embedding failed: {e}")
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts into embedding vectors (batched for efficiency).
        
        Batching is significantly more efficient than calling embed_query()
        multiple times because it processes texts in parallel.
        
        Args:
            texts (List[str]): List of input texts to embed
            
        Returns:
            np.ndarray: 2D array of embedding vectors
                Shape: (len(texts), dimension)
            
        Raises:
            ValueError: If texts list is empty or contains invalid entries
            RuntimeError: If embedding generation fails
        """
        if not texts:
            raise ValueError("texts list cannot be empty")
        
        # Validate texts
        for i, text in enumerate(texts):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} is empty")
        
        try:
            # Generate embeddings in batch
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                batch_size=32,  # Process in batches of 32
                show_progress_bar=False
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Batch embedding generation failed: {e}")
            raise RuntimeError(f"Batch embedding failed: {e}")
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension (vector size).
        
        Returns:
            int: Dimension of embedding vectors
        """
        return self._dimension
    
    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            dict: Model metadata
        """
        return {
            "model_type": "SentenceTransformer",
            "model_name": self.model_name,
            "dimension": self._dimension,
            "device": self.device,
            "max_seq_length": self._model.max_seq_length
        }
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"SentenceTransformerEmbedder(model={self.model_name}, "
            f"dim={self._dimension}, device={self.device})"
        )