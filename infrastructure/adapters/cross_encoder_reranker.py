"""
CrossEncoderReranker - Adapter for Cross-Encoder reranking models

This adapter implements the Reranker interface using cross-encoder models
for accurate document reranking. Cross-encoders directly score query-document
pairs, providing superior relevance assessment compared to bi-encoders.

Design Pattern: Adapter Pattern (adapts CrossEncoder to Reranker interface)
SOLID Principles:
- Single Responsibility: Only handles document reranking
- Dependency Inversion: Implements Reranker abstraction
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from core.interfaces import Reranker

# Cross-encoder imports (external dependency)
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False


class CrossEncoderReranker(Reranker):
    """
    Adapter for Cross-Encoder reranking models.
    
    Implements the Reranker interface using cross-encoder models which
    directly score query-document relevance. This provides more accurate
    relevance scores than bi-encoder approaches.
    
    Two-stage retrieval workflow:
    1. Initial retrieval: Get ~100 candidates using fast bi-encoder
    2. Reranking: Use cross-encoder to accurately score top candidates
    
    Features:
    - Direct query-document scoring
    - Batch processing for efficiency
    - GPU support (if available)
    - Automatic score normalization
    
    Attributes:
        model_name (str): HuggingFace cross-encoder model name
        device (str): Device for inference ('cpu' or 'cuda')
        max_length (int): Maximum sequence length
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        max_length: int = 512
    ):
        """
        Initialize CrossEncoder reranker.
        
        Args:
            model_name (str): HuggingFace cross-encoder model name
                Popular options:
                - "ms-marco-MiniLM-L-6-v2" (fast, good quality)
                - "ms-marco-MiniLM-L-12-v2" (slower, better quality)
                - "ms-marco-electra-base" (highest quality)
            device (str): Device for inference ('cpu' or 'cuda')
            max_length (int): Maximum sequence length for model
            
        Raises:
            ImportError: If sentence-transformers package is not installed
            RuntimeError: If model loading fails
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self._model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device
            )
            
            self.logger.info(
                f"CrossEncoderReranker loaded: {model_name} (device={device})"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load CrossEncoder model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate documents based on relevance to query.
        
        Args:
            query (str): The search query
            candidates (List[Dict[str, Any]]): Candidate documents to rerank
            top_k (Optional[int]): Return only top K results
            
        Returns:
            List[Dict[str, Any]]: Reranked documents sorted by relevance
                Each document has added field 'rerank_score'
                
        Raises:
            ValueError: If query is empty or candidates list is empty
            RuntimeError: If reranking fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        try:
            # Get relevance scores
            scores = self.get_scores(query, candidates)
            
            # Add rerank scores to candidates
            reranked_candidates = []
            for candidate, score in zip(candidates, scores):
                # Create a copy to avoid modifying original
                reranked_doc = candidate.copy()
                reranked_doc['rerank_score'] = float(score)
                reranked_candidates.append(reranked_doc)
            
            # Sort by rerank score (descending)
            reranked_candidates.sort(
                key=lambda x: x['rerank_score'],
                reverse=True
            )
            
            # Return top K if specified
            if top_k is not None:
                reranked_candidates = reranked_candidates[:top_k]
            
            self.logger.debug(
                f"Reranked {len(candidates)} candidates, "
                f"returning top {len(reranked_candidates)}"
            )
            
            return reranked_candidates
            
        except Exception as e:
            self.logger.error(f"Reranking failed: {e}")
            raise RuntimeError(f"Reranking failed: {e}")
    
    def get_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Get relevance scores without modifying candidates.
        
        Args:
            query (str): The search query
            candidates (List[Dict[str, Any]]): Candidate documents
            
        Returns:
            List[float]: Relevance scores (same order as input)
            
        Raises:
            ValueError: If query is empty or candidates list is empty
            RuntimeError: If scoring fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        if not candidates:
            raise ValueError("Candidates list cannot be empty")
        
        try:
            # Prepare query-document pairs
            pairs = []
            for candidate in candidates:
                content = candidate.get('content', '')
                if not content:
                    # If no content, try to get statement or text
                    content = candidate.get('statement', candidate.get('text', ''))
                
                pairs.append([query, content])
            
            # Get scores from cross-encoder
            # Scores are logits, can be any real number
            scores = self._model.predict(
                pairs,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Convert to list of floats
            return [float(score) for score in scores]
            
        except Exception as e:
            self.logger.error(f"Scoring failed: {e}")
            raise RuntimeError(f"Scoring failed: {e}")
    
    def rerank_with_scores(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Rerank documents and return both reranked list and scores.
        
        Args:
            query (str): The search query
            candidates (List[Dict[str, Any]]): Candidate documents
            top_k (Optional[int]): Return only top K results
            
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]:
                - Reranked documents
                - Corresponding scores
        """
        reranked_docs = self.rerank(query, candidates, top_k)
        scores = [doc.get('rerank_score', 0.0) for doc in reranked_docs]
        return reranked_docs, scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the reranking model.
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        return {
            "model_type": "CrossEncoder",
            "model_name": self.model_name,
            "device": self.device,
            "max_length": self.max_length
        }
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (
            f"CrossEncoderReranker(model={self.model_name}, device={self.device})"
        )
    
    