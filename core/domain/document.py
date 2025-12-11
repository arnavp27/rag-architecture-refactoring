"""
Document - Domain model representing a retrieved document

This is a pure data class representing a document retrieved from the vector store.
It includes the document content, metadata, and various relevance scores.

Design Pattern: Data Transfer Object (DTO)
SOLID Principle: Single Responsibility Principle (SRP) - only holds data
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class Document:
    """
    Represents a document retrieved from the vector store.
    
    A document goes through multiple stages in the RAG pipeline:
    1. Retrieved from vector store (has initial score)
    2. Optionally reranked (has rerank_score)
    3. Used as context for generation
    
    Attributes:
        content (str): The actual text/statement content
        metadata (Dict[str, Any]): Document metadata (theme, sentiment, politician, etc.)
        score (float): Initial retrieval score (from vector/keyword search)
        embedding_index (Optional[int]): Index in the original embedding collection
        rerank_score (Optional[float]): Score from reranker (if reranking was applied)
        document_id (Optional[str]): Unique identifier for the document
    """
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding_index: Optional[int] = None
    rerank_score: Optional[float] = None
    document_id: Optional[str] = None
    
    def __post_init__(self):
        """
        Validate document data after initialization.
        
        Ensures document is in a valid state.
        """
        # Validate content
        if not self.content or not self.content.strip():
            raise ValueError("Document content cannot be empty")
        
        # Validate scores
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be a number, got {type(self.score)}")
        
        if self.rerank_score is not None:
            if not isinstance(self.rerank_score, (int, float)):
                raise ValueError(
                    f"rerank_score must be a number, got {type(self.rerank_score)}"
                )
        
        # Ensure metadata is a dictionary
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert document to dictionary for serialization.
        
        This is useful for:
        - JSON serialization for API responses
        - Logging and debugging
        - Storage in databases
        
        Returns:
            Dict[str, Any]: Dictionary representation of the document
        """
        return {
            "content": self.content,
            "metadata": self.metadata,
            "score": self.score,
            "embedding_index": self.embedding_index,
            "rerank_score": self.rerank_score,
            "document_id": self.document_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """
        Create Document from dictionary.
        
        This is the inverse of to_dict(), useful for deserializing
        documents from JSON, databases, etc.
        
        Args:
            data (Dict[str, Any]): Dictionary containing document data
            
        Returns:
            Document: New Document instance
        """
        return cls(
            content=data["content"],
            metadata=data.get("metadata", {}),
            score=data.get("score", 0.0),
            embedding_index=data.get("embedding_index"),
            rerank_score=data.get("rerank_score"),
            document_id=data.get("document_id")
        )
    
    def get_final_score(self) -> float:
        """
        Get the most relevant score for this document.
        
        If rerank_score exists, it's more accurate than the initial score.
        Otherwise, use the initial retrieval score.
        
        Returns:
            float: The most relevant score to use for ranking
        """
        return self.rerank_score if self.rerank_score is not None else self.score
    
    def __str__(self) -> str:
        """String representation for debugging"""
        content_preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        score_info = f"score={self.score:.3f}"
        if self.rerank_score is not None:
            score_info += f", rerank={self.rerank_score:.3f}"
        return f"Document({score_info}, content='{content_preview}')"
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (
            f"Document(content={self.content[:50]!r}..., "
            f"score={self.score}, rerank_score={self.rerank_score}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )