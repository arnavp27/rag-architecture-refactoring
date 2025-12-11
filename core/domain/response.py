"""
RAGResponse - Domain model representing the complete RAG pipeline response

This is the complete response object returned to the user, containing the
generated answer, source documents, applied filters, and performance metrics.

Design Pattern: Data Transfer Object (DTO)
SOLID Principle: Single Responsibility Principle (SRP) - aggregates response data
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from core.domain.document import Document
from core.domain.metrics import PerformanceMetrics


@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.
    
    This is what the user receives after querying the RAG system.
    It contains everything needed to understand:
    - What answer was generated
    - Which sources were used
    - What filters were applied
    - How long it took (performance metrics)
    
    Attributes:
        answer (str): The generated answer text from the LLM
        sources (List[Document]): Source documents used to generate the answer
        filters_applied (Dict[str, Any]): Filters that were active during retrieval
        metrics (PerformanceMetrics): Performance timing data for each pipeline stage
    """
    
    answer: str
    sources: List[Document] = field(default_factory=list)
    filters_applied: Dict[str, Any] = field(default_factory=dict)
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    def __post_init__(self):
        """
        Validate response data after initialization.
        """
        # Validate answer
        if not isinstance(self.answer, str):
            raise ValueError(f"answer must be a string, got {type(self.answer)}")
        
        # Ensure sources is a list
        if self.sources is None:
            self.sources = []
        
        # Validate that sources are Document instances
        for i, source in enumerate(self.sources):
            if not isinstance(source, Document):
                raise ValueError(
                    f"sources[{i}] must be a Document instance, got {type(source)}"
                )
        
        # Ensure filters_applied is a dictionary
        if self.filters_applied is None:
            self.filters_applied = {}
        
        # Ensure metrics is a PerformanceMetrics instance
        if not isinstance(self.metrics, PerformanceMetrics):
            # If it's a dict, try to convert it
            if isinstance(self.metrics, dict):
                self.metrics = PerformanceMetrics.from_dict(self.metrics)
            else:
                raise ValueError(
                    f"metrics must be a PerformanceMetrics instance, got {type(self.metrics)}"
                )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert response to dictionary for serialization.
        
        This is essential for:
        - JSON API responses
        - Logging and debugging
        - Storage and caching
        
        Returns:
            Dict[str, Any]: Dictionary representation of the response
        """
        return {
            "answer": self.answer,
            "sources": [doc.to_dict() for doc in self.sources],
            "filters_applied": self.filters_applied,
            "metrics": self.metrics.to_dict(),
            "num_sources": len(self.sources)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RAGResponse':
        """
        Create RAGResponse from dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing response data
            
        Returns:
            RAGResponse: New RAGResponse instance
        """
        return cls(
            answer=data["answer"],
            sources=[Document.from_dict(doc) for doc in data.get("sources", [])],
            filters_applied=data.get("filters_applied", {}),
            metrics=PerformanceMetrics.from_dict(data.get("metrics", {}))
        )
    
    def get_source_preview(self, max_length: int = 100) -> List[str]:
        """
        Get preview snippets of source documents.
        
        Useful for displaying source context without overwhelming the user.
        
        Args:
            max_length (int): Maximum length of each preview snippet
            
        Returns:
            List[str]: List of preview snippets
        """
        previews = []
        for doc in self.sources:
            content = doc.content
            if len(content) > max_length:
                content = content[:max_length] + "..."
            previews.append(content)
        return previews
    
    def get_top_sources(self, n: int = 3) -> List[Document]:
        """
        Get the top N most relevant source documents.
        
        Sources are already sorted by relevance, so this just
        returns the first N documents.
        
        Args:
            n (int): Number of top sources to return
            
        Returns:
            List[Document]: Top N source documents
        """
        return self.sources[:n]
    
    def has_filters(self) -> bool:
        """
        Check if any filters were applied during retrieval.
        
        Returns:
            bool: True if filters were applied, False otherwise
        """
        return bool(self.filters_applied)
    
    def __str__(self) -> str:
        """String representation for debugging"""
        answer_preview = self.answer[:80] + "..." if len(self.answer) > 80 else self.answer
        return (
            f"RAGResponse(answer='{answer_preview}', "
            f"sources={len(self.sources)}, "
            f"time={self.metrics.total_time_ms:.1f}ms)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        return (
            f"RAGResponse(answer_len={len(self.answer)}, "
            f"num_sources={len(self.sources)}, "
            f"filters={list(self.filters_applied.keys())}, "
            f"total_time={self.metrics.total_time_ms:.1f}ms)"
        )