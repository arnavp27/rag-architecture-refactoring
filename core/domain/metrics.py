"""
PerformanceMetrics - Domain model for tracking RAG pipeline performance

This is a pure data class that captures performance metrics at each stage
of the RAG pipeline, enabling monitoring, optimization, and debugging.

Design Pattern: Data Transfer Object (DTO)
SOLID Principle: Single Responsibility Principle (SRP) - only holds metrics data
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for RAG pipeline execution.
    
    Tracks timing for each stage of the pipeline:
    1. Filter extraction (LLM extracts filters from query)
    2. Query embedding (text → vector)
    3. Retrieval (vector database search)
    4. Reranking (re-scoring candidates)
    5. Generation (LLM generates answer)
    
    Also tracks cache hits for optimization insights.
    
    Attributes:
        total_time_ms (float): Total end-to-end time in milliseconds
        filter_extraction_time_ms (float): Time to extract filters using LLM
        embedding_time_ms (float): Time to embed the query
        retrieval_time_ms (float): Time for vector database retrieval
        rerank_time_ms (float): Time for reranking candidates
        generation_time_ms (float): Time for LLM to generate answer
        cache_hits (Dict[str, bool]): Which operations hit cache
    """
    
    total_time_ms: float = 0.0
    filter_extraction_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    cache_hits: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validate metrics after initialization.
        """
        # Ensure cache_hits is a dictionary
        if self.cache_hits is None:
            self.cache_hits = {}
        
        # Validate that times are non-negative
        time_fields = [
            "total_time_ms",
            "filter_extraction_time_ms",
            "embedding_time_ms",
            "retrieval_time_ms",
            "rerank_time_ms",
            "generation_time_ms"
        ]
        
        for field_name in time_fields:
            value = getattr(self, field_name)
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative, got {value}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary for serialization.
        
        Useful for logging, API responses, and analytics.
        
        Returns:
            Dict[str, Any]: Dictionary representation of metrics
        """
        return {
            "total_time_ms": self.total_time_ms,
            "filter_extraction_time_ms": self.filter_extraction_time_ms,
            "embedding_time_ms": self.embedding_time_ms,
            "retrieval_time_ms": self.retrieval_time_ms,
            "rerank_time_ms": self.rerank_time_ms,
            "generation_time_ms": self.generation_time_ms,
            "cache_hits": self.cache_hits
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """
        Create PerformanceMetrics from dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary containing metrics data
            
        Returns:
            PerformanceMetrics: New PerformanceMetrics instance
        """
        return cls(
            total_time_ms=data.get("total_time_ms", 0.0),
            filter_extraction_time_ms=data.get("filter_extraction_time_ms", 0.0),
            embedding_time_ms=data.get("embedding_time_ms", 0.0),
            retrieval_time_ms=data.get("retrieval_time_ms", 0.0),
            rerank_time_ms=data.get("rerank_time_ms", 0.0),
            generation_time_ms=data.get("generation_time_ms", 0.0),
            cache_hits=data.get("cache_hits", {})
        )
    
    def get_breakdown_percentages(self) -> Dict[str, float]:
        """
        Calculate percentage of total time spent in each stage.
        
        Useful for identifying performance bottlenecks.
        
        Returns:
            Dict[str, float]: Percentage breakdown of time spent
        """
        if self.total_time_ms == 0:
            return {
                "filter_extraction": 0.0,
                "embedding": 0.0,
                "retrieval": 0.0,
                "rerank": 0.0,
                "generation": 0.0
            }
        
        return {
            "filter_extraction": (self.filter_extraction_time_ms / self.total_time_ms) * 100,
            "embedding": (self.embedding_time_ms / self.total_time_ms) * 100,
            "retrieval": (self.retrieval_time_ms / self.total_time_ms) * 100,
            "rerank": (self.rerank_time_ms / self.total_time_ms) * 100,
            "generation": (self.generation_time_ms / self.total_time_ms) * 100
        }
    
    def get_bottleneck(self) -> str:
        """
        Identify the slowest stage in the pipeline.
        
        Helps focus optimization efforts on the most time-consuming step.
        
        Returns:
            str: Name of the slowest stage
        """
        stages = {
            "filter_extraction": self.filter_extraction_time_ms,
            "embedding": self.embedding_time_ms,
            "retrieval": self.retrieval_time_ms,
            "rerank": self.rerank_time_ms,
            "generation": self.generation_time_ms
        }
        
        return max(stages.items(), key=lambda x: x[1])[0]
    
    def __str__(self) -> str:
        """String representation for debugging"""
        return (
            f"PerformanceMetrics(total={self.total_time_ms:.1f}ms, "
            f"bottleneck={self.get_bottleneck()})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging"""
        breakdown = self.get_breakdown_percentages()
        return (
            f"PerformanceMetrics(total={self.total_time_ms:.1f}ms, "
            f"filter={breakdown['filter_extraction']:.1f}%, "
            f"embed={breakdown['embedding']:.1f}%, "
            f"retrieval={breakdown['retrieval']:.1f}%, "
            f"rerank={breakdown['rerank']:.1f}%, "
            f"gen={breakdown['generation']:.1f}%)"
        )