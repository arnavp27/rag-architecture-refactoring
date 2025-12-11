"""
Core Domain Package

This package contains all domain models (data classes) used throughout
the RAG system. These are pure data structures with no business logic.

Following Domain-Driven Design principles, these models:
- Represent core business concepts
- Are independent of infrastructure
- Have no external dependencies
- Validate their own data

Available Domain Models:
- Query: User query to the RAG system
- Document: Retrieved document with metadata and scores
- RAGResponse: Complete response from the pipeline
- PerformanceMetrics: Performance timing data
"""

from core.domain.query import Query
from core.domain.document import Document
from core.domain.response import RAGResponse
from core.domain.metrics import PerformanceMetrics

__all__ = [
    "Query",
    "Document",
    "RAGResponse",
    "PerformanceMetrics"
]