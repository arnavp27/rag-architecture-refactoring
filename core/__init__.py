"""
Core Package - Domain Models and Interfaces

This is the heart of the RAG system architecture, containing:
1. Domain Models (Layer 1) - Pure data structures
2. Interfaces (Layer 2) - Abstract contracts

The core package has ZERO external dependencies and defines
the contracts that all other layers must follow.

Architecture Principle:
    All dependencies point INWARD to core.
    Core never depends on outer layers.

Import Structure:
    from core.domain import Query, Document, RAGResponse, PerformanceMetrics
    from core.interfaces import LLMProvider, VectorStore, Embedder, Reranker, RetrievalStrategy
"""

# Domain Models (Layer 1)
from core.domain import (
    Query,
    Document,
    RAGResponse,
    PerformanceMetrics
)

# Interfaces (Layer 2)
from core.interfaces import (
    LLMProvider,
    VectorStore,
    Embedder,
    Reranker,
    RetrievalStrategy
)

__all__ = [
    # Domain Models
    "Query",
    "Document",
    "RAGResponse",
    "PerformanceMetrics",
    
    # Interfaces
    "LLMProvider",
    "VectorStore",
    "Embedder",
    "Reranker",
    "RetrievalStrategy"
]

__version__ = "1.0.0"