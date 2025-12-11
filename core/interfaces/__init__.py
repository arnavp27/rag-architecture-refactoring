"""
Core Interfaces Package

This package contains all abstract base classes (interfaces) that define
the contracts for the RAG system components.

These interfaces enable:
- Dependency Inversion Principle (DIP)
- Adapter Pattern
- Strategy Pattern
- Testability (easy mocking)
- Flexibility (swap implementations)

Available Interfaces:
- LLMProvider: Language model providers
- VectorStore: Vector database operations
- Embedder: Text embedding models
- Reranker: Document reranking models
- RetrievalStrategy: Retrieval algorithm strategies
"""

from core.interfaces.llm_provider import LLMProvider, LLM
from core.interfaces.vector_store import VectorStore, VectorDB
from core.interfaces.embedder import Embedder
from core.interfaces.reranker import Reranker
from core.interfaces.retrieval_strategy import RetrievalStrategy, Strategy

__all__ = [
    # Primary interfaces
    "LLMProvider",
    "VectorStore",
    "Embedder",
    "Reranker",
    "RetrievalStrategy",
    
    # Type aliases
    "LLM",
    "VectorDB",
    "Strategy"
]