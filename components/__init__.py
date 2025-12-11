"""
Components Layer - Business logic and design patterns

This layer contains the implementation of Strategy and Decorator patterns
for retrieval operations.

Layer 4 of 5 in the architecture:
- Domain (Layer 1) - Pure data models
- Core Interfaces (Layer 2) - Abstract contracts
- Infrastructure (Layer 3) - External service adapters
- Components (Layer 4) - Business logic & patterns ← YOU ARE HERE
- Application (Layer 5) - High-level orchestration

Design Patterns Implemented:
- Strategy Pattern: Interchangeable retrieval algorithms
- Decorator Pattern: Cross-cutting concerns (caching, timing)
"""

# Import retrieval components for easier access
from components.retrieval.retriever import Retriever

from components.retrieval.strategies.vector_only_strategy import VectorOnlyStrategy
from components.retrieval.strategies.keyword_only_strategy import KeywordOnlyStrategy
from components.retrieval.strategies.hybrid_strategy import HybridStrategy

from components.retrieval.decorators.caching_retriever import CachingRetriever
from components.retrieval.decorators.timing_retriever import TimingRetriever

__all__ = [
    # Context class
    "Retriever",
    
    # Strategies
    "VectorOnlyStrategy",
    "KeywordOnlyStrategy",
    "HybridStrategy",
    
    # Decorators
    "CachingRetriever",
    "TimingRetriever",
]