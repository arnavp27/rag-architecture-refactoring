"""
Retrieval Decorators - Decorator Pattern implementations

Each decorator adds a cross-cutting concern to any retrieval strategy:
- CachingRetriever: Adds result caching with LRU eviction
- TimingRetriever: Adds performance timing and logging

Decorators can be stacked to combine multiple concerns:
    strategy = HybridStrategy(vector_store)
    strategy = CachingRetriever(strategy, cache_size=100)
    strategy = TimingRetriever(strategy, log_threshold_ms=1000)

All decorators implement the RetrievalStrategy interface, making them
transparent to client code.
"""

from components.retrieval.decorators.caching_retriever import CachingRetriever
from components.retrieval.decorators.timing_retriever import TimingRetriever

__all__ = [
    "CachingRetriever",
    "TimingRetriever",
]