"""
CachingRetriever - Decorator that adds caching to any retrieval strategy

This decorator wraps any RetrievalStrategy and adds caching functionality
to avoid redundant retrievals for the same query.

Ported from: RAG_v2/retrieval/retriever.py - result_cache and embedding_cache logic

Design Pattern: Decorator Pattern
SOLID Principles:
- Single Responsibility: Only handles caching concern
- Open/Closed: Adds caching without modifying wrapped strategy
- Liskov Substitution: Can be used wherever RetrievalStrategy is expected
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import hashlib
import json
from collections import OrderedDict

from core.interfaces.retrieval_strategy import RetrievalStrategy


class CachingRetriever(RetrievalStrategy):
    """
    Decorator that adds caching to any retrieval strategy.
    
    Caches retrieval results based on query text and filters to avoid
    redundant database queries. Uses LRU (Least Recently Used) eviction
    when cache size limit is reached.
    
    Example usage:
        # Wrap any strategy with caching
        base_strategy = HybridStrategy(vector_store)
        cached_strategy = CachingRetriever(
            wrapped=base_strategy,
            cache_size=100
        )
        
        # First call: retrieves from database
        results1 = cached_strategy.retrieve(vec, "economy", top_k=5)
        
        # Second call with same query: returns from cache (fast!)
        results2 = cached_strategy.retrieve(vec, "economy", top_k=5)
    """
    
    def __init__(
        self,
        wrapped: RetrievalStrategy,
        cache_size: int = 100
    ):
        """
        Initialize the caching decorator.
        
        Args:
            wrapped: The retrieval strategy to wrap with caching
            cache_size: Maximum number of cached results (LRU eviction)
        
        Raises:
            TypeError: If wrapped doesn't implement RetrievalStrategy
            ValueError: If cache_size < 1
        """
        if not isinstance(wrapped, RetrievalStrategy):
            raise TypeError(
                f"wrapped must implement RetrievalStrategy interface, "
                f"got {type(wrapped)}"
            )
        
        if cache_size < 1:
            raise ValueError(f"cache_size must be at least 1, got {cache_size}")
        
        self._wrapped = wrapped
        self._cache_size = cache_size
        
        # LRU cache using OrderedDict
        self._result_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
        
        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"CachingRetriever initialized (wrapping {wrapped.get_strategy_name()}, "
            f"cache_size={cache_size})"
        )
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with caching.
        
        Checks cache first. If miss, delegates to wrapped strategy
        and caches the result.
        
        Args:
            query_vector: Query embedding vector
            query_text: Original query text
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents (from cache or fresh retrieval)
        """
        # Generate cache key from query and filters
        cache_key = self._generate_cache_key(query_text, filters, top_k)
        
        # Check cache
        if cache_key in self._result_cache:
            self._cache_hits += 1
            
            # Move to end (mark as recently used)
            self._result_cache.move_to_end(cache_key)
            
            cached_results = self._result_cache[cache_key]
            
            self._logger.debug(
                f"Cache HIT for query: '{query_text[:30]}...' "
                f"(hit rate: {self.get_cache_hit_rate():.1%})"
            )
            
            # Return a copy to prevent cache corruption
            return [result.copy() for result in cached_results]
        
        # Cache miss - retrieve from wrapped strategy
        self._cache_misses += 1
        self._logger.debug(
            f"Cache MISS for query: '{query_text[:30]}...' "
            f"(hit rate: {self.get_cache_hit_rate():.1%})"
        )
        
        # Delegate to wrapped strategy
        results = self._wrapped.retrieve(
            query_vector=query_vector,
            query_text=query_text,
            top_k=top_k,
            filters=filters
        )
        
        # Store in cache
        self._cache_results(cache_key, results)
        
        return results
    
    def _generate_cache_key(
        self,
        query_text: str,
        filters: Optional[Dict[str, Any]],
        top_k: int
    ) -> str:
        """
        Generate a unique cache key from query and parameters.
        
        Args:
            query_text: Query text
            filters: Metadata filters
            top_k: Number of results
            
        Returns:
            MD5 hash as cache key
        """
        # Normalize query text
        normalized_query = query_text.strip().lower()
        
        # Sort filters for consistent hashing
        sorted_filters = None
        if filters:
            sorted_filters = json.dumps(filters, sort_keys=True)
        
        # Create key string
        key_string = f"{normalized_query}|{sorted_filters}|{top_k}"
        
        # Hash to fixed-length key
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_results(
        self,
        cache_key: str,
        results: List[Dict[str, Any]]
    ) -> None:
        """
        Store results in cache with LRU eviction.
        
        Args:
            cache_key: Cache key
            results: Results to cache
        """
        # Store in cache
        self._result_cache[cache_key] = [result.copy() for result in results]
        
        # LRU eviction if cache is full
        if len(self._result_cache) > self._cache_size:
            # Remove oldest entry (first item in OrderedDict)
            evicted_key = next(iter(self._result_cache))
            del self._result_cache[evicted_key]
            self._logger.debug(f"Evicted oldest cache entry (size limit: {self._cache_size})")
    
    def clear_cache(self) -> None:
        """Clear all cached results."""
        entries_cleared = len(self._result_cache)
        self._result_cache.clear()
        self._logger.info(f"Cache cleared ({entries_cleared} entries)")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self._result_cache),
            "max_cache_size": self._cache_size
        }
    
    def get_cache_hit_rate(self) -> float:
        """
        Get cache hit rate as a percentage.
        
        Returns:
            Hit rate (0.0 to 1.0)
        """
        total = self._cache_hits + self._cache_misses
        return self._cache_hits / total if total > 0 else 0.0
    
    def get_strategy_name(self) -> str:
        """Return the name of this decorator + wrapped strategy."""
        return f"Caching({self._wrapped.get_strategy_name()})"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this decorator and wrapped strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        wrapped_info = self._wrapped.get_strategy_info()
        
        return {
            "name": self.get_strategy_name(),
            "description": "Caching decorator for retrieval strategies",
            "decorator": "Caching",
            "wrapped_strategy": wrapped_info,
            "cache_config": {
                "max_size": self._cache_size,
                "current_size": len(self._result_cache)
            },
            "cache_stats": self.get_cache_stats()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"CachingRetriever(wrapped={self._wrapped}, cache_size={self._cache_size})"