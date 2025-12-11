"""
TimingRetriever - Decorator that adds timing/performance tracking to any retrieval strategy

This decorator wraps any RetrievalStrategy and measures execution time,
logging performance metrics for monitoring and optimization.

Design Pattern: Decorator Pattern
SOLID Principles:
- Single Responsibility: Only handles timing concern
- Open/Closed: Adds timing without modifying wrapped strategy
- Liskov Substitution: Can be used wherever RetrievalStrategy is expected
"""

from typing import List, Dict, Any, Optional
import numpy as np
import logging
import time
from statistics import mean, median

from core.interfaces.retrieval_strategy import RetrievalStrategy


class TimingRetriever(RetrievalStrategy):
    """
    Decorator that adds performance timing to any retrieval strategy.
    
    Measures and logs execution time for each retrieval operation.
    Tracks timing statistics (min, max, avg, median) across multiple calls.
    
    Example usage:
        # Wrap any strategy with timing
        base_strategy = HybridStrategy(vector_store)
        timed_strategy = TimingRetriever(wrapped=base_strategy)
        
        # Each retrieval is timed and logged
        results = timed_strategy.retrieve(vec, "economy", top_k=5)
        # Logs: "HybridStrategy completed in 245.3ms"
        
        # Get timing statistics
        stats = timed_strategy.get_timing_stats()
        # {"avg_time_ms": 240.5, "min_time_ms": 220.1, ...}
    """
    
    def __init__(
        self,
        wrapped: RetrievalStrategy,
        log_threshold_ms: float = 1000.0
    ):
        """
        Initialize the timing decorator.
        
        Args:
            wrapped: The retrieval strategy to wrap with timing
            log_threshold_ms: Log a WARNING if retrieval takes longer than this (ms)
        
        Raises:
            TypeError: If wrapped doesn't implement RetrievalStrategy
            ValueError: If log_threshold_ms < 0
        """
        if not isinstance(wrapped, RetrievalStrategy):
            raise TypeError(
                f"wrapped must implement RetrievalStrategy interface, "
                f"got {type(wrapped)}"
            )
        
        if log_threshold_ms < 0:
            raise ValueError(f"log_threshold_ms must be >= 0, got {log_threshold_ms}")
        
        self._wrapped = wrapped
        self._log_threshold_ms = log_threshold_ms
        
        # Timing history
        self._execution_times: List[float] = []
        self._total_calls = 0
        
        self._logger = logging.getLogger(__name__)
        self._logger.info(
            f"TimingRetriever initialized (wrapping {wrapped.get_strategy_name()}, "
            f"threshold={log_threshold_ms}ms)"
        )
    
    def retrieve(
        self,
        query_vector: np.ndarray,
        query_text: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with timing measurement.
        
        Times the wrapped strategy's retrieve() method and logs performance.
        
        Args:
            query_vector: Query embedding vector
            query_text: Original query text
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            
        Returns:
            List of retrieved documents from wrapped strategy
        """
        # Start timing
        start_time = time.perf_counter()
        
        try:
            # Delegate to wrapped strategy
            results = self._wrapped.retrieve(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                filters=filters
            )
            
            # Calculate elapsed time
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            # Update statistics
            self._execution_times.append(elapsed_ms)
            self._total_calls += 1
            
            # Log timing
            strategy_name = self._wrapped.get_strategy_name()
            
            if elapsed_ms > self._log_threshold_ms:
                # Slow query warning
                self._logger.warning(
                    f"{strategy_name} took {elapsed_ms:.1f}ms "
                    f"(threshold: {self._log_threshold_ms}ms) "
                    f"for query: '{query_text[:30]}...'"
                )
            else:
                # Normal info log
                self._logger.info(
                    f"{strategy_name} completed in {elapsed_ms:.1f}ms "
                    f"(returned {len(results)} results)"
                )
            
            # Log stats periodically (every 10 calls)
            if self._total_calls % 10 == 0:
                stats = self.get_timing_stats()
                self._logger.info(
                    f"Timing stats (last {len(self._execution_times)} calls): "
                    f"avg={stats['avg_time_ms']:.1f}ms, "
                    f"median={stats['median_time_ms']:.1f}ms"
                )
            
            return results
            
        except Exception as e:
            # Log timing even on failure
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            self._logger.error(
                f"{self._wrapped.get_strategy_name()} failed after {elapsed_ms:.1f}ms: {e}"
            )
            raise
    
    def get_timing_stats(self) -> Dict[str, Any]:
        """
        Get timing statistics across all retrieve() calls.
        
        Returns:
            Dictionary with timing statistics
        """
        if not self._execution_times:
            return {
                "total_calls": 0,
                "avg_time_ms": 0.0,
                "min_time_ms": 0.0,
                "max_time_ms": 0.0,
                "median_time_ms": 0.0
            }
        
        return {
            "total_calls": self._total_calls,
            "avg_time_ms": mean(self._execution_times),
            "min_time_ms": min(self._execution_times),
            "max_time_ms": max(self._execution_times),
            "median_time_ms": median(self._execution_times)
        }
    
    def reset_timing_stats(self) -> None:
        """Reset all timing statistics."""
        calls_cleared = self._total_calls
        self._execution_times.clear()
        self._total_calls = 0
        self._logger.info(f"Timing stats reset ({calls_cleared} calls cleared)")
    
    def get_last_execution_time_ms(self) -> Optional[float]:
        """
        Get the execution time of the last retrieve() call.
        
        Returns:
            Execution time in milliseconds, or None if no calls yet
        """
        if not self._execution_times:
            return None
        return self._execution_times[-1]
    
    def get_strategy_name(self) -> str:
        """Return the name of this decorator + wrapped strategy."""
        return f"Timing({self._wrapped.get_strategy_name()})"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this decorator and wrapped strategy.
        
        Returns:
            Dictionary with strategy metadata
        """
        wrapped_info = self._wrapped.get_strategy_info()
        
        return {
            "name": self.get_strategy_name(),
            "description": "Timing decorator for retrieval strategies",
            "decorator": "Timing",
            "wrapped_strategy": wrapped_info,
            "timing_config": {
                "log_threshold_ms": self._log_threshold_ms
            },
            "timing_stats": self.get_timing_stats()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"TimingRetriever(wrapped={self._wrapped})"