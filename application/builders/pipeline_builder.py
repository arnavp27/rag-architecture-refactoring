"""
RAGPipelineBuilder - Fluent API for constructing RAG pipelines

This builder provides a clean, fluent interface for constructing complex
RAG pipelines with all necessary components. It handles component creation,
validation, and assembly.

Design Pattern: Builder Pattern
SOLID Principles:
- Single Responsibility: Only builds RAGPipeline instances
- Open/Closed: New components can be added without modifying existing methods
- Dependency Inversion: Works with interfaces, not concrete implementations
"""

from typing import Optional
import logging

from core.interfaces.llm_provider import LLMProvider
from core.interfaces.embedder import Embedder
from core.interfaces.reranker import Reranker
from core.interfaces.vector_store import VectorStore
from core.interfaces.retrieval_strategy import RetrievalStrategy

from infrastructure.config.settings import Settings, get_settings
from infrastructure.adapters.weaviate_adapter import WeaviateAdapter

from components.retrieval.retriever import Retriever
from components.retrieval.strategies import (
    VectorOnlyStrategy,
    KeywordOnlyStrategy,
    HybridStrategy
)
from components.retrieval.decorators import (
    CachingRetriever,
    TimingRetriever
)

from application.factories.llm_factory import LLMFactory
from application.factories.model_factory import ModelFactory
from application.pipeline.rag_pipeline import RAGPipeline


class RAGPipelineBuilder:
    """
    Builder for constructing RAG pipelines with a fluent API.
    
    Provides methods to configure each component:
    - LLM provider (with fallback)
    - Embedder
    - Vector store
    - Retrieval strategy
    - Decorators (caching, timing)
    - Reranker
    
    All methods return `self` for method chaining.
    
    Example:
        pipeline = (RAGPipelineBuilder()
            .with_llm(primary="gemini", fallback="ollama")
            .with_embedder("sentence-transformers")
            .with_vector_store(host="localhost", port=8080)
            .with_retrieval_strategy("hybrid")
            .with_caching(cache_size=100)
            .with_timing()
            .with_reranker("cross-encoder")
            .build())
        
        response = pipeline.query("What are recent economic policies?")
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the builder.
        
        Args:
            settings: Configuration settings (optional, uses default if None)
        """
        self._settings = settings or get_settings()
        self._logger = logging.getLogger(__name__)
        
        # Components to be built
        self._llm: Optional[LLMProvider] = None
        self._embedder: Optional[Embedder] = None
        self._vector_store: Optional[VectorStore] = None
        self._retrieval_strategy: Optional[RetrievalStrategy] = None
        self._reranker: Optional[Reranker] = None
        
        # Flags
        self._use_caching = False
        self._cache_size = 100
        self._use_timing = False
        self._timing_threshold_ms = 1000.0
        
        # Build state
        self._built = False
        
        self._logger.debug("RAGPipelineBuilder initialized")
    
    def with_llm(
        self,
        primary: str = "gemini",
        fallback: Optional[str] = "ollama"
    ) -> 'RAGPipelineBuilder':
        """
        Configure LLM provider with fallback.
        
        Args:
            primary: Primary LLM provider ("gemini" or "ollama")
            fallback: Fallback LLM provider (optional)
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_llm(primary="gemini", fallback="ollama")
        """
        self._logger.info(f"Configuring LLM: primary={primary}, fallback={fallback}")
        self._llm = LLMFactory.create_with_fallback(
            primary=primary,
            fallback=fallback,
            settings=self._settings
        )
        return self
    
    def with_embedder(
        self,
        model_type: str = "sentence-transformers"
    ) -> 'RAGPipelineBuilder':
        """
        Configure embedder.
        
        Args:
            model_type: Type of embedder ("sentence-transformers")
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_embedder("sentence-transformers")
        """
        self._logger.info(f"Configuring embedder: {model_type}")
        self._embedder = ModelFactory.create_embedder(
            model_type=model_type,
            settings=self._settings
        )
        return self
    
    def with_vector_store(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None
    ) -> 'RAGPipelineBuilder':
        """
        Configure vector store (Weaviate).
        
        Args:
            host: Weaviate host (optional, uses settings default)
            port: Weaviate port (optional, uses settings default)
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_vector_store(host="localhost", port=8080)
        """
        host = host or self._settings.weaviate_host
        port = port or self._settings.weaviate_port
        
        self._logger.info(f"Configuring vector store: {host}:{port}")
        self._vector_store = WeaviateAdapter(
            host=host,
            port=port,
            collection_name=self._settings.weaviate_collection
        )
        return self
    
    def with_retrieval_strategy(
        self,
        strategy_type: str = "hybrid",
        **kwargs
    ) -> 'RAGPipelineBuilder':
        """
        Configure retrieval strategy.
        
        Args:
            strategy_type: Type of strategy ("vector", "keyword", or "hybrid")
            **kwargs: Additional strategy parameters (e.g., rrf_k for hybrid)
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_retrieval_strategy("hybrid", rrf_k=60)
        """
        if self._vector_store is None:
            raise RuntimeError(
                "Vector store must be configured before retrieval strategy. "
                "Call with_vector_store() first."
            )
        
        self._logger.info(f"Configuring retrieval strategy: {strategy_type}")
        
        strategy_type = strategy_type.lower()
        
        if strategy_type == "vector":
            self._retrieval_strategy = VectorOnlyStrategy(self._vector_store)
        
        elif strategy_type == "keyword":
            self._retrieval_strategy = KeywordOnlyStrategy(self._vector_store)
        
        elif strategy_type == "hybrid":
            rrf_k = kwargs.get("rrf_k", 60)
            self._retrieval_strategy = HybridStrategy(
                self._vector_store,
                rrf_k=rrf_k
            )
        
        else:
            raise ValueError(
                f"Unknown strategy type: {strategy_type}. "
                f"Supported types: 'vector', 'keyword', 'hybrid'"
            )
        
        return self
    
    def with_caching(
        self,
        cache_size: int = 100
    ) -> 'RAGPipelineBuilder':
        """
        Enable caching decorator for retrieval.
        
        Args:
            cache_size: Maximum number of cached results
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_caching(cache_size=100)
        """
        self._logger.info(f"Enabling caching with size: {cache_size}")
        self._use_caching = True
        self._cache_size = cache_size
        return self
    
    def with_timing(
        self,
        threshold_ms: float = 1000.0
    ) -> 'RAGPipelineBuilder':
        """
        Enable timing decorator for retrieval.
        
        Args:
            threshold_ms: Warning threshold in milliseconds
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_timing(threshold_ms=1000.0)
        """
        self._logger.info(f"Enabling timing with threshold: {threshold_ms}ms")
        self._use_timing = True
        self._timing_threshold_ms = threshold_ms
        return self
    
    def with_reranker(
        self,
        model_type: str = "cross-encoder"
    ) -> 'RAGPipelineBuilder':
        """
        Configure reranker.
        
        Args:
            model_type: Type of reranker ("cross-encoder")
            
        Returns:
            self for method chaining
            
        Example:
            builder.with_reranker("cross-encoder")
        """
        self._logger.info(f"Configuring reranker: {model_type}")
        self._reranker = ModelFactory.create_reranker(
            model_type=model_type,
            settings=self._settings
        )
        return self
    
    def build(self) -> RAGPipeline:
        """
        Build and return the RAG pipeline.
        
        Validates that all required components are configured,
        applies decorators, and constructs the final pipeline.
        
        Returns:
            RAGPipeline: Fully configured pipeline ready to use
            
        Raises:
            RuntimeError: If builder has already been used or missing components
            
        Example:
            pipeline = builder.build()
            response = pipeline.query("What is AI?")
        """
        if self._built:
            raise RuntimeError(
                "Builder can only be used once. Create a new builder for another pipeline."
            )
        
        # Validate required components
        self._validate_components()
        
        # Apply decorators to retrieval strategy
        strategy = self._apply_decorators()
        
        # Create retriever with decorated strategy
        retriever = Retriever(strategy=strategy)
        
        # Build the pipeline
        self._logger.info("Building RAG pipeline...")
        pipeline = RAGPipeline(
            llm=self._llm,
            embedder=self._embedder,
            retriever=retriever,
            reranker=self._reranker,
            vector_store=self._vector_store
        )
        
        self._built = True
        self._logger.info("RAG pipeline built successfully!")
        
        return pipeline
    
    def _validate_components(self) -> None:
        """
        Validate that all required components are configured.
        
        Raises:
            RuntimeError: If any required component is missing
        """
        missing = []
        
        if self._llm is None:
            missing.append("LLM (call with_llm())")
        
        if self._embedder is None:
            missing.append("Embedder (call with_embedder())")
        
        if self._vector_store is None:
            missing.append("Vector store (call with_vector_store())")
        
        if self._retrieval_strategy is None:
            missing.append("Retrieval strategy (call with_retrieval_strategy())")
        
        # Reranker is optional
        
        if missing:
            raise RuntimeError(
                f"Missing required components: {', '.join(missing)}"
            )
    
    def _apply_decorators(self) -> RetrievalStrategy:
        """
        Apply decorators to the retrieval strategy.
        
        Applies decorators in order: Caching (inner) → Timing (outer)
        This ensures timing includes cache lookup time.
        
        Returns:
            Decorated retrieval strategy
        """
        strategy = self._retrieval_strategy
        
        # Apply caching first (innermost decorator)
        if self._use_caching:
            self._logger.debug(f"Applying caching decorator (size={self._cache_size})")
            strategy = CachingRetriever(
                wrapped=strategy,
                cache_size=self._cache_size
            )
        
        # Apply timing last (outermost decorator)
        if self._use_timing:
            self._logger.debug(f"Applying timing decorator (threshold={self._timing_threshold_ms}ms)")
            strategy = TimingRetriever(
                wrapped=strategy,
                log_threshold_ms=self._timing_threshold_ms
            )
        
        return strategy