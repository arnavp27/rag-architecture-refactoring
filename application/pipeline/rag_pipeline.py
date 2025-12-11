"""
RAGPipeline - Main facade for the RAG system

This is the Facade that provides a simple interface to the complex RAG subsystem.
It orchestrates all components to execute the complete RAG pipeline.

Ported from: RAG_v2/pipeline/rag_pipeline.py
Updated to use: Interface-based dependency injection

Design Pattern: Facade Pattern
SOLID Principles:
- Single Responsibility: Orchestrates the RAG pipeline
- Dependency Inversion: Depends only on interfaces
- Open/Closed: Can extend without modification
"""

import time
import logging
from typing import Dict, List, Any, Optional

from core.domain.response import RAGResponse
from core.domain.metrics import PerformanceMetrics
from core.domain.document import Document

from core.interfaces.llm_provider import LLMProvider
from core.interfaces.embedder import Embedder
from core.interfaces.reranker import Reranker
from core.interfaces.vector_store import VectorStore

from components.retrieval.retriever import Retriever
from components.filters.filter_manager import FilterManager


class RAGPipeline:
    """
    Facade for the complete RAG system.
    
    Orchestrates the 6-step RAG process:
    1. Filter extraction (using LLM)
    2. Query embedding (using Embedder)
    3. Retrieval (using Retriever with strategy)
    4. Reranking (using Reranker, optional)
    5. Answer generation (using LLM)
    6. Response assembly (creating RAGResponse)
    
    Example usage:
        pipeline = RAGPipeline(
            llm=llm_provider,
            embedder=embedder,
            retriever=retriever,
            reranker=reranker,
            vector_store=vector_store
        )
        
        response = pipeline.query(
            "What are positive economic policies?",
            top_k=5
        )
        
        print(response.answer)
        print(f"Found {len(response.sources)} sources")
    """
    
    def __init__(
        self,
        llm: LLMProvider,
        embedder: Embedder,
        retriever: Retriever,
        reranker: Optional[Reranker],
        vector_store: VectorStore
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            llm: LLM provider for filter extraction and answer generation
            embedder: Embedder for converting queries to vectors
            retriever: Retriever with configured strategy
            reranker: Optional reranker for improving result ranking
            vector_store: Vector store (for cleanup/management)
        """
        # Validate interfaces
        if not isinstance(llm, LLMProvider):
            raise TypeError(f"llm must implement LLMProvider, got {type(llm)}")
        if not isinstance(embedder, Embedder):
            raise TypeError(f"embedder must implement Embedder, got {type(embedder)}")
        if not isinstance(retriever, Retriever):
            raise TypeError(f"retriever must be Retriever instance, got {type(retriever)}")
        if reranker is not None and not isinstance(reranker, Reranker):
            raise TypeError(f"reranker must implement Reranker, got {type(reranker)}")
        if not isinstance(vector_store, VectorStore):
            raise TypeError(f"vector_store must implement VectorStore, got {type(vector_store)}")
        
        # Store components
        self._llm = llm
        self._embedder = embedder
        self._retriever = retriever
        self._reranker = reranker
        self._vector_store = vector_store
        
        # Initialize filter manager
        self._filter_manager = FilterManager(llm=llm)
        
        # Logger
        self._logger = logging.getLogger(__name__)
        self._logger.info("RAG Pipeline initialized successfully")
    
    def query(
        self,
        text: str,
        top_k: int = 5,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> RAGResponse:
        """
        Execute the complete RAG pipeline.
        
        This is the main public method - the simple interface to the
        complex RAG subsystem (Facade Pattern).
        
        Args:
            text: User's query text
            top_k: Number of results to return
            conversation_history: Previous conversation turns (optional)
            
        Returns:
            RAGResponse with answer, sources, filters, and metrics
            
        Raises:
            ValueError: If query text is empty
            RuntimeError: If pipeline execution fails
        """
        # Validation
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty")
        
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")
        
        start_time = time.perf_counter()
        
        try:
            # Initialize metrics
            metrics = PerformanceMetrics()
            
            # Step 1: Extract filters
            self._logger.info(f"Processing query: '{text[:50]}...'")
            filter_start = time.perf_counter()
            
            filters = self._filter_manager.extract_and_update_filters(
                query=text,
                conversation_history=conversation_history
            )
            
            metrics.filter_extraction_time_ms = (time.perf_counter() - filter_start) * 1000
            self._logger.debug(f"Extracted filters: {filters}")
            
            # Step 2: Embed query
            embedding_start = time.perf_counter()
            
            query_vector = self._embedder.embed_query(text)
            
            metrics.embedding_time_ms = (time.perf_counter() - embedding_start) * 1000
            self._logger.debug(f"Embedded query (dimension: {len(query_vector)})")
            
            # Step 3: Retrieve candidates
            retrieval_start = time.perf_counter()
            
            candidates = self._retriever.retrieve(
                query_vector=query_vector,
                query_text=text,
                top_k=top_k * 2,  # Retrieve more for reranking
                filters=filters
            )
            
            metrics.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000
            self._logger.info(f"Retrieved {len(candidates)} candidates")
            
            # Step 4: Rerank (optional)
            if self._reranker and candidates:
                rerank_start = time.perf_counter()
                
                candidates = self._reranker.rerank(
                    query=text,
                    candidates=candidates,
                    top_k=top_k
                )
                
                metrics.rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
                self._logger.info(f"Reranked to top {len(candidates)} results")
            else:
                # No reranking - just take top_k
                candidates = candidates[:top_k]
                metrics.rerank_time_ms = 0.0
            
            # Convert candidates to Document objects
            sources = self._convert_to_documents(candidates)
            
            # Step 5: Generate answer
            generation_start = time.perf_counter()
            
            answer = self._generate_answer(
                query=text,
                sources=sources,
                filters=filters
            )
            
            metrics.generation_time_ms = (time.perf_counter() - generation_start) * 1000
            self._logger.debug(f"Generated answer ({len(answer)} chars)")
            
            # Step 6: Assemble response
            metrics.total_time_ms = (time.perf_counter() - start_time) * 1000
            
            response = RAGResponse(
                answer=answer,
                sources=sources,
                filters_applied=filters,
                metrics=metrics
            )
            
            self._logger.info(
                f"Query completed - Results: {len(sources)}, "
                f"Time: {metrics.total_time_ms:.1f}ms"
            )
            
            return response
            
        except Exception as e:
            self._logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            
            # Return error response
            return self._create_error_response(
                query=text,
                error=str(e),
                elapsed_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _generate_answer(
        self,
        query: str,
        sources: List[Document],
        filters: Dict[str, Any]
    ) -> str:
        """
        Generate answer using LLM based on retrieved sources.
        
        Args:
            query: User's query
            sources: Retrieved documents
            filters: Active filters
            
        Returns:
            Generated answer text
        """
        if not sources:
            return (
                "I couldn't find any relevant statements matching your query. "
                "Try adjusting your filters or rephrasing your question."
            )
        
        # Build context from sources
        context_parts = []
        for i, doc in enumerate(sources[:5], 1):  # Use top 5 for context
            metadata_str = ", ".join([
                f"{k}: {v}" for k, v in doc.metadata.items()
                if k in ["politician", "theme", "sentiment"]
            ])
            context_parts.append(
                f"[Source {i}] ({metadata_str})\n{doc.content}"
            )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt
        filter_info = ""
        if filters:
            filter_items = [f"{k}: {v}" for k, v in filters.items()]
            filter_info = f"\nActive filters: {', '.join(filter_items)}"
        
        prompt = f"""Based on the following political statements, please answer the user's question.

Question: {query}{filter_info}

Retrieved Statements:
{context}

Instructions:
1. Provide a clear, factual answer based on the statements above
2. Cite specific statements when making claims
3. If the statements don't fully answer the question, acknowledge limitations
4. Keep your response concise but informative (2-3 paragraphs)

Answer:"""
        
        try:
            answer = self._llm.generate(prompt)
            return answer.strip()
        
        except Exception as e:
            self._logger.error(f"Answer generation failed: {e}")
            return (
                "I found relevant statements but encountered an error generating "
                "a comprehensive answer. Please try again."
            )
    
    def _convert_to_documents(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Convert raw candidate dictionaries to Document objects.
        
        Args:
            candidates: List of candidate dictionaries from retrieval
            
        Returns:
            List of Document objects
        """
        documents = []
        
        for candidate in candidates:
            # Extract required fields
            content = candidate.get("content") or candidate.get("statement", "")
            metadata = candidate.get("metadata", {})
            
            # Extract scores
            score = candidate.get("score", 0.0)
            rerank_score = candidate.get("rerank_score")
            
            # Create Document
            doc = Document(
                content=content,
                metadata=metadata,
                score=score,
                embedding_index=candidate.get("embedding_index"),
                rerank_score=rerank_score
            )
            
            documents.append(doc)
        
        return documents
    
    def _create_error_response(
        self,
        query: str,
        error: str,
        elapsed_ms: float
    ) -> RAGResponse:
        """
        Create an error response.
        
        Args:
            query: Original query
            error: Error message
            elapsed_ms: Time elapsed before error
            
        Returns:
            RAGResponse with error information
        """
        metrics = PerformanceMetrics(total_time_ms=elapsed_ms)
        
        return RAGResponse(
            answer=f"An error occurred while processing your query: {error}",
            sources=[],
            filters_applied={},
            metrics=metrics
        )
    
    def get_filter_manager(self) -> FilterManager:
        """
        Get the filter manager (for advanced usage).
        
        Returns:
            FilterManager instance
        """
        return self._filter_manager
    
    def get_retriever(self) -> Retriever:
        """
        Get the retriever (for strategy switching).
        
        Returns:
            Retriever instance
        """
        return self._retriever
    
    def close(self) -> None:
        """
        Close the pipeline and clean up resources.
        """
        try:
            self._vector_store.close()
            self._logger.info("RAG Pipeline closed successfully")
        except Exception as e:
            self._logger.error(f"Error closing pipeline: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        strategy_name = self._retriever.get_current_strategy().get_strategy_name()
        has_reranker = "with reranker" if self._reranker else "without reranker"
        return f"RAGPipeline(strategy={strategy_name}, {has_reranker})"