"""
Pipeline - Main RAG system facade

This module contains the main RAGPipeline facade that orchestrates
the complete RAG system.

Design Pattern: Facade Pattern
"""

from application.pipeline.rag_pipeline import RAGPipeline

__all__ = [
    "RAGPipeline",
]