"""
Builders - Fluent API for constructing complex objects

This module contains builder classes for pipeline construction:
- RAGPipelineBuilder: Builds RAG pipelines with fluent API

Design Pattern: Builder Pattern
"""

from application.builders.pipeline_builder import RAGPipelineBuilder

__all__ = [
    "RAGPipelineBuilder",
]