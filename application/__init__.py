"""
Application Layer - High-level orchestration and facades

This is Layer 5 of the 5-layer architecture - the outermost layer that
provides high-level interfaces for using the RAG system.

Layer hierarchy:
- Domain (Layer 1) - Pure data models
- Core Interfaces (Layer 2) - Abstract contracts
- Infrastructure (Layer 3) - External service adapters
- Components (Layer 4) - Business logic & patterns
- Application (Layer 5) - High-level orchestration ← YOU ARE HERE

Design Patterns Implemented:
- Factory Pattern: Centralized object creation with fallback
- Builder Pattern: Fluent API for pipeline construction
- Facade Pattern: Simple interface to complex subsystem
"""

from application.factories import LLMFactory, ModelFactory
from application.builders import RAGPipelineBuilder
from application.pipeline import RAGPipeline

__all__ = [
    # Factories
    "LLMFactory",
    "ModelFactory",
    
    # Builders
    "RAGPipelineBuilder",
    
    # Pipeline
    "RAGPipeline",
]