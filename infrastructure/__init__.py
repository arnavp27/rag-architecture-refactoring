"""
Infrastructure Package - External Integrations

This is Layer 3 of the architecture, containing adapters that connect
external services (APIs, databases, models) to our core interfaces.

Architecture Principle:
    Infrastructure depends on Core interfaces, never the other way around.
    This implements the Dependency Inversion Principle.

Import Structure:
    from infrastructure.adapters import GeminiAdapter, OllamaAdapter, WeaviateAdapter
    from infrastructure.config import get_settings
"""

# Adapters
from infrastructure.adapters import (
    GeminiAdapter,
    OllamaAdapter,
    WeaviateAdapter,
    SentenceTransformerEmbedder,
    CrossEncoderReranker
)

# Configuration
from infrastructure.config import (
    Settings,
    get_settings,
    validate_settings
)

__all__ = [
    # Adapters
    "GeminiAdapter",
    "OllamaAdapter",
    "WeaviateAdapter",
    "SentenceTransformerEmbedder",
    "CrossEncoderReranker",
    
    # Configuration
    "Settings",
    "get_settings",
    "validate_settings"
]

__version__ = "1.0.0"