"""
Infrastructure Adapters Package

This package contains all adapters that connect external services
to our core interfaces, implementing the Adapter Pattern.

Available Adapters:
- GeminiAdapter: Google Gemini API → LLMProvider
- OllamaAdapter: Ollama API → LLMProvider
- WeaviateAdapter: Weaviate DB → VectorStore
- SentenceTransformerEmbedder: SentenceTransformers → Embedder
- CrossEncoderReranker: CrossEncoder → Reranker
"""

from infrastructure.adapters.gemini_adapter import GeminiAdapter
from infrastructure.adapters.ollama_adapter import OllamaAdapter
from infrastructure.adapters.weaviate_adapter import WeaviateAdapter
from infrastructure.adapters.sentence_transformer_embedder import SentenceTransformerEmbedder
from infrastructure.adapters.cross_encoder_reranker import CrossEncoderReranker

__all__ = [
    "GeminiAdapter",
    "OllamaAdapter",
    "WeaviateAdapter",
    "SentenceTransformerEmbedder",
    "CrossEncoderReranker"
]