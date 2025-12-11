"""
ModelFactory - Factory for creating embedders and rerankers

This factory centralizes creation of ML model components:
- Embedders (convert text to vectors)
- Rerankers (re-score retrieved documents)

Design Pattern: Factory Pattern
SOLID Principles:
- Single Responsibility: Only creates model instances
- Open/Closed: New models can be added by extending factory methods
- Dependency Inversion: Returns interface types, not concrete classes
"""

from typing import Optional
import logging

from core.interfaces.embedder import Embedder
from core.interfaces.reranker import Reranker
from infrastructure.adapters.sentence_transformer_embedder import SentenceTransformerEmbedder
from infrastructure.adapters.cross_encoder_reranker import CrossEncoderReranker
from infrastructure.config.settings import Settings


class ModelFactory:
    """
    Factory for creating embedder and reranker instances.
    
    Supports:
    - SentenceTransformer embedders (Hugging Face models)
    - CrossEncoder rerankers (Hugging Face models)
    """
    
    @staticmethod
    def create_embedder(
        model_type: str = "sentence-transformers",
        settings: Optional[Settings] = None
    ) -> Embedder:
        """
        Create an embedder instance.
        """
        logger = logging.getLogger(__name__)
        
        # Use default settings if not provided
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        if model_type.lower() == "sentence-transformers":
            logger.info(f"Creating SentenceTransformerEmbedder with model: {settings.embedder_model}")
            # FIX: Use settings.embedder_device and remove batch_size (not in Adapter __init__)
            return SentenceTransformerEmbedder(
                model_name=settings.embedder_model,
                device=settings.embedder_device
            )
        
        else:
            raise ValueError(
                f"Unknown embedder type: {model_type}. "
                f"Supported types: 'sentence-transformers'"
            )
    
    @staticmethod
    def create_reranker(
        model_type: str = "cross-encoder",
        settings: Optional[Settings] = None
    ) -> Reranker:
        """
        Create a reranker instance.
        """
        logger = logging.getLogger(__name__)
        
        # Use default settings if not provided
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        if model_type.lower() == "cross-encoder":
            logger.info(f"Creating CrossEncoderReranker with model: {settings.reranker_model}")
            # FIX: Use settings.reranker_device and remove batch_size (not in Adapter __init__)
            return CrossEncoderReranker(
                model_name=settings.reranker_model,
                device=settings.reranker_device
            )
        
        else:
            raise ValueError(
                f"Unknown reranker type: {model_type}. "
                f"Supported types: 'cross-encoder'"
            )
    
    @staticmethod
    def create_sentence_transformer_embedder(
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> Embedder:
        """
        Create a SentenceTransformer embedder directly.
        """
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        if model_name is None:
            model_name = settings.embedder_model
        
        # FIX: Use settings.embedder_device and remove batch_size
        return SentenceTransformerEmbedder(
            model_name=model_name,
            device=settings.embedder_device
        )
    
    @staticmethod
    def create_cross_encoder_reranker(
        model_name: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> Reranker:
        """
        Create a CrossEncoder reranker directly.
        """
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        if model_name is None:
            model_name = settings.reranker_model
        
        # FIX: Use settings.reranker_device and remove batch_size
        return CrossEncoderReranker(
            model_name=model_name,
            device=settings.reranker_device
        )