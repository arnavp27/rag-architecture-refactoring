"""
LLMFactory - Factory for creating LLM providers with fallback logic

This factory centralizes LLM provider creation and implements smart fallback:
- Try primary provider first
- Fall back to secondary if primary is unavailable
- Provides graceful degradation

Design Pattern: Factory Pattern
SOLID Principles:
- Single Responsibility: Only creates LLM providers
- Open/Closed: New providers can be added by extending factory methods
- Dependency Inversion: Returns interface types, not concrete classes
"""

from typing import Optional
import logging

from core.interfaces.llm_provider import LLMProvider
from infrastructure.adapters.gemini_adapter import GeminiAdapter
from infrastructure.adapters.ollama_adapter import OllamaAdapter
from infrastructure.config.settings import Settings


class LLMFactory:
    """
    Factory for creating LLM provider instances with fallback logic.
    
    Supports:
    - Gemini (Google's Generative AI)
    - Ollama (Local LLM inference)
    
    The factory tries the primary provider first, then falls back to
    secondary if primary is unavailable.
    """
    
    @staticmethod
    def create_with_fallback(
        primary: str,
        fallback: Optional[str] = None,
        settings: Optional[Settings] = None
    ) -> LLMProvider:
        """
        Create an LLM provider with automatic fallback.
        
        This is the main factory method. It attempts to create and verify
        the primary provider, falling back to the secondary if needed.
        
        Args:
            primary: Primary provider type ("gemini" or "ollama")
            fallback: Fallback provider type (optional)
            settings: Configuration settings (optional, will use default if None)
            
        Returns:
            LLMProvider: An initialized and available LLM provider
            
        Raises:
            RuntimeError: If no providers are available
            ValueError: If provider type is unknown
            
        Example:
            # Try Gemini first, fall back to Ollama
            llm = LLMFactory.create_with_fallback(
                primary="gemini",
                fallback="ollama",
                settings=settings
            )
            
            # If Gemini API is down, automatically uses Ollama
            answer = llm.generate("What is the capital of France?")
        """
        logger = logging.getLogger(__name__)
        
        # Use default settings if not provided
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        # Try primary provider
        logger.info(f"Attempting to create primary LLM provider: {primary}")
        try:
            primary_llm = LLMFactory._create_single(primary, settings)
            
            if primary_llm.is_available():
                logger.info(f"Primary LLM provider '{primary}' is available and ready")
                return primary_llm
            else:
                logger.warning(f"Primary LLM provider '{primary}' is not available")
        
        except Exception as e:
            logger.warning(f"Failed to create primary LLM provider '{primary}': {e}")
        
        # Try fallback if provided
        if fallback:
            logger.info(f"Attempting to create fallback LLM provider: {fallback}")
            try:
                fallback_llm = LLMFactory._create_single(fallback, settings)
                
                if fallback_llm.is_available():
                    logger.info(f"Fallback LLM provider '{fallback}' is available and ready")
                    return fallback_llm
                else:
                    logger.warning(f"Fallback LLM provider '{fallback}' is not available")
            
            except Exception as e:
                logger.error(f"Failed to create fallback LLM provider '{fallback}': {e}")
        
        # No providers available
        error_msg = f"No LLM providers available (tried: {primary}"
        if fallback:
            error_msg += f", {fallback}"
        error_msg += ")"
        
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    @staticmethod
    def _create_single(provider_type: str, settings: Settings) -> LLMProvider:
        """
        Create a single LLM provider instance.
        
        Args:
            provider_type: Type of provider ("gemini" or "ollama")
            settings: Configuration settings
            
        Returns:
            LLMProvider: Initialized provider instance
            
        Raises:
            ValueError: If provider type is unknown
        """
        logger = logging.getLogger(__name__)
        
        if provider_type.lower() == "gemini":
            logger.debug("Creating GeminiAdapter")
            return GeminiAdapter(
                api_key=settings.google_api_key,
                model_name=settings.gemini_model
            )
        
        elif provider_type.lower() == "ollama":
            logger.debug("Creating OllamaAdapter")
            return OllamaAdapter(
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
                timeout=settings.ollama_timeout
            )
        
        else:
            raise ValueError(
                f"Unknown LLM provider type: {provider_type}. "
                f"Supported types: 'gemini', 'ollama'"
            )
    
    @staticmethod
    def create_gemini(settings: Optional[Settings] = None) -> LLMProvider:
        """
        Create a Gemini provider directly (no fallback).
        
        Args:
            settings: Configuration settings (optional)
            
        Returns:
            LLMProvider: Gemini adapter instance
            
        Example:
            llm = LLMFactory.create_gemini()
        """
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        return GeminiAdapter(
            api_key=settings.google_api_key,
            model_name=settings.gemini_model
        )
    
    @staticmethod
    def create_ollama(settings: Optional[Settings] = None) -> LLMProvider:
        """
        Create an Ollama provider directly (no fallback).
        
        Args:
            settings: Configuration settings (optional)
            
        Returns:
            LLMProvider: Ollama adapter instance
            
        Example:
            llm = LLMFactory.create_ollama()
        """
        if settings is None:
            from infrastructure.config.settings import get_settings
            settings = get_settings()
        
        return OllamaAdapter(
            model_name=settings.ollama_model,
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout
        )