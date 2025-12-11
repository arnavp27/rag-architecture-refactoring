"""
Settings - Configuration management with Pydantic

This module manages all configuration for the RAG system using Pydantic
for validation and type safety. Configuration is loaded from environment
variables with sensible defaults.

Design Pattern: Singleton (via get_settings function)
SOLID Principle: Single Responsibility - only manages configuration
"""

from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with validation and type safety.
    
    All settings are loaded from environment variables with defaults.
    Use .env file for local development.
    
    Configuration sections:
    1. Google Gemini API
    2. Weaviate Vector Database
    3. Ollama Configuration
    4. Model Configuration
    5. System Configuration
    """
    
    # ===== Google Gemini API =====
    google_api_key: Optional[str] = Field(
        default=None,
        description="Google Gemini API key for LLM generation"
    )
    gemini_model: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model name"
    )
    gemini_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for Gemini generation"
    )
    gemini_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum tokens for Gemini generation"
    )
    
    # ===== Weaviate Vector Database =====
    weaviate_host: str = Field(
        default="localhost",
        description="Weaviate server host"
    )
    weaviate_port: int = Field(
        default=8080,
        gt=0,
        lt=65536,
        description="Weaviate server port"
    )
    weaviate_scheme: str = Field(
        default="http",
        description="Weaviate connection scheme (http/https)"
    )
    weaviate_collection: str = Field(
        default="PoliticalStatements",
        description="Weaviate collection name"
    )
    
    # ===== Ollama Configuration =====
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    ollama_model: str = Field(
        default="gemma:2b",
        description="Ollama model name"
    )
    ollama_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Temperature for Ollama generation"
    )
    ollama_timeout: int = Field(
        default=120,
        gt=0,
        description="Ollama request timeout in seconds"
    )
    
    # ===== Model Configuration =====
    embedder_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model name"
    )
    embedder_device: str = Field(
        default="cpu",
        description="Device for embedder (cpu/cuda)"
    )
    
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="CrossEncoder reranker model name"
    )
    reranker_device: str = Field(
        default="cpu",
        description="Device for reranker (cpu/cuda)"
    )
    
    # ===== System Configuration =====
    cache_size: int = Field(
        default=100,
        gt=0,
        description="Cache size for retrieval results"
    )
    top_k_results: int = Field(
        default=5,
        gt=0,
        le=100,
        description="Default number of results to return"
    )
    top_k_retrieval: int = Field(
        default=20,
        gt=0,
        le=100,
        description="Number of candidates to retrieve before reranking"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    # ===== API Retry Configuration =====
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of API retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        ge=0.0,
        description="Initial retry delay in seconds (exponential backoff)"
    )
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    def get_weaviate_url(self) -> str:
        """
        Get the full Weaviate connection URL.
        
        Returns:
            str: Full Weaviate URL (e.g., "http://localhost:8080")
        """
        return f"{self.weaviate_scheme}://{self.weaviate_host}:{self.weaviate_port}"
    
    def is_cuda_available(self) -> bool:
        """
        Check if CUDA is requested and available.
        
        Returns:
            bool: True if CUDA should be used
        """
        if self.embedder_device == "cuda" or self.reranker_device == "cuda":
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
        return False
    
    def validate_required_settings(self) -> List[str]:
        """
        Validate that required settings are present.
        
        Returns:
            List[str]: List of missing required settings (empty if all present)
        """
        missing = []
        
        # Check for at least one LLM provider
        if not self.google_api_key and not self.ollama_base_url:
            missing.append("At least one LLM provider (google_api_key or ollama_base_url)")
        
        return missing
    
    def __str__(self) -> str:
        """String representation (hides sensitive data)"""
        return (
            f"Settings(weaviate={self.get_weaviate_url()}, "
            f"gemini_configured={bool(self.google_api_key)}, "
            f"ollama_url={self.ollama_base_url})"
        )


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings (singleton pattern).
    
    Settings are loaded once and cached. This ensures consistent
    configuration throughout the application.
    
    Returns:
        Settings: Application settings instance
        
    Example:
        >>> settings = get_settings()
        >>> print(settings.weaviate_host)
        localhost
    """
    return Settings()


# Convenience function to validate settings on import
def validate_settings() -> None:
    """
    Validate settings and print warnings for missing required settings.
    
    This can be called at application startup to ensure configuration is valid.
    """
    settings = get_settings()
    missing = settings.validate_required_settings()
    
    if missing:
        import warnings
        warnings.warn(
            f"Missing required settings: {', '.join(missing)}. "
            "The application may not function correctly."
        )