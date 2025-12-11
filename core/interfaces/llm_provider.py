"""
LLMProvider ABC - Abstract interface for Language Model providers

This interface ensures that all LLM implementations (Gemini, Ollama, OpenAI, etc.) 
follow the same contract, enabling the Dependency Inversion Principle (DIP).

The high-level RAGPipeline depends on this abstraction, not on concrete implementations.
This allows us to swap LLM providers at runtime without changing any high-level code.

Design Pattern: Adapter Pattern (this is the target interface)
SOLID Principle: Dependency Inversion Principle (DIP)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class LLMProvider(ABC):
    """
    Abstract interface for Language Model providers.
    
    All LLM implementations MUST implement this interface to ensure
    they can be used interchangeably in the RAG pipeline.
    
    This interface supports:
    - Text generation (general purpose)
    - Structured output generation (for filter extraction, JSON responses)
    - Availability checking (for fallback mechanisms)
    """
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt.
        
        This is the primary method for LLM text generation. It takes a prompt
        and returns the generated text response.
        
        Args:
            prompt (str): The input prompt/question for the LLM
            **kwargs: Additional provider-specific parameters:
                - temperature (float): Sampling temperature (0.0 to 1.0)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter
                - stop_sequences (List[str]): Sequences that stop generation
                
        Returns:
            str: Generated text response from the LLM
            
        Raises:
            RuntimeError: If generation fails due to API errors, rate limits, etc.
            ValueError: If prompt is empty or invalid
            
        Example:
            >>> llm = GeminiAdapter(api_key="...")
            >>> response = llm.generate("What is the capital of France?")
            >>> print(response)
            "The capital of France is Paris."
        """
        pass
    
    @abstractmethod
    def generate_structured(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) matching the provided schema.
        
        This method is used when we need structured data from the LLM,
        such as filter extraction, metadata generation, or formatted responses.
        
        The LLM must return a dictionary that conforms to the provided schema.
        
        Args:
            prompt (str): The input prompt requesting structured output
            schema (Dict[str, Any]): JSON schema defining expected output structure
                Example: {"type": "object", "properties": {"theme": {"type": "array"}}}
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Dict[str, Any]: Dictionary matching the provided schema
            
        Raises:
            ValueError: If output doesn't match schema or is invalid JSON
            RuntimeError: If generation fails
            
        Example:
            >>> schema = {
            ...     "type": "object",
            ...     "properties": {
            ...         "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            ...         "topics": {"type": "array", "items": {"type": "string"}}
            ...     }
            ... }
            >>> result = llm.generate_structured("Analyze: 'Great product!'", schema)
            >>> print(result)
            {"sentiment": "positive", "topics": ["product", "review"]}
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is currently available and functional.
        
        This method is used by the LLMFactory to implement fallback logic.
        If the primary LLM provider is unavailable (API down, rate limited, 
        no API key, etc.), the factory can fall back to an alternative provider.
        
        This supports the reliability and fault-tolerance of the system.
        
        Returns:
            bool: True if provider is available and can process requests,
                  False if provider is unavailable or non-functional
                  
        Example:
            >>> primary_llm = GeminiAdapter(api_key="invalid")
            >>> if not primary_llm.is_available():
            ...     fallback_llm = OllamaAdapter()
            ...     llm = fallback_llm if fallback_llm.is_available() else None
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Optional method that can be overridden to provide metadata
        about the model being used.
        
        Returns:
            Dict[str, Any]: Model metadata (name, version, parameters, etc.)
            
        Note:
            This is NOT an abstract method - implementations can optionally override it.
        """
        return {
            "provider": self.__class__.__name__,
            "available": self.is_available()
        }


# Type alias for better code readability
LLM = LLMProvider