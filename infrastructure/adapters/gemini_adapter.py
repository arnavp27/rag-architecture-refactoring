"""
GeminiAdapter - Adapter for Google Gemini API

This adapter implements the LLMProvider interface for Google's Gemini API.
It handles API-specific details, error handling, and retry logic, presenting
a clean interface to the rest of the system.

Design Pattern: Adapter Pattern (adapts Gemini API to LLMProvider interface)
SOLID Principles:
- Single Responsibility: Only handles Gemini API communication
- Dependency Inversion: Implements LLMProvider abstraction
"""

import time
import json
import logging
from typing import Dict, Any, Optional
from core.interfaces import LLMProvider

# Google Generative AI import (external dependency)
try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiAdapter(LLMProvider):
    """
    Adapter for Google Gemini API.
    
    Implements the LLMProvider interface to provide a consistent way
    to interact with Gemini, regardless of its specific API details.
    
    Features:
    - Automatic retry with exponential backoff
    - Error translation to standard exceptions
    - Safety settings configuration
    - Structured output support
    
    Attributes:
        model_name (str): Gemini model name (e.g., "gemini-1.5-flash")
        api_key (str): Google API key
        temperature (float): Sampling temperature (0.0 to 1.0)
        max_tokens (int): Maximum tokens to generate
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key (str): Google API key
            model_name (str): Gemini model name
            temperature (float): Sampling temperature (0.0 to 1.0)
            max_tokens (int): Maximum tokens to generate
            max_retries (int): Maximum retry attempts for API calls
            retry_delay (float): Initial retry delay in seconds
            
        Raises:
            ImportError: If google-generativeai package is not installed
            ValueError: If API key is invalid
        """
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.logger = logging.getLogger(__name__)
        
        # Configure the API
        genai.configure(api_key=api_key)
        
        # Initialize the model with safety settings
        self._model = genai.GenerativeModel(
            model_name=model_name,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        
        self.logger.info(f"GeminiAdapter initialized with model: {model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt.
        
        Implements LLMProvider.generate() with retry logic and error handling.
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional generation parameters:
                - temperature (float): Override default temperature
                - max_tokens (int): Override default max_tokens
                
        Returns:
            str: Generated text response
            
        Raises:
            ValueError: If prompt is empty
            RuntimeError: If generation fails after retries
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Merge kwargs with defaults
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        
        # Generation config
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        # Retry loop with exponential backoff
        for attempt in range(self.max_retries):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract text from response
                if response.text:
                    return response.text.strip()
                else:
                    # Handle blocked content
                    self.logger.warning("Gemini response was blocked or empty")
                    return "I cannot provide a response to that query."
                    
            except Exception as e:
                self.logger.warning(
                    f"Gemini API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)
                else:
                    # Final attempt failed
                    raise RuntimeError(f"Gemini generation failed after {self.max_retries} attempts: {e}")
    
    def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured output (JSON) matching the provided schema.
        
        Implements LLMProvider.generate_structured() by instructing Gemini
        to return JSON and parsing the response.
        
        Args:
            prompt (str): The input prompt
            schema (Dict[str, Any]): JSON schema defining expected output
            **kwargs: Additional generation parameters
            
        Returns:
            Dict[str, Any]: Dictionary matching the schema
            
        Raises:
            ValueError: If output doesn't match schema or is invalid JSON
            RuntimeError: If generation fails
        """
        # Construct prompt with schema instructions
        structured_prompt = f"""
{prompt}

You MUST respond with ONLY valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Do not include any markdown formatting, explanations, or text outside the JSON object.
Respond with only the raw JSON.
"""
        
        # Generate response
        response_text = self.generate(structured_prompt, **kwargs)
        
        # Parse JSON from response
        try:
            # Try to extract JSON if it's wrapped in markdown code blocks
            if "```json" in response_text:
                # Extract content between ```json and ```
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                # Extract content between ``` and ```
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            # Parse JSON
            result = json.loads(response_text)
            
            # Basic schema validation (check that required keys exist)
            if "properties" in schema:
                required_keys = schema.get("required", [])
                for key in required_keys:
                    if key not in result:
                        raise ValueError(f"Missing required key in response: {key}")
            
            return result
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from Gemini response: {e}")
            self.logger.error(f"Response text: {response_text[:200]}")
            raise ValueError(f"Invalid JSON in Gemini response: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Gemini API is available and functional.
        
        Tests connectivity by making a simple API call.
        
        Returns:
            bool: True if Gemini is available, False otherwise
        """
        try:
            # Try a simple generation to test availability
            test_response = self._model.generate_content(
                "Say 'OK'",
                generation_config={"max_output_tokens": 10}
            )
            return bool(test_response.text)
        except Exception as e:
            self.logger.warning(f"Gemini availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Gemini model.
        
        Returns:
            Dict[str, Any]: Model metadata
        """
        return {
            "provider": "Google Gemini",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "available": self.is_available()
        }
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"GeminiAdapter(model={self.model_name}, available={self.is_available()})"