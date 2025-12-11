"""
Ollama Adapter - LLM Provider Implementation
Implements LLMProvider interface for local Ollama models
FIXED: Removes strict parameter validation that caused errors
"""

import logging
import json
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.interfaces.llm_provider import LLMProvider
from infrastructure.config.settings import get_settings

# Try to import the NEW langchain-ollama package (recommended)
try:
    from langchain_ollama import OllamaLLM
    LANGCHAIN_OLLAMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_OLLAMA_AVAILABLE = False
    logging.warning("langchain-ollama not available. Run: pip install -U langchain-ollama")

# Fallback to old package if new one not available
if not LANGCHAIN_OLLAMA_AVAILABLE:
    try:
        from langchain_community.llms import Ollama as OllamaLLM
        LANGCHAIN_OLLAMA_AVAILABLE = True
        logging.warning("Using deprecated langchain_community.llms.Ollama. Please upgrade to langchain-ollama")
    except ImportError:
        pass

# Direct ollama client as final fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class OllamaAdapter(LLMProvider):
    """
    Adapter for Ollama LLM models
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None, 
        temperature: Optional[float] = None,
        timeout: Optional[int] = None
    ):
        """
        Initialize Ollama adapter
        """
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # Use configuration structure
        self.model_name = model_name or self.settings.ollama_model
        self.base_url = base_url or self.settings.ollama_base_url
        self.temperature = temperature or self.settings.ollama_temperature
        self.timeout = timeout or self.settings.ollama_timeout
        
        # Max tokens default
        self.max_tokens = 2048
        
        # Client initialization
        self._client = None
        self._ollama_client = None
        self._is_available = None
        
        self.logger.info(f"Initializing OllamaAdapter with model: {self.model_name}")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the Ollama client"""
        try:
            if LANGCHAIN_OLLAMA_AVAILABLE:
                # Use LangChain wrapper
                self._client = OllamaLLM(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url=self.base_url,
                    timeout=self.timeout
                )
                self.logger.info("Initialized with langchain-ollama (recommended)")
                
            elif OLLAMA_AVAILABLE:
                # Use direct ollama client
                self._ollama_client = ollama.Client(host=self.base_url)
                self.logger.info("Initialized with direct ollama client")
                
            else:
                raise RuntimeError("No Ollama client available. Install: pip install -U langchain-ollama ollama")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama: {e}")
            self._is_available = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt
        """
        if not self.is_available():
            raise RuntimeError("Ollama is not available")
        
        try:
            if self._client:
                # Use LangChain client
                # FIXED: Simple invoke without extra kwargs to avoid errors
                response = self._client.invoke(prompt)
                return response.strip()
                
            elif self._ollama_client:
                # Use direct client
                # For direct client, we can be more specific
                options = {'temperature': kwargs.get('temperature', self.temperature)}
                if 'max_tokens' in kwargs:
                    options['num_predict'] = kwargs['max_tokens']
                    
                response = self._ollama_client.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )
                return response['response'].strip()
            
            else:
                raise RuntimeError("No Ollama client initialized")
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate structured JSON output matching schema
        """
        if not self.is_available():
            raise RuntimeError("Ollama is not available")
        
        try:
            # Add JSON formatting instruction to prompt
            json_prompt = f"""{prompt}

You must respond with ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Response (JSON only, no explanation):"""
            
            # Generate response
            response_text = self.generate(json_prompt)
            
            # Extract JSON from response
            response_text = response_text.strip()
            
            # Try to find JSON in response
            if '{' in response_text:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = response_text
            
            # Parse JSON
            try:
                result = json.loads(json_str)
                self.logger.debug(f"Parsed structured output: {result}")
                return result
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from Ollama response: {e}")
                self.logger.error(f"Response was: {response_text[:500]}")
                raise ValueError(f"Invalid JSON output: {e}")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            self.logger.error(f"Structured generation failed: {e}")
            raise RuntimeError(f"Ollama structured generation failed: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama is available and the model is loaded
        """
        if self._is_available is not None:
            return self._is_available
        
        try:
            # Try a simple generation
            if self._client:
                # Test with LangChain client
                # FIXED: Simple invoke, no args
                test_response = self._client.invoke("Hi")
                self._is_available = bool(test_response)
                
            elif self._ollama_client:
                # Test with direct client
                test_response = self._ollama_client.generate(
                    model=self.model_name,
                    prompt="Hi"
                )
                self._is_available = bool(test_response.get('response'))
                
            else:
                self._is_available = False
            
            if self._is_available:
                self.logger.info(f"Ollama model {self.model_name} is available")
            else:
                self.logger.warning(f"Ollama model {self.model_name} is not responding")
                
            return self._is_available
            
        except Exception as e:
            self.logger.error(f"Availability check failed: {e}")
            self._is_available = False
            return False
    
    def __repr__(self) -> str:
        return f"OllamaAdapter(model={self.model_name}, available={self.is_available()})"