"""
Free LLM service using Ollama and HuggingFace transformers.
No paid APIs required - fully open source solution.
Fixed with better error handling and provider management.
"""

import asyncio
import logging
import requests
import json
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response from LLM."""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if LLM is available."""
        pass


class OllamaProvider(BaseLLMProvider):
    """
    Ollama LLM provider - completely free local LLM.
    Supports Llama2, Mistral, CodeLlama, and many others.
    """
    
    def __init__(self, host: str = None, model: str = None):
        self.host = host or settings.OLLAMA_HOST
        self.model = model or settings.LLM_MODEL
        self.session = requests.Session()
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        self._is_healthy = None
        
    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using Ollama."""
        try:
            url = f"{self.host}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": settings.LLM_TEMPERATURE,
                    "num_predict": max_tokens or settings.MAX_TOKENS,
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            # Use asyncio to run the blocking request
            loop = asyncio.get_event_loop()
            headers = {"Content-Type": "application/json"}
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: self.session.post(url, json=payload, headers=headers,timeout=120)
                ),
                timeout=150.0  # Slightly longer than request timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except asyncio.TimeoutError:
            logger.error("Ollama generation timed out")
            raise Exception("Ollama generation timed out")
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            self._is_healthy = False  # Mark as unhealthy on error
            raise
    
    async def is_available(self) -> bool:
        """Check if Ollama is running and model is available."""
        current_time = time.time()
        
        # Use cached result if recent
        if (self._is_healthy is not None and 
            current_time - self._last_health_check < self._health_check_interval):
            return self._is_healthy
        
        try:
            # Check if Ollama is running
            url = f"{self.host}/api/tags"
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.session.get(url, timeout=10)
                ),
                timeout=15.0
            )
            
            if response.status_code != 200:
                self._is_healthy = False
                return False
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            is_available = any(self.model in name for name in model_names)
            
            # Update cache
            self._is_healthy = is_available
            self._last_health_check = current_time
            
            return is_available
            
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            self._is_healthy = False
            self._last_health_check = current_time
            return False


class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace transformers provider - free local inference.
    Uses smaller models that can run on CPU.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "microsoft/DialoGPT-medium"
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self._initialized = False
        self._initialization_error = None
        
    async def _initialize(self):
        """Initialize the model (only once)."""
        if self._initialized:
            return
        
        if self._initialization_error:
            raise self._initialization_error
            
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            
            # Use CPU for compatibility (can be changed to GPU if available)
            device = 0 if torch.cuda.is_available() else -1
            
            # Initialize text generation pipeline
            loop = asyncio.get_event_loop()
            self.pipeline = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        "text-generation",
                        model=self.model_name,
                        device=device,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True  # Allow custom model code
                    )
                ),
                timeout=300.0  # 5 minutes for model loading
            )
            
            self._initialized = True
            logger.info("HuggingFace model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize HuggingFace model: {e}"
            logger.error(error_msg)
            self._initialization_error = Exception(error_msg)
            raise self._initialization_error
    
    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using HuggingFace transformers."""
        await self._initialize()
        
        try:
            max_length = min((max_tokens or settings.MAX_TOKENS) + len(prompt.split()), 1024)
            
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.pipeline(
                        prompt,
                        max_length=max_length,
                        temperature=settings.LLM_TEMPERATURE,
                        do_sample=True,
                        pad_token_id=self.pipeline.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        truncation=True
                    )
                ),
                timeout=60.0  # 1 minute timeout for generation
            )
            
            # Extract generated text (remove the input prompt)
            generated_text = result[0]["generated_text"]
            response = generated_text[len(prompt):].strip()
            
            # Fallback if response is empty
            if not response:
                return "I understand your question, but I'm unable to generate a meaningful response at this time."
            
            return response
            
        except asyncio.TimeoutError:
            logger.error("HuggingFace generation timed out")
            return "Response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"HuggingFace generation error: {e}")
            return "I encountered an error generating a response. Please try again."
    
    async def is_available(self) -> bool:
        """Check if HuggingFace model can be loaded."""
        try:
            await asyncio.wait_for(self._initialize(), timeout=10.0)
            return self._initialized
        except Exception:
            return False


class FreeLLMService:
    """
    Service that manages free LLM providers with fallback support.
    """
    
    def __init__(self):
        self.providers = []
        self.current_provider = None
        self.provider_last_check = {}
        self.provider_check_interval = 300  # 5 minutes
        self._setup_providers()
        
    def _setup_providers(self):
        """Setup available LLM providers in order of preference."""
        
        # Add Ollama provider (best performance, requires Ollama running)
        self.providers.append(OllamaProvider())
        
        # Add HuggingFace provider (fallback, always works but slower)
        self.providers.append(HuggingFaceProvider())
        
    async def _find_available_provider(self) -> Optional[BaseLLMProvider]:
        """Find the first available provider."""
        current_time = time.time()
        
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            
            # Skip recently failed providers (except if it's been a while)
            last_check = self.provider_last_check.get(provider_name, 0)
            if current_time - last_check < 60:  # 1 minute cooldown
                continue
            
            try:
                is_available = await asyncio.wait_for(
                    provider.is_available(),
                    timeout=30.0  # 30 seconds timeout for availability check
                )
                
                self.provider_last_check[provider_name] = current_time
                
                if is_available:
                    logger.info(f"Using LLM provider: {provider_name}")
                    return provider
                else:
                    logger.info(f"Provider {provider_name} is not available")
                    
            except Exception as e:
                logger.warning(f"Provider {provider_name} availability check failed: {e}")
                self.provider_last_check[provider_name] = current_time
                continue
        
        return None
    
    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """
        Generate response using the first available provider.
        """
        # Validate input
        if not prompt or not prompt.strip():
            return "I need a question or prompt to respond to."
        
        # Truncate very long prompts
        if len(prompt) > 8000:
            prompt = prompt[:8000] + "..."
            logger.warning("Prompt truncated due to length")
        
        # Check current provider first
        if self.current_provider:
            try:
                return await self.current_provider.generate_response(prompt, max_tokens)
            except Exception as e:
                logger.error(f"Current provider {self.current_provider.__class__.__name__} failed: {e}")
                self.current_provider = None
        
        # Find an available provider
        self.current_provider = await self._find_available_provider()
        
        if not self.current_provider:
            error_msg = """No LLM providers are available. Please ensure:
1. Ollama is installed and running with models downloaded, OR
2. System has sufficient resources for HuggingFace models

To install Ollama: visit https://ollama.ai
To download models: run 'ollama pull llama2'"""
            
            logger.error("No LLM providers available")
            raise Exception(error_msg)
        
        # Try the new provider
        try:
            return await self.current_provider.generate_response(prompt, max_tokens)
        except Exception as e:
            logger.error(f"All providers failed. Last error: {e}")
            self.current_provider = None
            
            # Return a helpful error message instead of raising
            return f"I'm experiencing technical difficulties with the language models. Please try again in a few moments. Error: {str(e)[:100]}"


class FreeEmbeddingService:
    """
    Free embedding service using sentence-transformers.
    No API keys required.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._initialized = False
        self._initialization_error = None
        
    async def _initialize(self):
        """Initialize the embedding model."""
        if self._initialized:
            return
        
        if self._initialization_error:
            raise self._initialization_error
            
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            loop = asyncio.get_event_loop()
            self.model = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(self.model_name)
                ),
                timeout=300.0  # 5 minutes for model loading
            )
            
            self._initialized = True
            logger.info("Embedding model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize embedding model: {e}"
            logger.error(error_msg)
            self._initialization_error = Exception(error_msg)
            raise self._initialization_error
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""
        await self._initialize()
        
        try:
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000]
            
            loop = asyncio.get_event_loop()
            embedding = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.model.encode(text)
                ),
                timeout=30.0  # 30 seconds timeout
            )
            
            return embedding.tolist()
            
        except asyncio.TimeoutError:
            logger.error("Embedding generation timed out")
            raise Exception("Embedding generation timed out")
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        await self._initialize()
        
        try:
            # Truncate very long texts and limit batch size
            processed_texts = []
            for text in texts[:100]:  # Limit to 100 texts per batch
                if len(text) > 8000:
                    text = text[:8000]
                processed_texts.append(text)
            
            loop = asyncio.get_event_loop()
            embeddings = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.model.encode(processed_texts)
                ),
                timeout=min(60.0, len(processed_texts) * 2.0)  # Dynamic timeout
            )
            
            return embeddings.tolist()
            
        except asyncio.TimeoutError:
            logger.error("Batch embedding generation timed out")
            raise Exception("Batch embedding generation timed out")
        except Exception as e:
            logger.error(f"Batch embedding generation error: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if embedding model is available."""
        try:
            await asyncio.wait_for(self._initialize(), timeout=10.0)
            return self._initialized
        except Exception:
            return False


# Global instances
free_llm_service = FreeLLMService()
free_embedding_service = FreeEmbeddingService()