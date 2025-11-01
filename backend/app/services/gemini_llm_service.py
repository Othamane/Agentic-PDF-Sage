"""
Gemini LLM service using Google's Gemini API.
High-performance replacement for Ollama with better reliability.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import torch

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

class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider - fast and reliable.
    """
    
    def __init__(self, api_key: str = None, model_name: str = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.LLM_MODEL
        self._model = None
        self._initialized = False
        self._last_health_check = 0
        self._health_check_interval = 300  # 5 minutes
        self._is_healthy = None
        
        logger.info(f"Initializing Gemini with model: {self.model_name}")
        
        # Configure Gemini
        if self.api_key:
            genai.configure(api_key=self.api_key)
            logger.info("Gemini API configured successfully")
        else:
            logger.warning("No Gemini API key found")
    
    async def _initialize(self):
        """Initialize the Gemini model."""
        if self._initialized:
            return
            
        try:
            if not self.api_key:
                raise Exception("GEMINI_API_KEY not found in settings")
                
            # Initialize the model
            generation_config = genai.types.GenerationConfig(
                temperature=settings.LLM_TEMPERATURE,
                max_output_tokens=settings.MAX_TOKENS,
                top_p=0.95,
                top_k=40
            )
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            self._initialized = True
            logger.info(f"Gemini model {self.model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    async def generate_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using Gemini API."""
        await self._initialize()
        
        try:
            # Validate and truncate prompt if needed
            if not prompt or not prompt.strip():
                return "I need a question or prompt to respond to."
            
            if len(prompt) > 30000:
                prompt = prompt[:30000] + "..."
                logger.warning("Prompt truncated due to length")
            
            # Generate response
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._model.generate_content(prompt)
                ),
                timeout=30.0
            )
            
            # Extract text from response
            if response.parts:
                return response.text.strip()
            else:
                logger.warning("Gemini returned empty response")
                return "I'm sorry, I couldn't generate a response to that question."
                
        except asyncio.TimeoutError:
            logger.error("Gemini generation timed out")
            return "Response generation timed out. Please try again."
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            self._is_healthy = False
            return f"I encountered an error generating a response: {str(e)[:100]}"
    
    async def is_available(self) -> bool:
        """Check if Gemini API is available."""
        current_time = time.time()
        
        # Use cached result if recent
        if (self._is_healthy is not None and 
            current_time - self._last_health_check < self._health_check_interval):
            return self._is_healthy
        
        try:
            await self._initialize()
            
            # Test with a simple prompt
            test_response = await asyncio.wait_for(
                self.generate_response("Hello", max_tokens=10),
                timeout=15.0
            )
            
            is_available = len(test_response) > 0 and "error" not in test_response.lower()
            
            # Update cache
            self._is_healthy = is_available
            self._last_health_check = current_time
            
            return is_available
            
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            self._is_healthy = False
            self._last_health_check = current_time
            return False
class HuggingFaceProvider(BaseLLMProvider):
    """
    HuggingFace transformers provider - free local inference fallback.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
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
            
            from transformers import pipeline
            
            device = 0 if torch.cuda.is_available() else -1
            
            loop = asyncio.get_event_loop()
            self.pipeline = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: pipeline(
                        "text-generation",
                        model=self.model_name,
                        device=device,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        trust_remote_code=True
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
                timeout=60.0
            )
            
            generated_text = result[0]["generated_text"]
            response = generated_text[len(prompt):].strip()
            
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


class ImprovedLLMService:
    """
    Enhanced LLM service with Gemini as primary provider and HF as fallback.
    """
    
    def __init__(self):
        self.providers = []
        self.current_provider = None
        self.provider_last_check = {}
        self.provider_check_interval = 300  # 5 minutes
        self._setup_providers()
        
    def _setup_providers(self):
        """Setup available LLM providers in order of preference."""
        
        # Add Gemini provider (best performance and reliability)
        self.providers.append(GeminiProvider())
        
        # Add HuggingFace provider (fallback)
        self.providers.append(HuggingFaceProvider())
        
    async def _find_available_provider(self) -> Optional[BaseLLMProvider]:
        """Find the first available provider."""
        current_time = time.time()
        
        for provider in self.providers:
            provider_name = provider.__class__.__name__
            
            # Skip recently failed providers
            last_check = self.provider_last_check.get(provider_name, 0)
            if current_time - last_check < 60:  # 1 minute cooldown
                continue
            
            try:
                is_available = await asyncio.wait_for(
                    provider.is_available(),
                    timeout=30.0
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
        """Generate response using the first available provider."""
        if not prompt or not prompt.strip():
            return "I need a question or prompt to respond to."
        
        # Truncate very long prompts
        if len(prompt) > 30000:
            prompt = prompt[:30000] + "..."
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
1. GEMINI_API_KEY is set in your environment variables, OR
2. System has sufficient resources for HuggingFace models

To get a Gemini API key: visit https://makersuite.google.com/app/apikey"""
            
            logger.error("No LLM providers available")
            raise Exception(error_msg)
        
        # Try the new provider
        try:
            return await self.current_provider.generate_response(prompt, max_tokens)
        except Exception as e:
            logger.error(f"All providers failed. Last error: {e}")
            self.current_provider = None
            return f"I'm experiencing technical difficulties. Please try again in a few moments. Error: {str(e)[:100]}"

class ImprovedEmbeddingService:
    """
    Enhanced embedding service with better error handling and performance.
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.model = None
        self._initialized = False
        self._initialization_error = None
        self._init_lock = asyncio.Lock()  # Add this to prevent concurrent initialization
        
    async def _initialize(self):
        """Initialize the embedding model - FIXED to prevent recursion."""
        # Use lock to prevent concurrent initialization attempts
        async with self._init_lock:
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
                    timeout=300.0
                )
                
                # FIXED: Test the model with direct sync call to prevent recursion
                try:
                    test_embedding = self.model.encode("test", convert_to_numpy=True)
                    if test_embedding is None or len(test_embedding) == 0:
                        raise Exception("Model returned empty embedding")
                    
                    self._initialized = True
                    logger.info(f"Embedding model loaded successfully. Dimension: {len(test_embedding)}")
                    
                except Exception as test_error:
                    raise Exception(f"Model test failed: {test_error}")
                
            except Exception as e:
                error_msg = f"Failed to initialize embedding model: {e}"
                logger.error(error_msg)
                self._initialization_error = Exception(error_msg)
                raise self._initialization_error
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embeddings for a single text."""
        await self._initialize()
        
        try:
            if not text or not text.strip():
                # Return zero vector for empty text
                return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
            
            # Truncate very long texts
            if len(text) > 8000:
                text = text[:8000]
            
            loop = asyncio.get_event_loop()
            embedding = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.model.encode(text, convert_to_numpy=True)
                ),
                timeout=30.0
            )
            
            # Ensure we return a list
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return list(embedding)
            
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
            # Process texts and limit batch size
            processed_texts = []
            for text in texts[:50]:  # REDUCED batch size to prevent memory issues
                if not text or not text.strip():
                    processed_texts.append("empty")  # Placeholder for empty text
                elif len(text) > 8000:
                    processed_texts.append(text[:8000])
                else:
                    processed_texts.append(text)
            
            if not processed_texts:
                return []
            
            loop = asyncio.get_event_loop()
            embeddings = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.model.encode(processed_texts, convert_to_numpy=True)
                ),
                timeout=min(120.0, len(processed_texts) * 3.0)  # More generous timeout
            )
            
            # Convert to list format
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return [list(emb) for emb in embeddings]
            
        except asyncio.TimeoutError:
            logger.error("Batch embedding generation timed out")
            raise Exception("Batch embedding generation timed out")
        except Exception as e:
            logger.error(f"Batch embedding generation error: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if embedding model is available."""
        try:
            await asyncio.wait_for(self._initialize(), timeout=30.0)  # Increased timeout
            return self._initialized
        except Exception as e:
            logger.error(f"Embedding availability check failed: {e}")
            return False
# Global instances
improved_llm_service = ImprovedLLMService()
improved_embedding_service = ImprovedEmbeddingService()