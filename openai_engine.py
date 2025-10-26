"""
OpenAI Translation Engine.
Supports GPT-4, GPT-4-Turbo, GPT-3.5-Turbo.
"""
import time
from typing import List, Optional, Dict, Any
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
import logging

from ..core.interfaces import ITranslationEngine
from ..core.models import SUPPORTED_LANGUAGES


logger = logging.getLogger(__name__)


class OpenAIEngine(ITranslationEngine):
    """
    OpenAI-based translation engine.
    Uses chat completions API for translation.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize OpenAI engine.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo)
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
        """
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self._model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        
        # Usage tracking
        self._total_tokens = 0
        self._total_requests = 0
        self._total_errors = 0
        
        logger.info(f"Initialized OpenAI engine with model: {model}")
    
    @property
    def name(self) -> str:
        """Engine name."""
        return "openai"
    
    @property
    def model_name(self) -> str:
        """Current model name."""
        return self._model
    
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None
    ) -> str:
        """
        Translate single text using OpenAI API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional context for better translation
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails after retries
        """
        if not text.strip():
            return text
        
        # Build prompt
        prompt = self._build_prompt(text, source_lang, target_lang, context)
        
        # Translate with retry logic
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional translator. Translate the text accurately while preserving formatting, tone, and meaning. Only return the translation, nothing else."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                # Extract translation
                translated_text = response.choices[0].message.content.strip()
                
                # Update stats
                self._total_tokens += response.usage.total_tokens
                self._total_requests += 1
                
                logger.debug(
                    f"Translated {len(text)} chars, "
                    f"used {response.usage.total_tokens} tokens"
                )
                
                return translated_text
                
            except RateLimitError as e:
                logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self._total_errors += 1
                    raise TranslationError(f"Rate limit exceeded: {e}")
            
            except APITimeoutError as e:
                logger.warning(f"Timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    self._total_errors += 1
                    raise TranslationError(f"Request timeout: {e}")
            
            except APIError as e:
                logger.error(f"API error: {e}")
                self._total_errors += 1
                raise TranslationError(f"OpenAI API error: {e}")
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                self._total_errors += 1
                raise TranslationError(f"Translation failed: {e}")
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        contexts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Note: OpenAI doesn't have native batch API, so this translates
        texts one by one. Consider using async version for better performance.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: Optional contexts for each text
            
        Returns:
            List of translated texts (same order)
        """
        if contexts is None:
            contexts = [None] * len(texts)
        
        translations = []
        for text, context in zip(texts, contexts):
            try:
                translated = self.translate(text, source_lang, target_lang, context)
                translations.append(translated)
            except Exception as e:
                logger.error(f"Failed to translate text in batch: {e}")
                translations.append(text)  # Return original on error
        
        return translations
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of language codes
        """
        return list(SUPPORTED_LANGUAGES.keys())
    
    def is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if language pair is supported.
        
        OpenAI supports all language pairs in SUPPORTED_LANGUAGES.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if supported
        """
        supported = self.get_supported_languages()
        return source_lang in supported and target_lang in supported
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with usage metrics
        """
        return {
            'engine': self.name,
            'model': self.model_name,
            'total_requests': self._total_requests,
            'total_tokens': self._total_tokens,
            'total_errors': self._total_errors,
            'estimated_cost_usd': self._estimate_cost()
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration by making a test request.
        
        Returns:
            True if valid
        """
        try:
            # Simple test translation
            result = self.translate("Hello", "en", "es")
            return len(result) > 0
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _build_prompt(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str]
    ) -> str:
        """Build translation prompt."""
        source_name = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
        target_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        
        prompt = f"Translate the following text from {source_name} to {target_name}:\n\n{text}"
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return prompt
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost in USD based on tokens used.
        
        Pricing (as of 2024):
        - GPT-4: ~$0.03/1K input tokens, ~$0.06/1K output tokens
        - GPT-3.5-Turbo: ~$0.001/1K input tokens, ~$0.002/1K output tokens
        
        Returns:
            Estimated cost in USD
        """
        if 'gpt-4' in self._model:
            # Rough estimate: assume 50/50 input/output
            cost_per_token = (0.03 + 0.06) / 2000  # Average per token
        else:
            cost_per_token = (0.001 + 0.002) / 2000
        
        return self._total_tokens * cost_per_token


class TranslationError(Exception):
    """Exception raised when translation fails."""
    pass


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    
    # Initialize engine
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    engine = OpenAIEngine(api_key=api_key, model="gpt-4o-mini")
    
    # Validate
    if engine.validate_config():
        print("✓ Configuration valid")
    else:
        print("✗ Configuration invalid")
        exit(1)
    
    # Single translation
    text = "This Agreement is binding between the parties."
    translated = engine.translate(text, "en", "ru")
    print(f"\nOriginal: {text}")
    print(f"Translated: {translated}")
    
    # Batch translation
    texts = [
        "Hello, world!",
        "How are you?",
        "Goodbye!"
    ]
    translations = engine.translate_batch(texts, "en", "es")
    print(f"\nBatch translation:")
    for orig, trans in zip(texts, translations):
        print(f"  {orig} → {trans}")
    
    # Usage stats
    stats = engine.get_usage_stats()
    print(f"\nUsage stats:")
    print(f"  Requests: {stats['total_requests']}")
    print(f"  Tokens: {stats['total_tokens']}")
    print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")