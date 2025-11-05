"""
DeepL Translation Engine.
Professional translation service with high quality.

Improvements:
- Session reuse with connection pooling
- Proper exception hierarchy (imported from core)
- Exponential backoff with jitter
- Enhanced error context
- Configuration validation
- Improved logging
- Resource cleanup support
"""
import time
import random
from typing import List, Optional, Dict, Any
import logging
import requests
from functools import wraps

from ..core.interfaces import ITranslationEngine
from ..core.exceptions import (
    TranslationError,
    RetryableTranslationError,
    RateLimitError,
    QuotaExceededError,
    AuthenticationError,
    InvalidRequestError
)


logger = logging.getLogger(__name__)


# Configuration constants
class DeepLConfig:
    """DeepL API configuration constants."""
    FREE_API_URL = "https://api-free.deepl.com/v2"
    PRO_API_URL = "https://api.deepl.com/v2"
    BATCH_SIZE_LIMIT = 50
    MAX_TEXT_LENGTH = 130000  # DeepL character limit
    DEFAULT_TIMEOUT = 30
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_POOL_CONNECTIONS = 10
    DEFAULT_POOL_MAXSIZE = 20
    
    # Pricing (USD, as of 2024)
    FREE_MONTHLY_CHAR_LIMIT = 500000
    PRO_COST_PER_MILLION_CHARS = 5.49


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 32.0):
    """
    Decorator for retry logic with exponential backoff and jitter.
    
    Only retries on RetryableTranslationError and its subclasses.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RetryableTranslationError as e:
                    last_exception = e
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries ({max_retries}) exceeded: {e}")
                        raise
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    jitter = delay * (0.5 + random.random())  # 50-150% of delay
                    
                    logger.warning(
                        f"Retry attempt {attempt + 1}/{max_retries} after {jitter:.2f}s: {e}",
                        extra={'attempt': attempt + 1, 'delay': jitter}
                    )
                    time.sleep(jitter)
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


class DeepLEngine(ITranslationEngine):
    """
    DeepL-based translation engine with improved reliability and performance.
    
    Features:
    - Session reuse with connection pooling
    - Exponential backoff retry logic
    - Proper exception handling
    - Resource cleanup support
    
    Supports both Free and Pro APIs.
    """
    
    # Language mapping (DeepL uses different codes)
    LANGUAGE_MAP = {
        'en': 'EN',
        'de': 'DE',
        'fr': 'FR',
        'es': 'ES',
        'it': 'IT',
        'nl': 'NL',
        'pl': 'PL',
        'pt': 'PT-PT',
        'ru': 'RU',
        'ja': 'JA',
        'zh': 'ZH',
        'bg': 'BG',
        'cs': 'CS',
        'da': 'DA',
        'el': 'EL',
        'et': 'ET',
        'fi': 'FI',
        'hu': 'HU',
        'id': 'ID',
        'ko': 'KO',
        'lt': 'LT',
        'lv': 'LV',
        'nb': 'NB',
        'ro': 'RO',
        'sk': 'SK',
        'sl': 'SL',
        'sv': 'SV',
        'tr': 'TR',
        'uk': 'UK'
    }
    
    # Languages that support formality parameter
    FORMALITY_SUPPORTED = {"DE", "FR", "IT", "ES", "NL", "PL", "PT-PT", "RU"}
    
    def __init__(
        self,
        api_key: str,
        pro: bool = False,
        formality: str = "default",
        timeout: int = DeepLConfig.DEFAULT_TIMEOUT,
        max_retries: int = DeepLConfig.DEFAULT_MAX_RETRIES,
        session: Optional[requests.Session] = None
    ):
        """
        Initialize DeepL engine with validation and session pooling.
        
        Args:
            api_key: DeepL API key
            pro: Use Pro API (default: Free)
            formality: Tone (default, more, less, prefer_more, prefer_less)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
            session: Optional pre-configured session (for testing)
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate configuration
        if not api_key or not api_key.strip():
            raise ValueError("DeepL API key is required and cannot be empty")
        
        if timeout <= 0:
            raise ValueError(f"Timeout must be positive, got {timeout}")
        
        if max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {max_retries}")
        
        if formality not in ["default", "more", "less", "prefer_more", "prefer_less"]:
            raise ValueError(f"Invalid formality value: {formality}")
        
        self.api_key = api_key
        self.base_url = DeepLConfig.PRO_API_URL if pro else DeepLConfig.FREE_API_URL
        self.formality = formality
        self.timeout = timeout
        self.max_retries = max_retries
        self._model = "deepl-pro" if pro else "deepl-free"
        
        # Session with connection pooling
        self._session = session or self._create_session()
        self._owns_session = session is None  # Track if we created the session
        
        # Usage tracking
        self._total_chars = 0
        self._total_requests = 0
        self._total_errors = 0
        
        logger.info(
            f"Initialized DeepL engine",
            extra={
                'model': self._model,
                'timeout': timeout,
                'max_retries': max_retries
            }
        )
    
    def _create_session(self) -> requests.Session:
        """Create and configure a session with connection pooling."""
        session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=DeepLConfig.DEFAULT_POOL_CONNECTIONS,
            pool_maxsize=DeepLConfig.DEFAULT_POOL_MAXSIZE,
            max_retries=0  # We handle retries manually
        )
        
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        # Set default headers
        session.headers.update({
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "Python-DeepL-Client/1.0"
        })
        
        return session
    
    @property
    def name(self) -> str:
        """Engine name."""
        return "deepl"
    
    @property
    def model_name(self) -> str:
        """Current model name."""
        return self._model
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=32.0)
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None
    ) -> str:
        """
        Translate single text using DeepL API.
        
        Args:
            text: Text to translate (max 130,000 characters)
            source_lang: Source language code (e.g., 'en', 'de')
            target_lang: Target language code
            context: Optional context (Pro API only)
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: Base class for all translation errors
            RateLimitError: Rate limit exceeded (retryable)
            QuotaExceededError: Character quota exceeded (not retryable)
            AuthenticationError: Invalid API key (not retryable)
            InvalidRequestError: Bad request parameters (not retryable)
            RetryableTranslationError: Transient errors (retryable)
        """
        # Validate input
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        
        if not text.strip():
            return text
        
        if len(text) > DeepLConfig.MAX_TEXT_LENGTH:
            raise InvalidRequestError(
                f"Text exceeds DeepL limit of {DeepLConfig.MAX_TEXT_LENGTH} chars ({len(text)})",
                source_lang=source_lang,
                target_lang=target_lang
            )
        
        if source_lang == target_lang:
            logger.warning(f"Source and target languages are identical: {source_lang}")
            return text
        
        # Convert language codes
        try:
            source = self._convert_lang_code(source_lang)
            target = self._convert_lang_code(target_lang)
        except ValueError as e:
            raise InvalidRequestError(str(e), source_lang=source_lang, target_lang=target_lang)
        
        # Prepare request
        url = f"{self.base_url}/translate"
        data = {
            "text": [text],
            "source_lang": source,
            "target_lang": target
        }
        
        # Add formality if supported
        if self.formality != "default" and target in self.FORMALITY_SUPPORTED:
            data["formality"] = self.formality
        
        # Add context if available (Pro only)
        if context and "pro" in self._model:
            data["context"] = context
        
        # Make request (retry decorator handles retries)
        try:
            response = self._session.post(
                url,
                json=data,
                timeout=(5, self.timeout)  # (connect_timeout, read_timeout)
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                translated_text = result["translations"][0]["text"]
                
                # Update stats
                self._total_chars += len(text)
                self._total_requests += 1
                
                logger.debug(
                    f"Translation successful",
                    extra={
                        'chars': len(text),
                        'source': source_lang,
                        'target': target_lang
                    }
                )
                
                return translated_text
            
            # Handle error responses
            self._handle_error_response(response, source_lang, target_lang)
            
        except requests.Timeout as e:
            self._total_errors += 1
            raise RetryableTranslationError(
                f"Request timeout after {self.timeout}s",
                source_lang=source_lang,
                target_lang=target_lang
            ) from e
        
        except requests.ConnectionError as e:
            self._total_errors += 1
            raise RetryableTranslationError(
                f"Connection error: {e}",
                source_lang=source_lang,
                target_lang=target_lang
            ) from e
        
        except requests.RequestException as e:
            self._total_errors += 1
            raise TranslationError(
                f"Request failed: {e}",
                source_lang=source_lang,
                target_lang=target_lang
            ) from e
    
    def _handle_error_response(
        self,
        response: requests.Response,
        source_lang: str,
        target_lang: str
    ) -> None:
        """
        Handle HTTP error responses from DeepL API.
        
        Raises appropriate exception based on status code.
        """
        status = response.status_code
        
        # Try to get error message from response
        try:
            error_data = response.json()
            error_msg = error_data.get('message', response.text)
        except Exception:
            error_msg = response.text
        
        self._total_errors += 1
        
        # Rate limit (retryable)
        if status == 429:
            raise RateLimitError(
                "Rate limit exceeded - too many requests",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
        
        # Quota exceeded (not retryable)
        elif status == 456:
            raise QuotaExceededError(
                "DeepL quota exceeded - upgrade plan or wait for reset",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
        
        # Authentication error (not retryable)
        elif status == 403:
            raise AuthenticationError(
                "Invalid DeepL API key or insufficient permissions",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
        
        # Client errors (not retryable)
        elif 400 <= status < 500:
            raise InvalidRequestError(
                f"Bad request: {error_msg}",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
        
        # Server errors (retryable)
        elif 500 <= status < 600:
            raise RetryableTranslationError(
                f"DeepL server error ({status}): {error_msg}",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
        
        # Unknown error
        else:
            raise TranslationError(
                f"Unexpected response ({status}): {error_msg}",
                source_lang=source_lang,
                target_lang=target_lang,
                status_code=status,
                response_body=error_msg
            )
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        contexts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Translate multiple texts in batch with partial failure handling.
        
        DeepL supports up to 50 texts per request.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: Optional contexts (not used in batch)
            
        Returns:
            List of translated texts (original text if individual translation fails)
            
        Note:
            If all translations fail, raises TranslationError.
            Individual failures are logged but don't stop the batch.
        """
        if not texts:
            return []
        
        all_translations = []
        total_errors = 0
        
        # Process in batches
        for i in range(0, len(texts), DeepLConfig.BATCH_SIZE_LIMIT):
            batch = texts[i:i + DeepLConfig.BATCH_SIZE_LIMIT]
            
            try:
                # Convert language codes
                source = self._convert_lang_code(source_lang)
                target = self._convert_lang_code(target_lang)
                
                url = f"{self.base_url}/translate"
                data = {
                    "text": batch,
                    "source_lang": source,
                    "target_lang": target
                }
                
                if self.formality != "default" and target in self.FORMALITY_SUPPORTED:
                    data["formality"] = self.formality
                
                response = self._session.post(
                    url,
                    json=data,
                    timeout=(5, self.timeout)
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translations = [t["text"] for t in result["translations"]]
                    all_translations.extend(translations)
                    
                    # Update stats
                    self._total_chars += sum(len(t) for t in batch)
                    self._total_requests += 1
                    
                    logger.debug(f"Batch translated: {len(batch)} texts")
                else:
                    # Batch failed, fall back to individual translations
                    logger.warning(
                        f"Batch translation failed ({response.status_code}), "
                        f"falling back to individual translations"
                    )
                    for text in batch:
                        try:
                            translated = self.translate(text, source_lang, target_lang)
                            all_translations.append(translated)
                        except TranslationError as e:
                            logger.error(f"Individual translation failed: {e}")
                            all_translations.append(text)  # Return original
                            total_errors += 1
                            
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                # Fall back to individual translations
                for text in batch:
                    try:
                        translated = self.translate(text, source_lang, target_lang)
                        all_translations.append(translated)
                    except TranslationError as err:
                        logger.error(f"Individual translation failed: {err}")
                        all_translations.append(text)
                        total_errors += 1
        
        # If everything failed, raise error
        if total_errors == len(texts):
            raise TranslationError(
                f"All {len(texts)} translations failed",
                source_lang=source_lang,
                target_lang=target_lang
            )
        
        return all_translations
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
        """
        return list(self.LANGUAGE_MAP.keys())
    
    def is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if language pair is supported.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if both languages are supported
        """
        try:
            self._convert_lang_code(source_lang)
            self._convert_lang_code(target_lang)
            return True
        except ValueError:
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics including DeepL API usage info.
        
        Returns:
            Dict with usage metrics and cost estimation
        """
        usage_info = self._get_usage_info()
        
        return {
            'engine': self.name,
            'model': self.model_name,
            'total_requests': self._total_requests,
            'total_chars': self._total_chars,
            'total_errors': self._total_errors,
            'api_character_count': usage_info.get('character_count', 0),
            'api_character_limit': usage_info.get('character_limit', 0),
            'estimated_cost_usd': self._estimate_cost()
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration by checking DeepL API usage.
        
        Returns:
            True if API key is valid and has quota
        """
        try:
            usage = self._get_usage_info()
            has_limit = usage.get('character_limit', 0) > 0
            
            if not has_limit:
                logger.error("API key validation failed: no character limit found")
                return False
            
            logger.info(
                f"API key valid: {usage.get('character_count', 0):,} / "
                f"{usage.get('character_limit', 0):,} characters used"
            )
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _convert_lang_code(self, code: str) -> str:
        """
        Convert standard ISO 639-1 language code to DeepL format.
        
        Args:
            code: ISO 639-1 language code (e.g., 'en', 'de')
            
        Returns:
            DeepL language code (e.g., 'EN', 'DE')
            
        Raises:
            ValueError: If language code is unsupported
        """
        # Check if already in DeepL format
        if code.upper() in self.LANGUAGE_MAP.values():
            return code.upper()
        
        # Convert from ISO format
        code_lower = code.lower()
        if code_lower in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[code_lower]
        
        raise ValueError(f"Unsupported language code: {code}")
    
    def _get_usage_info(self) -> Dict[str, Any]:
        """
        Get usage information from DeepL API.
        
        Returns:
            Dict with character_count and character_limit
        """
        try:
            url = f"{self.base_url}/usage"
            response = self._session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get usage info: HTTP {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get usage info: {e}")
            return {}
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost in USD based on character usage.
        
        DeepL pricing (as of 2024):
        - Free: 500,000 chars/month (free)
        - Pro: $5.49 per 1M chars
        
        Returns:
            Estimated cost in USD
        """
        if "free" in self._model:
            return 0.0
        else:
            cost_per_char = DeepLConfig.PRO_COST_PER_MILLION_CHARS / 1_000_000
            return self._total_chars * cost_per_char
    
    def close(self) -> None:
        """Close session and cleanup resources."""
        if self._owns_session and hasattr(self, '_session'):
            self._session.close()
            logger.debug(f"Closed session for {self.name} engine")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()
