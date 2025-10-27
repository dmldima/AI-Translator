"""
DeepL Translation Engine.
Professional translation service with high quality.
"""
import time
from typing import List, Optional, Dict, Any
import logging
import requests

from ..core.interfaces import ITranslationEngine


logger = logging.getLogger(__name__)


class DeepLEngine(ITranslationEngine):
    """
    DeepL-based translation engine.
    Supports both Free and Pro APIs.
    """
    
    # DeepL API endpoints
    FREE_API_URL = "https://api-free.deepl.com/v2"
    PRO_API_URL = "https://api.deepl.com/v2"
    
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
    
    def __init__(
        self,
        api_key: str,
        pro: bool = False,
        formality: str = "default",
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize DeepL engine.
        
        Args:
            api_key: DeepL API key
            pro: Use Pro API (default: Free)
            formality: Tone (default, more, less, prefer_more, prefer_less)
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts
        """
        self.api_key = api_key
        self.base_url = self.PRO_API_URL if pro else self.FREE_API_URL
        self.formality = formality
        self.timeout = timeout
        self.max_retries = max_retries
        self._model = "deepl-pro" if pro else "deepl-free"
        
        # Usage tracking
        self._total_chars = 0
        self._total_requests = 0
        self._total_errors = 0
        
        logger.info(f"Initialized DeepL engine ({'Pro' if pro else 'Free'})")
    
    @property
    def name(self) -> str:
        """Engine name."""
        return "deepl"
    
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
        Translate single text using DeepL API.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional context (DeepL supports context in Pro)
            
        Returns:
            Translated text
            
        Raises:
            TranslationError: If translation fails after retries
        """
        if not text.strip():
            return text
        
        # Convert language codes
        source = self._convert_lang_code(source_lang)
        target = self._convert_lang_code(target_lang)
        
        # Prepare request
        url = f"{self.base_url}/translate"
        headers = {
            "Authorization": f"DeepL-Auth-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "text": [text],
            "source_lang": source,
            "target_lang": target
        }
        
        # Add formality if supported
        if self.formality != "default" and target in ["DE", "FR", "IT", "ES", "NL", "PL", "PT-PT", "RU"]:
            data["formality"] = self.formality
        
        # Add context if available (Pro only)
        if context and "pro" in self._model:
            data["context"] = context
        
        # Translate with retry logic
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translated_text = result["translations"][0]["text"]
                    
                    # Update stats
                    self._total_chars += len(text)
                    self._total_requests += 1
                    
                    logger.debug(f"Translated {len(text)} chars")
                    return translated_text
                
                elif response.status_code == 429:
                    # Rate limit
                    logger.warning(f"Rate limit hit, attempt {attempt + 1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)
                    else:
                        self._total_errors += 1
                        raise TranslationError("Rate limit exceeded")
                
                elif response.status_code == 456:
                    # Quota exceeded
                    self._total_errors += 1
                    raise TranslationError("DeepL quota exceeded")
                
                else:
                    self._total_errors += 1
                    raise TranslationError(f"DeepL API error: {response.status_code} - {response.text}")
                    
            except requests.Timeout:
                logger.warning(f"Timeout, attempt {attempt + 1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    self._total_errors += 1
                    raise TranslationError("Request timeout")
            
            except requests.RequestException as e:
                logger.error(f"Request error: {e}")
                self._total_errors += 1
                raise TranslationError(f"DeepL API error: {e}")
            
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
        
        DeepL supports up to 50 texts per request.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: Optional contexts (not used in batch)
            
        Returns:
            List of translated texts
        """
        if not texts:
            return []
        
        # DeepL batch limit
        BATCH_SIZE = 50
        
        all_translations = []
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            
            # Convert language codes
            source = self._convert_lang_code(source_lang)
            target = self._convert_lang_code(target_lang)
            
            url = f"{self.base_url}/translate"
            headers = {
                "Authorization": f"DeepL-Auth-Key {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "text": batch,
                "source_lang": source,
                "target_lang": target
            }
            
            if self.formality != "default":
                data["formality"] = self.formality
            
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    translations = [t["text"] for t in result["translations"]]
                    all_translations.extend(translations)
                    
                    # Update stats
                    self._total_chars += sum(len(t) for t in batch)
                    self._total_requests += 1
                else:
                    logger.error(f"Batch translation failed: {response.status_code}")
                    # Return originals on error
                    all_translations.extend(batch)
                    
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                all_translations.extend(batch)
        
        return all_translations
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of language codes
        """
        return list(self.LANGUAGE_MAP.keys())
    
    def is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if language pair is supported.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if supported
        """
        try:
            self._convert_lang_code(source_lang)
            self._convert_lang_code(target_lang)
            return True
        except ValueError:
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with usage metrics
        """
        # Get usage info from DeepL
        usage_info = self._get_usage_info()
        
        return {
            'engine': self.name,
            'model': self.model_name,
            'total_requests': self._total_requests,
            'total_chars': self._total_chars,
            'total_errors': self._total_errors,
            'character_count': usage_info.get('character_count', 0),
            'character_limit': usage_info.get('character_limit', 0),
            'estimated_cost_usd': self._estimate_cost()
        }
    
    def validate_config(self) -> bool:
        """
        Validate configuration by checking usage.
        
        Returns:
            True if valid
        """
        try:
            usage = self._get_usage_info()
            return usage.get('character_limit', 0) > 0
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _convert_lang_code(self, code: str) -> str:
        """Convert standard language code to DeepL code."""
        if code.upper() in self.LANGUAGE_MAP.values():
            return code.upper()
        
        code_lower = code.lower()
        if code_lower in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[code_lower]
        
        raise ValueError(f"Unsupported language code: {code}")
    
    def _get_usage_info(self) -> Dict[str, Any]:
        """Get usage information from DeepL API."""
        try:
            url = f"{self.base_url}/usage"
            headers = {"Authorization": f"DeepL-Auth-Key {self.api_key}"}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get usage info: {e}")
            return {}
    
    def _estimate_cost(self) -> float:
        """
        Estimate cost in USD.
        
        DeepL pricing (as of 2024):
        - Free: 500,000 chars/month (free)
        - Pro: $5.49 per 1M chars (€4.99)
        
        Returns:
            Estimated cost in USD
        """
        if "free" in self._model:
            return 0.0
        else:
            # Pro pricing: ~$5.49 per 1M characters
            cost_per_char = 5.49 / 1_000_000
            return self._total_chars * cost_per_char


class TranslationError(Exception):
    """Exception raised when translation fails."""
    pass


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    
    # Initialize engine
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        print("Please set DEEPL_API_KEY environment variable")
        exit(1)
    
    engine = DeepLEngine(api_key=api_key, pro=False)
    
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
    translations = engine.translate_batch(texts, "en", "de")
    print(f"\nBatch translation:")
    for orig, trans in zip(texts, translations):
        print(f"  {orig} → {trans}")
    
    # Usage stats
    stats = engine.get_usage_stats()
    print(f"\nUsage stats:")
    print(f"  Requests: {stats['total_requests']}")
    print(f"  Characters: {stats['total_chars']}")
    print(f"  Usage: {stats['character_count']}/{stats['character_limit']}")
    print(f"  Estimated cost: ${stats['estimated_cost_usd']:.4f}")
