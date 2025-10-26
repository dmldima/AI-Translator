"""
Unified interfaces for translation system integration.
Provides consistent API for cache and glossary modules.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime


# ===== Common Data Models =====

@dataclass
class TranslationRequest:
    """Unified translation request."""
    source_text: str
    source_lang: str
    target_lang: str
    domain: str = "general"
    glossary_version: str = "latest"
    
    def __post_init__(self):
        if not all([self.source_text, self.source_lang, self.target_lang]):
            raise ValueError("source_text, source_lang, target_lang are required")


@dataclass
class TranslationResult:
    """Unified translation result."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    domain: str
    model: Optional[str] = None
    confidence: float = 1.0
    cached: bool = False
    glossary_applied: bool = False
    glossary_terms_used: List[str] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.glossary_terms_used is None:
            self.glossary_terms_used = []
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


# ===== Cache Interface =====

class ITranslationCache(ABC):
    """
    Standard interface for translation cache systems.
    All cache implementations must conform to this interface.
    """
    
    @abstractmethod
    def get(self, request: TranslationRequest) -> Optional[TranslationResult]:
        """
        Retrieve cached translation.
        
        Args:
            request: Translation request
            
        Returns:
            TranslationResult if found and valid, None otherwise
        """
        pass
    
    @abstractmethod
    def set(self, request: TranslationRequest, result: TranslationResult) -> None:
        """
        Store translation in cache.
        
        Args:
            request: Original request
            result: Translation result to cache
        """
        pass
    
    @abstractmethod
    def invalidate(self, glossary_version: str) -> int:
        """
        Invalidate cache entries for specific glossary version.
        
        Args:
            glossary_version: Version to invalidate
            
        Returns:
            Number of entries invalidated
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> int:
        """
        Remove stale/expired entries.
        
        Returns:
            Number of entries removed
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with metrics (hits, misses, size, etc.)
        """
        pass


# ===== Glossary Interface =====

class IGlossaryProcessor(ABC):
    """
    Standard interface for glossary processing.
    All glossary implementations must conform to this interface.
    """
    
    @abstractmethod
    def preprocess(self, request: TranslationRequest) -> TranslationRequest:
        """
        Apply glossary terms before translation.
        
        Args:
            request: Original request
            
        Returns:
            Modified request with glossary terms marked
        """
        pass
    
    @abstractmethod
    def postprocess(self, request: TranslationRequest, result: TranslationResult) -> TranslationResult:
        """
        Apply glossary terms after translation.
        
        Args:
            request: Original request
            result: Translation result
            
        Returns:
            Modified result with glossary terms applied
        """
        pass
    
    @abstractmethod
    def get_applicable_terms(self, request: TranslationRequest) -> List[Dict[str, str]]:
        """
        Get glossary terms applicable to the request.
        
        Args:
            request: Translation request
            
        Returns:
            List of applicable terms
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get glossary statistics.
        
        Returns:
            Dict with metrics (total_terms, by_domain, etc.)
        """
        pass


# ===== Translation Pipeline Interface =====

class ITranslationPipeline(ABC):
    """
    Standard interface for translation pipeline.
    Orchestrates cache, glossary, and translation engine.
    """
    
    @abstractmethod
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """
        Execute full translation pipeline.
        
        Pipeline:
        1. Check cache
        2. Apply glossary preprocessing (optional)
        3. Call translation engine
        4. Apply glossary postprocessing
        5. Update cache
        
        Args:
            request: Translation request
            
        Returns:
            TranslationResult
        """
        pass
    
    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """
        Get pipeline health status.
        
        Returns:
            Dict with component statuses
        """
        pass


# ===== Adapter Pattern for existing modules =====

class CacheAdapter(ITranslationCache):
    """
    Adapter for existing cache_manager module.
    Provides unified interface for translation pipeline.
    """
    
    def __init__(self, cache_manager):
        """
        Args:
            cache_manager: Instance of CacheManager from cache_manager.py
        """
        self.cache_manager = cache_manager
    
    def get(self, request: TranslationRequest) -> Optional[TranslationResult]:
        """Retrieve from cache using unified interface."""
        from cache_manager import CacheEntry
        
        entry = self.cache_manager.get(
            source_text=request.source_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            glossary_version=request.glossary_version,
            domain=request.domain
        )
        
        if entry is None:
            return None
        
        # Convert CacheEntry to TranslationResult
        return TranslationResult(
            source_text=entry.source,
            translated_text=entry.target,
            source_lang=entry.source_lang,
            target_lang=entry.target_lang,
            domain=entry.domain,
            model=entry.model,
            confidence=entry.confidence,
            cached=True,
            timestamp=entry.timestamp
        )
    
    def set(self, request: TranslationRequest, result: TranslationResult) -> None:
        """Store in cache using unified interface."""
        from cache_manager import CacheEntry
        
        entry = CacheEntry(
            source=request.source_text,
            target=result.translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            model=result.model or "unknown",
            glossary_version=request.glossary_version,
            domain=request.domain,
            confidence=result.confidence,
            timestamp=result.timestamp
        )
        
        self.cache_manager.set(entry)
    
    def invalidate(self, glossary_version: str) -> int:
        """Invalidate cache entries."""
        return self.cache_manager.evict_glossary(glossary_version)
    
    def cleanup(self) -> int:
        """Clean stale entries."""
        return self.cache_manager.evict_stale()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.cache_manager.get_stats()


class GlossaryAdapter(IGlossaryProcessor):
    """
    Adapter for existing glossary_manager module.
    Provides unified interface for translation pipeline.
    """
    
    def __init__(self, glossary_manager):
        """
        Args:
            glossary_manager: Instance of GlossaryManager from glossary_manager.py
        """
        self.glossary_manager = glossary_manager
    
    def preprocess(self, request: TranslationRequest) -> TranslationRequest:
        """
        Mark glossary terms in source text before translation.
        
        Strategy: Replace terms with special markers that translator preserves.
        """
        from glossary_manager import TermStatus
        
        result = self.glossary_manager.apply_to_text(
            text=request.source_text,
            domain=request.domain,
            strategy="mark",  # @@term@@
            status_filter=TermStatus.APPROVED
        )
        
        # Return modified request
        return TranslationRequest(
            source_text=result.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            domain=request.domain,
            glossary_version=request.glossary_version
        )
    
    def postprocess(self, request: TranslationRequest, result: TranslationResult) -> TranslationResult:
        """
        Apply glossary terms to translated text.
        
        Strategy: Direct replacement of terms in target text.
        """
        from glossary_manager import TermStatus
        
        glossary_result = self.glossary_manager.apply_to_text(
            text=result.translated_text,
            domain=request.domain,
            strategy="replace",
            status_filter=TermStatus.APPROVED
        )
        
        # Update result with glossary info
        result.translated_text = glossary_result.text
        result.glossary_applied = glossary_result.replacements > 0
        result.glossary_terms_used = glossary_result.terms_applied
        
        return result
    
    def get_applicable_terms(self, request: TranslationRequest) -> List[Dict[str, str]]:
        """Get terms applicable to this request."""
        from glossary_manager import TermStatus
        
        terms = list(self.glossary_manager.list_terms(
            domain=request.domain,
            status=TermStatus.APPROVED
        ))
        
        # Filter terms that appear in source text
        applicable = []
        source_lower = request.source_text.lower()
        
        for term in terms:
            if term.source.lower() in source_lower:
                applicable.append({
                    'source': term.source,
                    'target': term.target,
                    'priority': term.priority,
                    'definition': term.definition
                })
        
        return applicable
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.glossary_manager.get_stats()


# ===== Translation Pipeline Implementation =====

class TranslationPipeline(ITranslationPipeline):
    """
    Standard translation pipeline with cache and glossary.
    """
    
    def __init__(
        self,
        cache: ITranslationCache,
        glossary: IGlossaryProcessor,
        translation_engine,  # Your actual translation engine (e.g., OpenAI, DeepL)
        use_preprocessing: bool = False,  # Apply glossary before translation
        use_postprocessing: bool = True   # Apply glossary after translation
    ):
        self.cache = cache
        self.glossary = glossary
        self.engine = translation_engine
        self.use_preprocessing = use_preprocessing
        self.use_postprocessing = use_postprocessing
    
    def translate(self, request: TranslationRequest) -> TranslationResult:
        """Execute full translation pipeline."""
        
        # Step 1: Check cache
        cached_result = self.cache.get(request)
        if cached_result:
            return cached_result
        
        # Step 2: Preprocessing (optional)
        if self.use_preprocessing:
            request = self.glossary.preprocess(request)
        
        # Step 3: Call translation engine
        # This is where your actual translation happens
        # Example interface:
        translated_text = self.engine.translate(
            text=request.source_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang
        )
        
        result = TranslationResult(
            source_text=request.source_text,
            translated_text=translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            domain=request.domain,
            model=getattr(self.engine, 'model_name', 'unknown'),
            cached=False
        )
        
        # Step 4: Postprocessing
        if self.use_postprocessing:
            result = self.glossary.postprocess(request, result)
        
        # Step 5: Update cache
        self.cache.set(request, result)
        
        return result
    
    def get_health(self) -> Dict[str, Any]:
        """Get pipeline health status."""
        return {
            'cache': {
                'status': 'healthy',
                'stats': self.cache.get_stats()
            },
            'glossary': {
                'status': 'healthy',
                'stats': self.glossary.get_stats()
            },
            'engine': {
                'status': 'healthy',
                'model': getattr(self.engine, 'model_name', 'unknown')
            }
        }


# ===== Factory для удобной инициализации =====

class TranslationSystemFactory:
    """Factory for creating translation system with all components."""
    
    @staticmethod
    def create_pipeline(
        cache_db_path: str = "cache.db",
        glossary_db_path: str = "glossary.db",
        translation_engine = None,
        cache_ttl_days: int = 180,
        log_level: str = "INFO"
    ) -> TranslationPipeline:
        """
        Create fully configured translation pipeline.
        
        Args:
            cache_db_path: Path to cache database
            glossary_db_path: Path to glossary database
            translation_engine: Your translation engine instance
            cache_ttl_days: Cache TTL in days
            log_level: Logging level
            
        Returns:
            Configured TranslationPipeline
        """
        from pathlib import Path
        from cache_manager import (
            CacheConfig, SQLiteStorage, CacheManager, setup_logger as cache_logger
        )
        from glossary_manager import (
            GlossaryConfig, SQLiteGlossary, GlossaryManager, setup_logger as glossary_logger
        )
        
        # Setup cache
        cache_config = CacheConfig(
            max_age_days=cache_ttl_days,
            log_level=log_level
        )
        cache_log = cache_logger("cache", cache_config)
        cache_storage = SQLiteStorage(Path(cache_db_path), cache_log)
        cache_manager = CacheManager(cache_storage, cache_config, cache_log)
        
        # Setup glossary
        glossary_config = GlossaryConfig(log_level=log_level)
        glossary_log = glossary_logger("glossary", glossary_config)
        glossary_storage = SQLiteGlossary(Path(glossary_db_path), glossary_log)
        glossary_manager = GlossaryManager(glossary_storage, glossary_config, glossary_log)
        
        # Create adapters
        cache_adapter = CacheAdapter(cache_manager)
        glossary_adapter = GlossaryAdapter(glossary_manager)
        
        # Create pipeline
        return TranslationPipeline(
            cache=cache_adapter,
            glossary=glossary_adapter,
            translation_engine=translation_engine
        )


# ===== Example Usage =====

if __name__ == "__main__":
    # Mock translation engine for demonstration
    class MockTranslationEngine:
        model_name = "mock-translator-v1"
        
        def translate(self, text: str, source_lang: str, target_lang: str) -> str:
            return f"[MOCK TRANSLATION: {text}]"
    
    # Create pipeline using factory
    engine = MockTranslationEngine()
    pipeline = TranslationSystemFactory.create_pipeline(
        translation_engine=engine
    )
    
    # Use unified interface
    request = TranslationRequest(
        source_text="This Agreement is binding.",
        source_lang="en",
        target_lang="ru",
        domain="legal"
    )
    
    result = pipeline.translate(request)
    
    print(f"Source: {result.source_text}")
    print(f"Translation: {result.translated_text}")
    print(f"Cached: {result.cached}")
    print(f"Glossary applied: {result.glossary_applied}")
    print(f"Terms used: {result.glossary_terms_used}")
    
    # Health check
    health = pipeline.get_health()
    print(f"\nSystem Health: {health}")