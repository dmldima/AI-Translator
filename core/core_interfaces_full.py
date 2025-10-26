"""
Core interfaces for the translation system.
Defines abstract interfaces that all components must implement.
Includes adapter classes for integrating existing modules.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import (
    Document,
    TextSegment,
    TranslationRequest,
    TranslationResult,
    TranslationJob,
    FileType
)


# ===== Translation Engine Interface =====

class ITranslationEngine(ABC):
    """
    Interface for translation engines.
    All translation engines (OpenAI, DeepL, etc.) must implement this.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name (e.g., 'openai', 'deepl')."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        pass
    
    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        context: Optional[str] = None
    ) -> str:
        """
        Translate single text.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            context: Optional context for better translation
            
        Returns:
            Translated text
        """
        pass
    
    @abstractmethod
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        contexts: Optional[List[str]] = None
    ) -> List[str]:
        """
        Translate multiple texts in batch.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            contexts: Optional contexts for each text
            
        Returns:
            List of translated texts
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of language codes
        """
        pass
    
    @abstractmethod
    def is_language_pair_supported(self, source_lang: str, target_lang: str) -> bool:
        """
        Check if language pair is supported.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            True if supported
        """
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict with usage metrics (tokens, requests, cost, etc.)
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """
        Validate engine configuration.
        
        Returns:
            True if configuration is valid
        """
        pass


# ===== Cache Interface =====

class ITranslationCache(ABC):
    """
    Interface for translation cache systems.
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
    Interface for glossary processing.
    """
    
    @abstractmethod
    def preprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Apply glossary preprocessing before translation.
        
        Args:
            text: Source text
            domain: Domain context
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Preprocessed text with glossary markers
        """
        pass
    
    @abstractmethod
    def postprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> Any:
        """
        Apply glossary postprocessing after translation.
        
        Args:
            text: Translated text
            domain: Domain context
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Result object with processed text and applied terms
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


# ===== Document Parser Interface =====

class IDocumentParser(ABC):
    """
    Interface for document parsers.
    """
    
    @property
    @abstractmethod
    def supported_file_type(self) -> FileType:
        """File type this parser supports."""
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file can be parsed.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if can parse
        """
        pass
    
    @abstractmethod
    def validate_document(self, file_path: Path) -> bool:
        """
        Validate document before parsing.
        
        Args:
            file_path: Path to document
            
        Returns:
            True if valid
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """
        Parse document and extract content with formatting.
        
        Args:
            file_path: Path to document
            
        Returns:
            Document object with segments
        """
        pass
    
    @abstractmethod
    def extract_segments(self, document: Document) -> List[TextSegment]:
        """
        Extract translatable segments from document.
        
        Args:
            document: Parsed document
            
        Returns:
            List of segments ready for translation
        """
        pass


# ===== Document Formatter Interface =====

class IDocumentFormatter(ABC):
    """
    Interface for document formatters.
    """
    
    @property
    @abstractmethod
    def supported_file_type(self) -> FileType:
        """File type this formatter supports."""
        pass
    
    @abstractmethod
    def format(
        self,
        document: Document,
        output_path: Path,
        preserve_formatting: bool = True
    ) -> Path:
        """
        Format and save document.
        
        Args:
            document: Document with translated segments
            output_path: Where to save
            preserve_formatting: Whether to preserve formatting
            
        Returns:
            Path to saved document
        """
        pass
    
    @abstractmethod
    def preserve_styles(
        self,
        original: Document,
        translated: Document
    ) -> Document:
        """
        Copy styles from original to translated document.
        
        Args:
            original: Original document
            translated: Translated document
            
        Returns:
            Translated document with preserved styles
        """
        pass
    
    @abstractmethod
    def validate_output(self, output_path: Path) -> bool:
        """
        Validate output document.
        
        Args:
            output_path: Path to output document
            
        Returns:
            True if valid
        """
        pass


# ===== Progress Callback Interface =====

class IProgressCallback(ABC):
    """
    Interface for progress callbacks.
    """
    
    @abstractmethod
    def on_start(self, job: TranslationJob) -> None:
        """Called when translation starts."""
        pass
    
    @abstractmethod
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        """Called on progress update."""
        pass
    
    @abstractmethod
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        """Called when a segment is translated."""
        pass
    
    @abstractmethod
    def on_complete(self, job: TranslationJob) -> None:
        """Called when translation completes."""
        pass
    
    @abstractmethod
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        """Called on error."""
        pass


# ===== Translation Pipeline Interface =====

class ITranslationPipeline(ABC):
    """
    Interface for translation pipeline.
    """
    
    @abstractmethod
    def translate_document(
        self,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        domain: str = "general",
        progress_callback: Optional[IProgressCallback] = None
    ) -> TranslationJob:
        """
        Translate entire document.
        
        Args:
            input_path: Path to input document
            output_path: Path for output document
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for glossary
            progress_callback: Optional progress callback
            
        Returns:
            TranslationJob with results
        """
        pass
    
    @abstractmethod
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: str = "general"
    ) -> TranslationResult:
        """
        Translate plain text (no document).
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for glossary
            
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


# ===== Adapter Classes =====

class CacheAdapter(ITranslationCache):
    """
    Adapter for existing cache_manager module.
    Bridges cache_manager.CacheManager to ITranslationCache interface.
    """
    
    def __init__(self, cache_manager):
        """
        Args:
            cache_manager: Instance of CacheManager from cache_manager.py
        """
        self.cache_manager = cache_manager
    
    def get(self, request: TranslationRequest) -> Optional[TranslationResult]:
        """Retrieve from cache using unified interface."""
        entry = self.cache_manager.get(
            source_text=request.text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            glossary_version=getattr(request, 'glossary_version', 'latest'),
            domain=request.domain
        )
        
        if entry is None:
            return None
        
        # Convert CacheEntry to TranslationResult
        return TranslationResult(
            original_text=entry.source,
            translated_text=entry.target,
            source_lang=entry.source_lang,
            target_lang=entry.target_lang,
            domain=entry.domain,
            engine="cached",
            model=entry.model,
            confidence=entry.confidence,
            cached=True,
            timestamp=entry.timestamp
        )
    
    def set(self, request: TranslationRequest, result: TranslationResult) -> None:
        """Store in cache using unified interface."""
        # Import here to avoid circular dependency
        try:
            from ..cache.cache_manager import CacheEntry
        except ImportError:
            # Fallback for different import structure
            from cache_manager import CacheEntry
        
        entry = CacheEntry(
            source=request.text,
            target=result.translated_text,
            source_lang=request.source_lang,
            target_lang=request.target_lang,
            model=result.model or "unknown",
            glossary_version=getattr(request, 'glossary_version', 'latest'),
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
    Bridges glossary_manager.GlossaryManager to IGlossaryProcessor interface.
    """
    
    def __init__(self, glossary_manager):
        """
        Args:
            glossary_manager: Instance of GlossaryManager from glossary_manager.py
        """
        self.glossary_manager = glossary_manager
    
    def preprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Mark glossary terms in source text before translation.
        
        Strategy: Replace terms with special markers that translator preserves.
        """
        try:
            from ..glossary.glossary_manager import TermStatus
        except ImportError:
            from glossary_manager import TermStatus
        
        result = self.glossary_manager.apply_to_text(
            text=text,
            domain=domain,
            strategy="mark",  # @@term@@
            status_filter=TermStatus.APPROVED
        )
        
        return result.text
    
    def postprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> Any:
        """
        Apply glossary terms to translated text.
        
        Strategy: Direct replacement of terms in target text.
        """
        try:
            from ..glossary.glossary_manager import TermStatus
        except ImportError:
            from glossary_manager import TermStatus
        
        glossary_result = self.glossary_manager.apply_to_text(
            text=text,
            domain=domain,
            strategy="replace",
            status_filter=TermStatus.APPROVED
        )
        
        return glossary_result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return self.glossary_manager.get_stats()


# ===== Simple Implementations for Testing =====

class ConsoleProgressCallback(IProgressCallback):
    """Simple console progress callback."""
    
    def on_start(self, job: TranslationJob) -> None:
        print(f"Starting translation: {job.input_file.name}")
        print(f"  {job.source_lang} -> {job.target_lang}")
    
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        progress = (current / total * 100) if total > 0 else 0
        print(f"  Progress: {current}/{total} ({progress:.1f}%)")
    
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        pass  # Silent for individual segments
    
    def on_complete(self, job: TranslationJob) -> None:
        print(f"✓ Translation complete!")
        print(f"  Output: {job.output_file}")
        print(f"  Duration: {job.duration:.2f}s")
        print(f"  Segments: {job.translated_segments}/{job.total_segments}")
        print(f"  Cached: {job.cached_segments}")
    
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        print(f"✗ Translation failed: {error}")


class NoOpProgressCallback(IProgressCallback):
    """No-operation progress callback (silent)."""
    
    def on_start(self, job: TranslationJob) -> None:
        pass
    
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        pass
    
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        pass
    
    def on_complete(self, job: TranslationJob) -> None:
        pass
    
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        pass


# ===== Example Usage =====

if __name__ == "__main__":
    from .models import TranslationJob, TranslationStatus
    from pathlib import Path
    
    # Test progress callback
    callback = ConsoleProgressCallback()
    
    job = TranslationJob(
        job_id="test_123",
        input_file=Path("test.docx"),
        output_file=Path("test_es.docx"),
        source_lang="en",
        target_lang="es",
        domain="general",
        engine="openai"
    )
    
    callback.on_start(job)
    job.total_segments = 10
    job.started_at = datetime.utcnow()
    
    for i in range(1, 11):
        job.update_progress(i)
        callback.on_progress(job, i, 10)
    
    job.status = TranslationStatus.COMPLETED
    job.completed_at = datetime.utcnow()
    callback.on_complete(job)
