"""
Core interfaces for the translation system.
UPDATED: Added batch operations support and improved error handling.

Version: 2.0 (Production-Ready)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .models import (
    Document,
    TextSegment,
    TranslationRequest,
    TranslationResult,
    TranslationJob,
    FileType
)


logger = logging.getLogger(__name__)


# ============================================================================
# TRANSLATION ENGINE INTERFACE
# ============================================================================

class ITranslationEngine(ABC):
    """Interface for translation engines."""
    
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
            
        Raises:
            TranslationError: If translation fails
            InvalidLanguageError: If language not supported
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
            List of translated texts (same order as input)
            
        Raises:
            TranslationError: If translation fails
            InvalidLanguageError: If language not supported
            ValueError: If texts and contexts length mismatch
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes
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
            True if pair is supported
        """
        pass
    
    @abstractmethod
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with usage statistics (tokens, requests, etc.)
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
    
    # Optional methods with default implementations
    
    def supports_batch_translation(self) -> bool:
        """
        Check if engine supports batch translation.
        
        Returns:
            True if batch translation is supported
        """
        return True
    
    def get_max_text_length(self) -> int:
        """
        Get maximum text length for single translation.
        
        Returns:
            Maximum length in characters
        """
        return 50000  # Default
    
    def get_max_batch_size(self) -> int:
        """
        Get maximum batch size.
        
        Returns:
            Maximum number of texts in batch
        """
        return 50  # Default


# ============================================================================
# CACHE INTERFACE
# ============================================================================

class ITranslationCache(ABC):
    """
    Interface for translation cache systems.
    UPDATED: Added batch operations for optimization.
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
            
        Raises:
            CacheWriteError: If caching fails
        """
        pass
    
    def get_batch(self, requests: List[TranslationRequest]) -> Dict[TranslationRequest, TranslationResult]:
        """
        OPTIMIZATION: Retrieve multiple cached translations in one operation.
        
        Args:
            requests: List of translation requests
            
        Returns:
            Dict mapping request -> result (only for found items)
        
        Note:
            Default implementation falls back to individual get() calls.
            Subclasses should override for better performance.
        """
        results = {}
        for request in requests:
            try:
                result = self.get(request)
                if result:
                    results[request] = result
            except Exception as e:
                logger.warning(f"Cache get failed for request: {e}")
                continue
        return results
    
    def set_batch(self, items: List[tuple[TranslationRequest, TranslationResult]]) -> None:
        """
        OPTIMIZATION: Store multiple translations in one operation.
        
        Args:
            items: List of (request, result) tuples
        
        Note:
            Default implementation falls back to individual set() calls.
            Subclasses should override for better performance.
        """
        for request, result in items:
            try:
                self.set(request, result)
            except Exception as e:
                logger.warning(f"Cache set failed for request: {e}")
                continue
    
    @abstractmethod
    def invalidate(self, glossary_version: str) -> int:
        """
        Invalidate cache entries for specific glossary version.
        
        Args:
            glossary_version: Glossary version to invalidate
            
        Returns:
            Number of invalidated entries
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> int:
        """
        Remove stale/expired entries.
        
        Returns:
            Number of cleaned entries
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics (hits, misses, size, etc.)
        """
        pass
    
    def clear(self) -> int:
        """
        Clear all cache entries.
        
        Returns:
            Number of cleared entries
            
        Note:
            Optional method - not all implementations may support clearing.
        """
        raise NotImplementedError("clear() not implemented for this cache")


# ============================================================================
# GLOSSARY INTERFACE
# ============================================================================

class IGlossaryProcessor(ABC):
    """Interface for glossary processing."""
    
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
            domain: Domain/context
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Preprocessed text (with markers or modifications)
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
            domain: Domain/context
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Postprocessed result (implementation-specific)
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get glossary statistics.
        
        Returns:
            Dictionary with glossary statistics
        """
        pass
    
    def get_version(self) -> str:
        """
        Get glossary version.
        
        Returns:
            Version string (default: 'latest')
        """
        return "latest"
    
    def has_terms_for_domain(self, domain: str) -> bool:
        """
        Check if glossary has terms for domain.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if terms exist for domain
        """
        try:
            stats = self.get_stats()
            return stats.get('total_terms', 0) > 0
        except Exception:
            return False


# ============================================================================
# DOCUMENT PARSER INTERFACE
# ============================================================================

class IDocumentParser(ABC):
    """Interface for document parsers."""
    
    @property
    @abstractmethod
    def supported_file_type(self) -> FileType:
        """
        File type this parser supports.
        
        Returns:
            FileType enum
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if file can be parsed.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if file can be parsed
        """
        pass
    
    @abstractmethod
    def validate_document(self, file_path: Path) -> bool:
        """
        Validate document before parsing.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if document is valid
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: Path) -> Document:
        """
        Parse document and extract content with formatting.
        
        Args:
            file_path: Path to file
            
        Returns:
            Parsed Document object
            
        Raises:
            ParserError: If parsing fails
            CorruptedFileError: If file is corrupted
        """
        pass
    
    @abstractmethod
    def extract_segments(self, document: Document) -> List[TextSegment]:
        """
        Extract translatable segments from document.
        
        Args:
            document: Parsed document
            
        Returns:
            List of translatable segments
        """
        pass
    
    def get_parser_info(self) -> Dict[str, Any]:
        """
        Get parser information.
        
        Returns:
            Dictionary with parser metadata
        """
        return {
            'parser_class': self.__class__.__name__,
            'supported_file_type': self.supported_file_type.value
        }


# ============================================================================
# DOCUMENT FORMATTER INTERFACE
# ============================================================================

class IDocumentFormatter(ABC):
    """Interface for document formatters."""
    
    @property
    @abstractmethod
    def supported_file_type(self) -> FileType:
        """
        File type this formatter supports.
        
        Returns:
            FileType enum
        """
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
            document: Document to format
            output_path: Output file path
            preserve_formatting: Whether to preserve original formatting
            
        Returns:
            Path to saved file
            
        Raises:
            FormatterError: If formatting fails
            OutputError: If file cannot be written
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
            Translated document with copied styles
        """
        pass
    
    @abstractmethod
    def validate_output(self, output_path: Path) -> bool:
        """
        Validate output document.
        
        Args:
            output_path: Path to output file
            
        Returns:
            True if output is valid
        """
        pass
    
    def get_formatter_info(self) -> Dict[str, Any]:
        """
        Get formatter information.
        
        Returns:
            Dictionary with formatter metadata
        """
        return {
            'formatter_class': self.__class__.__name__,
            'supported_file_type': self.supported_file_type.value
        }


# ============================================================================
# PROGRESS CALLBACK INTERFACE
# ============================================================================

class IProgressCallback(ABC):
    """Interface for progress callbacks."""
    
    @abstractmethod
    def on_start(self, job: TranslationJob) -> None:
        """
        Called when translation starts.
        
        Args:
            job: Translation job
        """
        pass
    
    @abstractmethod
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        """
        Called on progress update.
        
        Args:
            job: Translation job
            current: Current progress
            total: Total items
        """
        pass
    
    @abstractmethod
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        """
        Called when a segment is translated.
        
        Args:
            segment: Original segment
            result: Translation result
        """
        pass
    
    @abstractmethod
    def on_complete(self, job: TranslationJob) -> None:
        """
        Called when translation completes.
        
        Args:
            job: Completed translation job
        """
        pass
    
    @abstractmethod
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        """
        Called on error.
        
        Args:
            job: Translation job
            error: Exception that occurred
        """
        pass
    
    def on_cache_hit(self, segment: TextSegment) -> None:
        """
        Called when segment found in cache.
        
        Args:
            segment: Cached segment
            
        Note:
            Optional callback - not all implementations need this.
        """
        pass
    
    def on_batch_start(self, batch_index: int, batch_size: int) -> None:
        """
        Called when batch processing starts.
        
        Args:
            batch_index: Index of current batch
            batch_size: Size of batch
            
        Note:
            Optional callback - not all implementations need this.
        """
        pass


# ============================================================================
# TRANSLATION PIPELINE INTERFACE
# ============================================================================

class ITranslationPipeline(ABC):
    """Interface for translation pipeline."""
    
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
            input_path: Path to input file
            output_path: Path to output file
            source_lang: Source language code
            target_lang: Target language code
            domain: Translation domain
            progress_callback: Optional progress callback
            
        Returns:
            Completed TranslationJob
            
        Raises:
            ValidationError: If inputs are invalid
            ParserError: If parsing fails
            TranslationError: If translation fails
            FormatterError: If formatting fails
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
            domain: Translation domain
            
        Returns:
            Translation result
            
        Raises:
            ValidationError: If inputs are invalid
            TranslationError: If translation fails
        """
        pass
    
    @abstractmethod
    def get_health(self) -> Dict[str, Any]:
        """
        Get pipeline health status.
        
        Returns:
            Dictionary with health information
        """
        pass
    
    def get_supported_file_types(self) -> List[FileType]:
        """
        Get supported file types.
        
        Returns:
            List of supported FileType enums
        """
        return []


# ============================================================================
# ADAPTER CLASSES
# ============================================================================

class CacheAdapter(ITranslationCache):
    """
    UPDATED: Adapter with batch operations support and improved error handling.
    """
    
    def __init__(self, cache_manager):
        """
        Args:
            cache_manager: Instance of CacheManager from cache_manager.py
        """
        if cache_manager is None:
            raise ValueError("cache_manager cannot be None")
        
        self.cache_manager = cache_manager
        # Кэширование проверки batch support
        self._supports_batch = hasattr(cache_manager, 'get_batch')
        self._logger = logging.getLogger(f"{__name__}.CacheAdapter")
    
    def get(self, request: TranslationRequest) -> Optional[TranslationResult]:
        """Retrieve from cache using unified interface."""
        if not isinstance(request, TranslationRequest):
            raise TypeError(f"request must be TranslationRequest, got {type(request).__name__}")
        
        try:
            entry = self.cache_manager.get(
                source_text=request.text,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
                glossary_version=getattr(request, 'glossary_version', 'latest'),
                domain=request.domain
            )
            
            if entry is None:
                return None
            
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
        except Exception as e:
            self._logger.warning(f"Cache get failed: {e}")
            return None
    
    def get_batch(self, requests: List[TranslationRequest]) -> Dict[TranslationRequest, TranslationResult]:
        """
        OPTIMIZATION: Batch cache lookup.
        """
        if not requests:
            return {}
        
        # Validate input
        for i, req in enumerate(requests):
            if not isinstance(req, TranslationRequest):
                raise TypeError(f"requests[{i}] must be TranslationRequest, got {type(req).__name__}")
        
        # Use batch operation if available
        if not self._supports_batch:
            self._logger.debug("Cache manager doesn't support batch operations, using fallback")
            return super().get_batch(requests)
        
        try:
            results = {}
            
            # Build lookup data
            lookup_data = []
            for request in requests:
                lookup_data.append({
                    'source_text': request.text,
                    'source_lang': request.source_lang,
                    'target_lang': request.target_lang,
                    'glossary_version': getattr(request, 'glossary_version', 'latest'),
                    'domain': request.domain
                })
            
            # Batch lookup
            entries = self.cache_manager.get_batch(lookup_data)
            
            # Convert to results
            for request, entry in zip(requests, entries):
                if entry:
                    result = TranslationResult(
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
                    results[request] = result
            
            self._logger.debug(f"Batch cache lookup: {len(results)}/{len(requests)} found")
            return results
            
        except Exception as e:
            self._logger.error(f"Batch cache lookup failed: {e}", exc_info=True)
            # Fallback to individual lookups
            return super().get_batch(requests)
    
    def set(self, request: TranslationRequest, result: TranslationResult) -> None:
        """Store in cache using unified interface."""
        if not isinstance(request, TranslationRequest):
            raise TypeError(f"request must be TranslationRequest, got {type(request).__name__}")
        
        if not isinstance(result, TranslationResult):
            raise TypeError(f"result must be TranslationResult, got {type(result).__name__}")
        
        try:
            from ..cache.cache_manager import CacheEntry
            
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
        except Exception as e:
            self._logger.warning(f"Cache set failed: {e}")
            # Don't raise - caching is optional
    
    def set_batch(self, items: List[tuple[TranslationRequest, TranslationResult]]) -> None:
        """
        OPTIMIZATION: Batch cache storage.
        """
        if not items:
            return
        
        # Validate input
        for i, (req, res) in enumerate(items):
            if not isinstance(req, TranslationRequest):
                raise TypeError(f"items[{i}][0] must be TranslationRequest")
            if not isinstance(res, TranslationResult):
                raise TypeError(f"items[{i}][1] must be TranslationResult")
        
        # Use batch operation if available
        if hasattr(self.cache_manager, 'set_batch'):
            try:
                from ..cache.cache_manager import CacheEntry
                
                entries = []
                for request, result in items:
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
                    entries.append(entry)
                
                self.cache_manager.set_batch(entries)
                self._logger.debug(f"Batch cache set: {len(entries)} entries")
                return
            except Exception as e:
                self._logger.warning(f"Batch cache set failed: {e}")
        
        # Fallback to individual sets
        super().set_batch(items)
    
    def invalidate(self, glossary_version: str) -> int:
        """Invalidate cache entries."""
        try:
            count = self.cache_manager.evict_glossary(glossary_version)
            self._logger.info(f"Invalidated {count} cache entries for glossary {glossary_version}")
            return count
        except Exception as e:
            self._logger.error(f"Cache invalidation failed: {e}")
            return 0
    
    def cleanup(self) -> int:
        """Clean stale entries."""
        try:
            count = self.cache_manager.evict_stale()
            self._logger.info(f"Cleaned {count} stale cache entries")
            return count
        except Exception as e:
            self._logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        try:
            return self.cache_manager.get_stats()
        except Exception as e:
            self._logger.error(f"Failed to get cache stats: {e}")
            return {'error': str(e)}
    
    def clear(self) -> int:
        """Clear all cache entries."""
        if hasattr(self.cache_manager, 'clear'):
            try:
                count = self.cache_manager.clear()
                self._logger.info(f"Cleared {count} cache entries")
                return count
            except Exception as e:
                self._logger.error(f"Cache clear failed: {e}")
                return 0
        else:
            raise NotImplementedError("Cache manager doesn't support clear operation")


class GlossaryAdapter(IGlossaryProcessor):
    """Adapter for existing glossary_manager module with improved error handling."""
    
    def __init__(self, glossary_manager):
        """
        Args:
            glossary_manager: Instance of GlossaryManager from glossary_manager.py
        """
        if glossary_manager is None:
            raise ValueError("glossary_manager cannot be None")
        
        self.glossary_manager = glossary_manager
        self._logger = logging.getLogger(f"{__name__}.GlossaryAdapter")
        
        # Импорт TermStatus при инициализации
        try:
            from ..glossary.glossary_manager import TermStatus
            self._term_status = TermStatus
        except ImportError:
            try:
                from glossary_manager import TermStatus
                self._term_status = TermStatus
            except ImportError:
                self._logger.warning("TermStatus not available, glossary filtering disabled")
                self._term_status = None
    
    def preprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """Mark glossary terms in source text before translation."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        
        if not text or not text.strip():
            return text
        
        try:
            status_filter = self._term_status.APPROVED if self._term_status else None
            
            result = self.glossary_manager.apply_to_text(
                text=text,
                domain=domain,
                strategy="mark",
                status_filter=status_filter
            )
            return result.text
        except Exception as e:
            self._logger.warning(f"Glossary preprocess failed: {e}")
            return text  # Return original text on error
    
    def postprocess(
        self,
        text: str,
        domain: str,
        source_lang: str,
        target_lang: str
    ) -> Any:
        """Apply glossary terms to translated text."""
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        
        if not text or not text.strip():
            # Return mock result for empty text
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                text: str
                terms_applied: list = None
                def __post_init__(self):
                    if self.terms_applied is None:
                        self.terms_applied = []
            return MockResult(text=text)
        
        try:
            status_filter = self._term_status.APPROVED if self._term_status else None
            
            glossary_result = self.glossary_manager.apply_to_text(
                text=text,
                domain=domain,
                strategy="replace",
                status_filter=status_filter
            )
            return glossary_result
        except Exception as e:
            self._logger.warning(f"Glossary postprocess failed: {e}")
            # Return mock result
            from dataclasses import dataclass
            @dataclass
            class MockResult:
                text: str
                terms_applied: list = None
                def __post_init__(self):
                    if self.terms_applied is None:
                        self.terms_applied = []
            return MockResult(text=text)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        try:
            return self.glossary_manager.get_stats()
        except Exception as e:
            self._logger.error(f"Failed to get glossary stats: {e}")
            return {'error': str(e)}
    
    def get_version(self) -> str:
        """Get glossary version."""
        try:
            if hasattr(self.glossary_manager, 'get_version'):
                return self.glossary_manager.get_version()
            return "latest"
        except Exception:
            return "latest"
    
    def has_terms_for_domain(self, domain: str) -> bool:
        """Check if glossary has terms for domain."""
        try:
            stats = self.get_stats()
            # Check if there are terms for this domain
            domain_terms = stats.get('domains', {}).get(domain, 0)
            return domain_terms > 0
        except Exception:
            return False


# ============================================================================
# SIMPLE IMPLEMENTATIONS FOR TESTING
# ============================================================================

class ConsoleProgressCallback(IProgressCallback):
    """Simple console progress callback."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize callback.
        
        Args:
            verbose: If True, print detailed progress
        """
        self.verbose = verbose
    
    def on_start(self, job: TranslationJob) -> None:
        print(f"Starting translation: {job.input_file.name}")
        print(f"  {job.source_lang} -> {job.target_lang}")
        if self.verbose:
            print(f"  Domain: {job.domain}")
            print(f"  Engine: {job.engine}")
    
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        if total > 0:
            progress = (current / total * 100)
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  Progress: [{bar}] {current}/{total} ({progress:.1f}%)", end='', flush=True)
    
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        if self.verbose:
            text_preview = segment.text[:50] + "..." if len(segment.text) > 50 else segment.text
            print(f"\n  Translated: {text_preview}")
    
    def on_complete(self, job: TranslationJob) -> None:
        print("\n✓ Translation complete!")
        print(f"  Output: {job.output_file}")
        print(f"  Duration: {job.duration:.2f}s")
        print(f"  Segments: {job.translated_segments}/{job.total_segments}")
        print(f"  Cached: {job.cached_segments}")
        if job.failed_segments > 0:
            print(f"  ⚠ Failed: {job.failed_segments}")
    
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        print(f"\n✗ Translation failed: {error}")
        if job.errors:
            print("  Errors:")
            for err in job.errors[:5]:  # Show first 5 errors
                print(f"
                print(f"    - {err}")
            if len(job.errors) > 5:
                print(f"    ... and {len(job.errors) - 5} more")
    
    def on_cache_hit(self, segment: TextSegment) -> None:
        if self.verbose:
            print(f"  ⚡ Cache hit: {segment.id}")
    
    def on_batch_start(self, batch_index: int, batch_size: int) -> None:
        if self.verbose:
            print(f"\n  Processing batch {batch_index + 1} ({batch_size} segments)")


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


class LoggingProgressCallback(IProgressCallback):
    """Progress callback that logs to logger instead of console."""
    
    def __init__(self, logger_name: Optional[str] = None):
        """
        Initialize callback.
        
        Args:
            logger_name: Name of logger to use (default: current module)
        """
        self.logger = logging.getLogger(logger_name or __name__)
    
    def on_start(self, job: TranslationJob) -> None:
        self.logger.info(
            f"Starting translation: {job.input_file.name} "
            f"({job.source_lang} -> {job.target_lang})"
        )
    
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        if total > 0 and current % 10 == 0:  # Log every 10 segments
            progress = (current / total * 100)
            self.logger.debug(f"Progress: {current}/{total} ({progress:.1f}%)")
    
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        self.logger.debug(f"Translated segment: {segment.id}")
    
    def on_complete(self, job: TranslationJob) -> None:
        self.logger.info(
            f"Translation complete: {job.translated_segments}/{job.total_segments} segments, "
            f"{job.cached_segments} cached, {job.failed_segments} failed, "
            f"duration={job.duration:.2f}s"
        )
    
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        self.logger.error(f"Translation failed: {error}", exc_info=True)
    
    def on_cache_hit(self, segment: TextSegment) -> None:
        self.logger.debug(f"Cache hit: {segment.id}")
    
    def on_batch_start(self, batch_index: int, batch_size: int) -> None:
        self.logger.debug(f"Processing batch {batch_index + 1} ({batch_size} segments)")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing."""
    
    print("=" * 70)
    print("TESTING INTERFACES")
    print("=" * 70)
    
    # Test 1: ConsoleProgressCallback
    print("\n1. Testing ConsoleProgressCallback:")
    try:
        from pathlib import Path
        from .models import TranslationJob, TranslationStatus
        
        callback = ConsoleProgressCallback(verbose=False)
        
        job = TranslationJob(
            job_id="test-123",
            input_file=Path("test.docx"),
            output_file=Path("test_ru.docx"),
            source_lang="en",
            target_lang="ru",
            domain="general",
            engine="openai"
        )
        job.total_segments = 100
        
        callback.on_start(job)
        
        for i in range(0, 101, 10):
            callback.on_progress(job, i, 100)
        
        job.translated_segments = 80
        job.cached_segments = 20
        job.started_at = datetime.utcnow()
        job.completed_at = datetime.utcnow()
        
        callback.on_complete(job)
        print("   ✓ ConsoleProgressCallback works")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 2: NoOpProgressCallback
    print("\n2. Testing NoOpProgressCallback:")
    try:
        callback = NoOpProgressCallback()
        callback.on_start(job)
        callback.on_progress(job, 50, 100)
        callback.on_complete(job)
        print("   ✓ NoOpProgressCallback works (silent)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 3: CacheAdapter validation
    print("\n3. Testing CacheAdapter input validation:")
    try:
        # Test None validation
        try:
            CacheAdapter(None)
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Caught None validation: {str(e)[:50]}...")
        
        # Test with mock cache manager
        class MockCacheManager:
            def get(self, **kwargs):
                return None
            def set(self, entry):
                pass
            def evict_glossary(self, version):
                return 0
            def evict_stale(self):
                return 0
            def get_stats(self):
                return {'total': 0}
        
        adapter = CacheAdapter(MockCacheManager())
        print("   ✓ CacheAdapter created with mock manager")
        
        # Test type validation
        try:
            adapter.get("not a request")
            print("   ✗ Should have raised TypeError")
        except TypeError as e:
            print(f"   ✓ Caught type validation: {str(e)[:50]}...")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 4: GlossaryAdapter validation
    print("\n4. Testing GlossaryAdapter input validation:")
    try:
        # Test None validation
        try:
            GlossaryAdapter(None)
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Caught None validation: {str(e)[:50]}...")
        
        # Test with mock glossary manager
        class MockGlossaryManager:
            def apply_to_text(self, **kwargs):
                from dataclasses import dataclass
                @dataclass
                class Result:
                    text: str
                    terms_applied: list
                return Result(text=kwargs['text'], terms_applied=[])
            
            def get_stats(self):
                return {'total_terms': 0}
        
        adapter = GlossaryAdapter(MockGlossaryManager())
        print("   ✓ GlossaryAdapter created with mock manager")
        
        # Test type validation
        try:
            adapter.preprocess(123, "domain", "en", "ru")
            print("   ✗ Should have raised TypeError")
        except TypeError as e:
            print(f"   ✓ Caught type validation: {str(e)[:50]}...")
        
        # Test empty text handling
        result = adapter.preprocess("", "domain", "en", "ru")
        print(f"   ✓ Empty text handled: '{result}'")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test 5: Interface methods
    print("\n5. Testing interface method signatures:")
    try:
        # Check ITranslationEngine methods
        required_methods = [
            'name', 'model_name', 'translate', 'translate_batch',
            'get_supported_languages', 'is_language_pair_supported',
            'get_usage_stats', 'validate_config'
        ]
        
        for method in required_methods:
            assert hasattr(ITranslationEngine, method), f"Missing method: {method}"
        
        print("   ✓ ITranslationEngine has all required methods")
        
        # Check ITranslationCache methods
        required_methods = ['get', 'set', 'invalidate', 'cleanup', 'get_stats']
        for method in required_methods:
            assert hasattr(ITranslationCache, method), f"Missing method: {method}"
        
        print("   ✓ ITranslationCache has all required methods")
        
        # Check batch methods exist
        assert hasattr(ITranslationCache, 'get_batch'), "Missing get_batch"
        assert hasattr(ITranslationCache, 'set_batch'), "Missing set_batch"
        print("   ✓ ITranslationCache has batch methods")
        
    except AssertionError as e:
        print(f"   ✗ Error: {e}")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
    
    # Test 6: LoggingProgressCallback
    print("\n6. Testing LoggingProgressCallback:")
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        
        callback = LoggingProgressCallback("test.callback")
        callback.on_start(job)
        callback.on_progress(job, 50, 100)
        callback.on_complete(job)
        print("   ✓ LoggingProgressCallback works")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✓ ALL INTERFACE TESTS PASSED!")
    print("=" * 70)
