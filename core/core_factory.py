"""
Factory classes for creating translation system components.
Provides centralized component creation and configuration.
"""
import os
import logging
from pathlib import Path
from typing import Dict, Type, Optional, List

from .interfaces import (
    ITranslationEngine,
    IDocumentParser,
    IDocumentFormatter,
    ITranslationCache,
    IGlossaryProcessor,
    ITranslationPipeline
)
from .models import FileType
from .pipeline import TranslationPipeline


logger = logging.getLogger(__name__)


# ===== Parser Factory =====

class ParserFactory:
    """Factory for creating document parsers."""
    
    _parsers: Dict[FileType, Type[IDocumentParser]] = {}
    
    @classmethod
    def register(cls, file_type: FileType, parser_class: Type[IDocumentParser]):
        """
        Register a parser for a file type.
        
        Args:
            file_type: File type to handle
            parser_class: Parser class
        """
        cls._parsers[file_type] = parser_class
        logger.info(f"Registered parser for {file_type.value}: {parser_class.__name__}")
    
    @classmethod
    def get_parser(cls, file_type: FileType) -> IDocumentParser:
        """
        Get parser for file type.
        
        Args:
            file_type: File type
            
        Returns:
            Parser instance
            
        Raises:
            ValueError: If no parser available
        """
        if file_type not in cls._parsers:
            raise ValueError(f"No parser available for {file_type.value}")
        
        parser_class = cls._parsers[file_type]
        return parser_class()
    
    @classmethod
    def get_parser_for_file(cls, file_path: Path) -> IDocumentParser:
        """
        Get parser for file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Parser instance
        """
        file_type = FileType.from_extension(file_path.suffix)
        return cls.get_parser(file_type)
    
    @classmethod
    def get_supported_types(cls) -> List[FileType]:
        """Get list of supported file types."""
        return list(cls._parsers.keys())
    
    @classmethod
    def is_supported(cls, file_type: FileType) -> bool:
        """Check if file type is supported."""
        return file_type in cls._parsers


# Auto-register parsers
try:
    from ..parsers.docx_parser import DocxParser
    ParserFactory.register(FileType.DOCX, DocxParser)
except ImportError:
    logger.warning("DOCX parser not available")

try:
    from ..parsers.xlsx_parser import XlsxParser
    ParserFactory.register(FileType.XLSX, XlsxParser)
except ImportError:
    logger.warning("XLSX parser not available")


# ===== Formatter Factory =====

class FormatterFactory:
    """Factory for creating document formatters."""
    
    _formatters: Dict[FileType, Type[IDocumentFormatter]] = {}
    
    @classmethod
    def register(cls, file_type: FileType, formatter_class: Type[IDocumentFormatter]):
        """
        Register a formatter for a file type.
        
        Args:
            file_type: File type to handle
            formatter_class: Formatter class
        """
        cls._formatters[file_type] = formatter_class
        logger.info(f"Registered formatter for {file_type.value}: {formatter_class.__name__}")
    
    @classmethod
    def get_formatter(cls, file_type: FileType) -> IDocumentFormatter:
        """
        Get formatter for file type.
        
        Args:
            file_type: File type
            
        Returns:
            Formatter instance
            
        Raises:
            ValueError: If no formatter available
        """
        if file_type not in cls._formatters:
            raise ValueError(f"No formatter available for {file_type.value}")
        
        formatter_class = cls._formatters[file_type]
        return formatter_class()
    
    @classmethod
    def get_formatter_for_file(cls, file_path: Path) -> IDocumentFormatter:
        """
        Get formatter for file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Formatter instance
        """
        file_type = FileType.from_extension(file_path.suffix)
        return cls.get_formatter(file_type)
    
    @classmethod
    def get_supported_types(cls) -> List[FileType]:
        """Get list of supported file types."""
        return list(cls._formatters.keys())
    
    @classmethod
    def is_supported(cls, file_type: FileType) -> bool:
        """Check if file type is supported."""
        return file_type in cls._formatters


# Auto-register formatters
try:
    from ..formatters.docx_formatter import DocxFormatter
    FormatterFactory.register(FileType.DOCX, DocxFormatter)
except ImportError:
    logger.warning("DOCX formatter not available")

try:
    from ..formatters.xlsx_formatter import XlsxFormatter
    FormatterFactory.register(FileType.XLSX, XlsxFormatter)
except ImportError:
    logger.warning("XLSX formatter not available")


# ===== Engine Factory =====

class EngineFactory:
    """Factory for creating translation engines."""
    
    _engines: Dict[str, Type[ITranslationEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[ITranslationEngine]):
        """
        Register an engine.
        
        Args:
            name: Engine name
            engine_class: Engine class
        """
        cls._engines[name] = engine_class
        logger.info(f"Registered engine: {name}")
    
    @classmethod
    def get_engine(cls, name: str, **kwargs) -> ITranslationEngine:
        """
        Create engine instance.
        
        Args:
            name: Engine name
            **kwargs: Engine-specific parameters
            
        Returns:
            Engine instance
            
        Raises:
            ValueError: If engine not found
        """
        if name not in cls._engines:
            available = ', '.join(cls.list_engines())
            raise ValueError(
                f"Unknown engine: {name}. "
                f"Available engines: {available}"
            )
        
        engine_class = cls._engines[name]
        return engine_class(**kwargs)
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """Get list of registered engine names."""
        return list(cls._engines.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if engine is available."""
        return name in cls._engines


# Auto-register engines
try:
    from ..engines.openai_engine import OpenAIEngine
    EngineFactory.register('openai', OpenAIEngine)
except ImportError:
    logger.warning("OpenAI engine not available")

try:
    from ..engines.deepl_engine import DeepLEngine
    EngineFactory.register('deepl', DeepLEngine)
except ImportError:
    logger.warning("DeepL engine not available")


# ===== Translation System Factory =====

class TranslationSystemFactory:
    """
    Factory for creating complete translation systems.
    Handles all component creation and wiring.
    """
    
    @staticmethod
    def create_pipeline(
        engine_name: str,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        glossary_enabled: bool = True,
        cache_db_path: str = "cache.db",
        glossary_db_path: str = "glossary.db",
        cache_ttl_days: int = 180,
        log_level: str = "INFO",
        **engine_kwargs
    ) -> ITranslationPipeline:
        """
        Create fully configured translation pipeline.
        
        Args:
            engine_name: Name of translation engine ('openai', 'deepl')
            api_key: API key (or None to use env var)
            cache_enabled: Enable translation cache
            glossary_enabled: Enable glossary processing
            cache_db_path: Path to cache database
            glossary_db_path: Path to glossary database
            cache_ttl_days: Cache TTL in days
            log_level: Logging level
            **engine_kwargs: Additional engine parameters (model, pro, etc.)
            
        Returns:
            Configured TranslationPipeline
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Get API key from environment if not provided
        if api_key is None:
            if engine_name == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
            elif engine_name == 'deepl':
                api_key = os.getenv('DEEPL_API_KEY')
            
            if not api_key:
                raise ValueError(
                    f"API key required for {engine_name}. "
                    f"Set {engine_name.upper()}_API_KEY environment variable "
                    f"or provide api_key parameter"
                )
        
        # Create translation engine
        logger.info(f"Creating {engine_name} engine...")
        engine = EngineFactory.get_engine(
            engine_name,
            api_key=api_key,
            **engine_kwargs
        )
        
        # Validate engine
        if not engine.validate_config():
            raise ValueError(f"Invalid {engine_name} configuration")
        
        # Create cache if enabled
        cache = None
        if cache_enabled:
            logger.info("Setting up cache...")
            cache = TranslationSystemFactory._create_cache(
                cache_db_path,
                cache_ttl_days,
                log_level
            )
        
        # Create glossary if enabled
        glossary = None
        if glossary_enabled:
            logger.info("Setting up glossary...")
            glossary = TranslationSystemFactory._create_glossary(
                glossary_db_path,
                log_level
            )
        
        # Create pipeline
        pipeline = TranslationPipeline(
            engine=engine,
            cache=cache,
            glossary=glossary
        )
        
        logger.info("Translation system ready")
        return pipeline
    
    @staticmethod
    def _create_cache(
        db_path: str,
        ttl_days: int,
        log_level: str
    ) -> Optional[ITranslationCache]:
        """Create cache adapter."""
        try:
            from ..cache.cache_manager import (
                CacheConfig,
                SQLiteStorage,
                CacheManager,
                setup_logger
            )
            from .interfaces import ITranslationCache
            
            # Check if we have the adapter in interfaces
            # If not, create inline
            class CacheAdapter(ITranslationCache):
                """Adapter for cache manager."""
                
                def __init__(self, cache_manager):
                    self.cache_manager = cache_manager
                
                def get(self, request):
                    entry = self.cache_manager.get(
                        source_text=request.text,
                        source_lang=request.source_lang,
                        target_lang=request.target_lang,
                        glossary_version=getattr(request, 'glossary_version', 'latest'),
                        domain=request.domain
                    )
                    
                    if entry is None:
                        return None
                    
                    from ..core.models import TranslationResult
                    return TranslationResult(
                        original_text=entry.source,
                        translated_text=entry.target,
                        source_lang=entry.source_lang,
                        target_lang=entry.target_lang,
                        domain=entry.domain,
                        engine="cached",
                        model=entry.model,
                        cached=True,
                        timestamp=entry.timestamp
                    )
                
                def set(self, request, result):
                    from ..cache.cache_manager import CacheEntry
                    
                    entry = CacheEntry(
                        source=request.text,
                        target=result.translated_text,
                        source_lang=request.source_lang,
                        target_lang=request.target_lang,
                        model=result.model,
                        glossary_version=getattr(request, 'glossary_version', 'latest'),
                        domain=request.domain,
                        confidence=result.confidence
                    )
                    
                    self.cache_manager.set(entry)
                
                def invalidate(self, glossary_version: str) -> int:
                    return self.cache_manager.evict_glossary(glossary_version)
                
                def cleanup(self) -> int:
                    return self.cache_manager.evict_stale()
                
                def get_stats(self):
                    return self.cache_manager.get_stats()
            
            config = CacheConfig(
                max_age_days=ttl_days,
                log_level=log_level
            )
            cache_logger = setup_logger("cache", config)
            storage = SQLiteStorage(Path(db_path), cache_logger)
            cache_manager = CacheManager(storage, config, cache_logger)
            
            return CacheAdapter(cache_manager)
            
        except ImportError as e:
            logger.warning(f"Cache not available: {e}")
            return None
    
    @staticmethod
    def _create_glossary(
        db_path: str,
        log_level: str
    ) -> Optional[IGlossaryProcessor]:
        """Create glossary adapter."""
        try:
            from ..glossary.glossary_manager import (
                GlossaryConfig,
                SQLiteGlossary,
                GlossaryManager,
                setup_logger,
                TermStatus
            )
            from .interfaces import IGlossaryProcessor
            
            class GlossaryAdapter(IGlossaryProcessor):
                """Adapter for glossary manager."""
                
                def __init__(self, glossary_manager):
                    self.glossary_manager = glossary_manager
                
                def preprocess(self, text, domain, source_lang, target_lang):
                    result = self.glossary_manager.apply_to_text(
                        text=text,
                        domain=domain,
                        strategy="mark",
                        status_filter=TermStatus.APPROVED
                    )
                    return result.text
                
                def postprocess(self, text, domain, source_lang, target_lang):
                    return self.glossary_manager.apply_to_text(
                        text=text,
                        domain=domain,
                        strategy="replace",
                        status_filter=TermStatus.APPROVED
                    )
                
                def get_stats(self):
                    return self.glossary_manager.get_stats()
            
            config = GlossaryConfig(log_level=log_level)
            glossary_logger = setup_logger("glossary", config)
            storage = SQLiteGlossary(Path(db_path), glossary_logger)
            glossary_manager = GlossaryManager(storage, config, glossary_logger)
            
            return GlossaryAdapter(glossary_manager)
            
        except ImportError as e:
            logger.warning(f"Glossary not available: {e}")
            return None
    
    @staticmethod
    def create_engine(
        engine_name: str,
        api_key: Optional[str] = None,
        **kwargs
    ) -> ITranslationEngine:
        """
        Create translation engine only.
        
        Args:
            engine_name: Engine name
            api_key: API key
            **kwargs: Engine parameters
            
        Returns:
            Translation engine
        """
        if api_key is None:
            if engine_name == 'openai':
                api_key = os.getenv('OPENAI_API_KEY')
            elif engine_name == 'deepl':
                api_key = os.getenv('DEEPL_API_KEY')
        
        if not api_key:
            raise ValueError(f"API key required for {engine_name}")
        
        return EngineFactory.get_engine(
            engine_name,
            api_key=api_key,
            **kwargs
        )


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    
    # List available components
    print("Available parsers:")
    for ft in ParserFactory.get_supported_types():
        print(f"  - {ft.value}")
    
    print("\nAvailable formatters:")
    for ft in FormatterFactory.get_supported_types():
        print(f"  - {ft.value}")
    
    print("\nAvailable engines:")
    for name in EngineFactory.list_engines():
        print(f"  - {name}")
    
    # Create pipeline
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("\nCreating translation system...")
        
        try:
            pipeline = TranslationSystemFactory.create_pipeline(
                engine_name="openai",
                api_key=api_key,
                model="gpt-4o-mini",
                cache_enabled=True,
                glossary_enabled=False
            )
            
            print("✓ Pipeline created successfully")
            
            # Test translation
            result = pipeline.translate_text(
                text="Hello, world!",
                source_lang="en",
                target_lang="es"
            )
            
            print(f"\nTest translation:")
            print(f"  Original: {result.original_text}")
            print(f"  Translated: {result.translated_text}")
            print(f"  Cached: {result.cached}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
    else:
        print("\nSet OPENAI_API_KEY to test pipeline creation")
