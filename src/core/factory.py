"""
Enhanced Factory Classes - Production-Ready Implementation.
Implements validation, error handling, and dependency injection best practices.
"""
import os
import logging
import threading
from pathlib import Path
from typing import Dict, Type, Optional, List, Any

from .interfaces import (
    ITranslationEngine,
    IDocumentParser,
    IDocumentFormatter,
    ITranslationCache,
    IGlossaryProcessor,
    ITranslationPipeline
)
from .models import FileType
from .exceptions import ValidationError, ConfigurationError
from .pipeline import EnhancedTranslationPipeline


logger = logging.getLogger(__name__)


# === Thread-safe Registry Base Class ===

class ThreadSafeRegistry:
    """Thread-safe base class for component registries."""
    
    def __init__(self):
        self._registry = {}
        self._lock = threading.RLock()
    
    def register(self, key: Any, value: Any) -> None:
        """Thread-safe registration."""
        with self._lock:
            self._registry[key] = value
    
    def get(self, key: Any) -> Optional[Any]:
        """Thread-safe retrieval."""
        with self._lock:
            return self._registry.get(key)
    
    def has(self, key: Any) -> bool:
        """Thread-safe existence check."""
        with self._lock:
            return key in self._registry
    
    def list_keys(self) -> List[Any]:
        """Thread-safe key listing."""
        with self._lock:
            return list(self._registry.keys())
    
    def clear(self) -> None:
        """Thread-safe clear."""
        with self._lock:
            self._registry.clear()


# === Parser Factory ===

class ParserFactory(ThreadSafeRegistry):
    """
    Thread-safe factory for creating document parsers.
    
    Features:
    - Thread-safe registration and retrieval
    - Automatic parser discovery
    - Validation before registration
    - Comprehensive error handling
    """
    
    def register(self, file_type: FileType, parser_class: Type[IDocumentParser]) -> None:
        """
        Register a parser for a file type with validation.
        
        Args:
            file_type: File type to handle
            parser_class: Parser class (must implement IDocumentParser)
            
        Raises:
            ValidationError: If parser invalid
        """
        # Validate parser class
        if not issubclass(parser_class, IDocumentParser):
            raise ValidationError(
                f"Parser class must implement IDocumentParser: {parser_class}"
            )
        
        # Validate file type matches
        try:
            parser_instance = parser_class()
            if parser_instance.supported_file_type != file_type:
                logger.warning(
                    f"Parser {parser_class.__name__} supports {parser_instance.supported_file_type} "
                    f"but registered for {file_type}"
                )
        except Exception as e:
            raise ValidationError(f"Failed to instantiate parser: {e}")
        
        super().register(file_type, parser_class)
        logger.info(f"✓ Registered parser: {file_type.value} → {parser_class.__name__}")
    
    def get_parser(self, file_type: FileType) -> IDocumentParser:
        """
        Get parser instance for file type.
        
        Args:
            file_type: File type
            
        Returns:
            Parser instance
            
        Raises:
            ValidationError: If no parser available
        """
        parser_class = self.get(file_type)
        
        if parser_class is None:
            available = ', '.join(ft.value for ft in self.get_supported_types())
            raise ValidationError(
                f"No parser available for {file_type.value}. "
                f"Available: {available}"
            )
        
        try:
            return parser_class()
        except Exception as e:
            raise ValidationError(f"Failed to create parser for {file_type.value}: {e}")
    
    def get_parser_for_file(self, file_path: Path) -> IDocumentParser:
        """
        Get parser for file with validation.
        
        Args:
            file_path: Path to file
            
        Returns:
            Parser instance
            
        Raises:
            ValidationError: If file invalid or unsupported
        """
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValidationError(f"Not a file: {file_path}")
        
        try:
            file_type = FileType.from_extension(file_path.suffix)
        except ValueError as e:
            raise ValidationError(f"Unsupported file type: {file_path.suffix}") from e
        
        return self.get_parser(file_type)
    
    def get_supported_types(self) -> List[FileType]:
        """Get list of supported file types."""
        return self.list_keys()
    
    def is_supported(self, file_type: FileType) -> bool:
        """Check if file type is supported."""
        return self.has(file_type)


# === Formatter Factory ===

class FormatterFactory(ThreadSafeRegistry):
    """
    Thread-safe factory for creating document formatters.
    
    Features:
    - Thread-safe registration and retrieval
    - Automatic formatter discovery
    - Validation before registration
    - Comprehensive error handling
    """
    
    def register(self, file_type: FileType, formatter_class: Type[IDocumentFormatter]) -> None:
        """
        Register a formatter for a file type with validation.
        
        Args:
            file_type: File type to handle
            formatter_class: Formatter class (must implement IDocumentFormatter)
            
        Raises:
            ValidationError: If formatter invalid
        """
        # Validate formatter class
        if not issubclass(formatter_class, IDocumentFormatter):
            raise ValidationError(
                f"Formatter class must implement IDocumentFormatter: {formatter_class}"
            )
        
        # Validate file type matches
        try:
            formatter_instance = formatter_class()
            if formatter_instance.supported_file_type != file_type:
                logger.warning(
                    f"Formatter {formatter_class.__name__} supports "
                    f"{formatter_instance.supported_file_type} but registered for {file_type}"
                )
        except Exception as e:
            raise ValidationError(f"Failed to instantiate formatter: {e}")
        
        super().register(file_type, formatter_class)
        logger.info(f"✓ Registered formatter: {file_type.value} → {formatter_class.__name__}")
    
    def get_formatter(self, file_type: FileType) -> IDocumentFormatter:
        """
        Get formatter instance for file type.
        
        Args:
            file_type: File type
            
        Returns:
            Formatter instance
            
        Raises:
            ValidationError: If no formatter available
        """
        formatter_class = self.get(file_type)
        
        if formatter_class is None:
            available = ', '.join(ft.value for ft in self.get_supported_types())
            raise ValidationError(
                f"No formatter available for {file_type.value}. "
                f"Available: {available}"
            )
        
        try:
            return formatter_class()
        except Exception as e:
            raise ValidationError(f"Failed to create formatter for {file_type.value}: {e}")
    
    def get_formatter_for_file(self, file_path: Path) -> IDocumentFormatter:
        """
        Get formatter for file with validation.
        
        Args:
            file_path: Path to file
            
        Returns:
            Formatter instance
            
        Raises:
            ValidationError: If file type unsupported
        """
        try:
            file_type = FileType.from_extension(file_path.suffix)
        except ValueError as e:
            raise ValidationError(f"Unsupported file type: {file_path.suffix}") from e
        
        return self.get_formatter(file_type)
    
    def get_supported_types(self) -> List[FileType]:
        """Get list of supported file types."""
        return self.list_keys()
    
    def is_supported(self, file_type: FileType) -> bool:
        """Check if file type is supported."""
        return self.has(file_type)


# === Engine Factory ===

class EngineFactory(ThreadSafeRegistry):
    """
    Thread-safe factory for creating translation engines.
    
    Features:
    - Thread-safe registration and retrieval
    - API key validation
    - Configuration validation
    - Health checks
    """
    
    def register(self, name: str, engine_class: Type[ITranslationEngine]) -> None:
        """
        Register an engine with validation.
        
        Args:
            name: Engine name (lowercase)
            engine_class: Engine class (must implement ITranslationEngine)
            
        Raises:
            ValidationError: If engine invalid
        """
        if not name or not name.strip():
            raise ValidationError("Engine name cannot be empty")
        
        name = name.lower().strip()
        
        if not issubclass(engine_class, ITranslationEngine):
            raise ValidationError(
                f"Engine class must implement ITranslationEngine: {engine_class}"
            )
        
        super().register(name, engine_class)
        logger.info(f"✓ Registered engine: {name} → {engine_class.__name__}")
    
    def get_engine(self, name: str, api_key: Optional[str] = None, **kwargs) -> ITranslationEngine:
        """
        Create engine instance with validation.
        
        Args:
            name: Engine name
            api_key: API key (or None to use environment)
            **kwargs: Engine-specific parameters
            
        Returns:
            Engine instance
            
        Raises:
            ValidationError: If engine not found or configuration invalid
        """
        name = name.lower().strip()
        engine_class = self.get(name)
        
        if engine_class is None:
            available = ', '.join(self.list_engines())
            raise ValidationError(
                f"Unknown engine: {name}. Available: {available}"
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            env_key = f"{name.upper()}_API_KEY"
            api_key = os.getenv(env_key)
            
            if not api_key:
                raise ValidationError(
                    f"API key required for {name}. "
                    f"Set {env_key} environment variable or provide api_key parameter"
                )
        
        # Validate API key format
        self._validate_api_key(name, api_key)
        
        # Create engine
        try:
            engine = engine_class(api_key=api_key, **kwargs)
        except Exception as e:
            raise ValidationError(f"Failed to create {name} engine: {e}") from e
        
        # Validate engine configuration
        try:
            if not engine.validate_config():
                raise ValidationError(f"{name} engine configuration invalid")
        except Exception as e:
            raise ValidationError(f"Engine validation failed: {e}") from e
        
        logger.info(f"✓ Created engine: {name} ({engine.model_name})")
        return engine
    
    def list_engines(self) -> List[str]:
        """Get list of registered engine names."""
        return self.list_keys()
    
    def is_available(self, name: str) -> bool:
        """Check if engine is available."""
        return self.has(name.lower())
    
    def _validate_api_key(self, engine: str, api_key: str) -> None:
        """
        Validate API key format.
        
        Args:
            engine: Engine name
            api_key: API key to validate
            
        Raises:
            ValidationError: If API key invalid
        """
        if not api_key or not api_key.strip():
            raise ValidationError(f"API key for {engine} cannot be empty")
        
        if len(api_key) < 20:
            logger.warning(f"API key for {engine} seems too short: {len(api_key)} chars")
        
        # Engine-specific validation
        if engine == "openai":
            if not api_key.startswith("sk-"):
                logger.warning("OpenAI API keys typically start with 'sk-'")
        
        elif engine == "deepl":
            if "-" not in api_key:
                logger.warning("DeepL API keys typically contain hyphens")


# === Global Factory Instances ===

_parser_factory: Optional[ParserFactory] = None
_formatter_factory: Optional[FormatterFactory] = None
_engine_factory: Optional[EngineFactory] = None
_factories_lock = threading.Lock()


def get_parser_factory() -> ParserFactory:
    """Get global parser factory instance."""
    global _parser_factory
    
    if _parser_factory is None:
        with _factories_lock:
            if _parser_factory is None:
                _parser_factory = ParserFactory()
                _register_default_parsers(_parser_factory)
    
    return _parser_factory


def get_formatter_factory() -> FormatterFactory:
    """Get global formatter factory instance."""
    global _formatter_factory
    
    if _formatter_factory is None:
        with _factories_lock:
            if _formatter_factory is None:
                _formatter_factory = FormatterFactory()
                _register_default_formatters(_formatter_factory)
    
    return _formatter_factory


def get_engine_factory() -> EngineFactory:
    """Get global engine factory instance."""
    global _engine_factory
    
    if _engine_factory is None:
        with _factories_lock:
            if _engine_factory is None:
                _engine_factory = EngineFactory()
                _register_default_engines(_engine_factory)
    
    return _engine_factory


# === Auto-Registration Functions ===

def _register_default_parsers(factory: ParserFactory) -> None:
    """Auto-register default parsers."""
    try:
        from ..parsers.docx_parser import DocxParser
        factory.register(FileType.DOCX, DocxParser)
    except ImportError as e:
        logger.warning(f"DOCX parser not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register DOCX parser: {e}")
    
    try:
        from ..parsers.xlsx_parser import XlsxParser
        factory.register(FileType.XLSX, XlsxParser)
    except ImportError as e:
        logger.warning(f"XLSX parser not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register XLSX parser: {e}")


def _register_default_formatters(factory: FormatterFactory) -> None:
    """Auto-register default formatters (use enhanced versions)."""
    try:
        from ..formatters.docx_formatter import EnhancedDocxFormatter
        factory.register(FileType.DOCX, EnhancedDocxFormatter)
    except ImportError:
        # Fallback to standard formatter
        try:
            from ..formatters.docx_formatter import DocxFormatter
            factory.register(FileType.DOCX, DocxFormatter)
            logger.warning("Using standard DOCX formatter (enhanced version not available)")
        except ImportError as e:
            logger.warning(f"DOCX formatter not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register DOCX formatter: {e}")
    
    try:
        from ..formatters.xlsx_formatter import EnhancedXlsxFormatter
        factory.register(FileType.XLSX, EnhancedXlsxFormatter)
    except ImportError:
        # Fallback to standard formatter
        try:
            from ..formatters.xlsx_formatter import XlsxFormatter
            factory.register(FileType.XLSX, XlsxFormatter)
            logger.warning("Using standard XLSX formatter (enhanced version not available)")
        except ImportError as e:
            logger.warning(f"XLSX formatter not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register XLSX formatter: {e}")


def _register_default_engines(factory: EngineFactory) -> None:
    """Auto-register default engines."""
    try:
        from ..engines.openai_engine import OpenAIEngine
        factory.register('openai', OpenAIEngine)
    except ImportError as e:
        logger.warning(f"OpenAI engine not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register OpenAI engine: {e}")
    
    try:
        from ..engines.deepl_engine import DeepLEngine
        factory.register('deepl', DeepLEngine)
    except ImportError as e:
        logger.warning(f"DeepL engine not available: {e}")
    except Exception as e:
        logger.error(f"Failed to register DeepL engine: {e}")


# === Translation System Factory ===

class TranslationSystemFactory:
    """
    Main factory for creating complete translation systems.
    Handles all component creation and wiring with comprehensive validation.
    """
    
    @staticmethod
    def create_pipeline(
        engine_name: str,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        glossary_enabled: bool = True,
        cache_db_path: str = "data/cache.db",
        glossary_db_path: str = "data/glossary.db",
        cache_ttl_days: int = 180,
        log_level: str = "INFO",
        temp_dir: Optional[str] = None,
        **engine_kwargs
    ) -> ITranslationPipeline:
        """
        Create fully configured translation pipeline with validation.
        
        Args:
            engine_name: Name of translation engine ('openai', 'deepl')
            api_key: API key (or None to use environment variable)
            cache_enabled: Enable translation cache
            glossary_enabled: Enable glossary processing
            cache_db_path: Path to cache database
            glossary_db_path: Path to glossary database
            cache_ttl_days: Cache TTL in days
            log_level: Logging level
            temp_dir: Temporary directory for intermediate files
            **engine_kwargs: Additional engine parameters (model, pro, etc.)
            
        Returns:
            Configured TranslationPipeline
            
        Raises:
            ValidationError: If configuration is invalid
            ConfigurationError: If component creation fails
        """
        logger.info(f"Creating translation pipeline: engine={engine_name}")
        
        try:
            # Get factories
            engine_factory = get_engine_factory()
            parser_factory = get_parser_factory()
            formatter_factory = get_formatter_factory()
            
            # Validate factories have components
            if not parser_factory.get_supported_types():
                raise ConfigurationError("No parsers available")
            
            if not formatter_factory.get_supported_types():
                raise ConfigurationError("No formatters available")
            
            if not engine_factory.list_engines():
                raise ConfigurationError("No engines available")
            
            # Create translation engine with validation
            engine = engine_factory.get_engine(
                engine_name,
                api_key=api_key,
                **engine_kwargs
            )
            
            # Create cache if enabled
            cache = None
            if cache_enabled:
                logger.info("Setting up cache...")
                cache = TranslationSystemFactory._create_cache(
                    cache_db_path,
                    cache_ttl_days,
                    log_level
                )
                
                if cache is None:
                    logger.warning("Cache creation failed, continuing without cache")
            
            # Create glossary if enabled
            glossary = None
            if glossary_enabled:
                logger.info("Setting up glossary...")
                glossary = TranslationSystemFactory._create_glossary(
                    glossary_db_path,
                    log_level
                )
                
                if glossary is None:
                    logger.warning("Glossary creation failed, continuing without glossary")
            
            # Create pipeline
            pipeline = EnhancedTranslationPipeline(
                engine=engine,
                parser_factory=parser_factory,
                formatter_factory=formatter_factory,
                cache=cache,
                glossary=glossary,
                temp_dir=Path(temp_dir) if temp_dir else None
            )
            
            # Verify pipeline health
            health = pipeline.get_health()
            if health['status'] == 'error':
                raise ConfigurationError(f"Pipeline unhealthy: {health}")
            
            logger.info("✓ Translation system ready")
            logger.info(f"  Supported formats: {', '.join(ft.value for ft in pipeline.get_supported_file_types())}")
            logger.info(f"  Cache: {'enabled' if cache else 'disabled'}")
            logger.info(f"  Glossary: {'enabled' if glossary else 'disabled'}")
            
            return pipeline
            
        except (ValidationError, ConfigurationError):
            raise
        except Exception as e:
            raise ConfigurationError(f"Failed to create pipeline: {e}") from e
    
    @staticmethod
    def _create_cache(
        db_path: str,
        ttl_days: int,
        log_level: str
    ) -> Optional[ITranslationCache]:
        """Create cache adapter with error handling."""
        try:
            from ..cache.cache_manager import (
                CacheConfig,
                SQLiteStorage,
                CacheManager
            )
            from .interfaces import CacheAdapter
            
            config = CacheConfig(
                max_age_days=ttl_days,
                log_level=log_level
            )
            
            from ..utils.logger import setup_logging
            cache_logger = setup_logging("cache", log_level=log_level)
            
            storage = SQLiteStorage(Path(db_path), cache_logger)
            cache_manager = CacheManager(storage, config, cache_logger)
            
            return CacheAdapter(cache_manager)
            
        except ImportError as e:
            logger.error(f"Cache dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create cache: {e}")
            return None
    
    @staticmethod
    def _create_glossary(
        db_path: str,
        log_level: str
    ) -> Optional[IGlossaryProcessor]:
        """Create glossary adapter with error handling."""
        try:
            from ..glossary.glossary_manager import (
                GlossaryConfig,
                SQLiteGlossary,
                GlossaryManager
            )
            from .interfaces import GlossaryAdapter
            
            config = GlossaryConfig(log_level=log_level)
            
            from ..utils.logger import setup_logging
            glossary_logger = setup_logging("glossary", log_level=log_level)
            
            storage = SQLiteGlossary(Path(db_path), glossary_logger)
            glossary_manager = GlossaryManager(storage, config, glossary_logger)
            
            return GlossaryAdapter(glossary_manager)
            
        except ImportError as e:
            logger.error(f"Glossary dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to create glossary: {e}")
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
        engine_factory = get_engine_factory()
        return engine_factory.get_engine(engine_name, api_key=api_key, **kwargs)
