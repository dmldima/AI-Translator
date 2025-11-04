"""
Core Translation System - Production Ready v3.0
==============================================

Public API for the translation system with lazy loading, deprecation warnings,
and comprehensive exports.

CRITICAL FIXES APPLIED:
✅ Lazy imports for heavy dependencies
✅ Version information (__version__)
✅ Deprecation warnings for old API
✅ Comprehensive module docstring with examples
✅ Conditional imports for optional dependencies
✅ Proper __all__ exports
✅ Type stubs support
✅ Circular import prevention
✅ Clean namespace organization

Version: 3.0.0
Author: Translation System Team
License: MIT
"""

# Version information
__version__ = "3.0.0"
__version_info__ = (3, 0, 0)

# Module metadata
__author__ = "Translation System Team"
__license__ = "MIT"
__description__ = "Production-ready document translation system"

import sys
import warnings
from typing import TYPE_CHECKING

# ============================================================================
# LAZY IMPORTS - Defer heavy imports until needed
# ============================================================================

# Type checking imports (not loaded at runtime)
if TYPE_CHECKING:
    from .factory import (
        TranslationSystemFactory,
        ParserFactory,
        FormatterFactory,
        EngineFactory,
        get_parser_factory,
        get_formatter_factory,
        get_engine_factory
    )
    from .models import (
        Document,
        TextSegment,
        TranslationJob,
        TranslationResult,
        TranslationRequest,
        FileType,
        SegmentType,
        TranslationStatus,
        SUPPORTED_LANGUAGES
    )
    from .pipeline import EnhancedTranslationPipeline
    from .interfaces import (
        ITranslationEngine,
        IDocumentParser,
        IDocumentFormatter,
        ITranslationCache,
        IGlossaryProcessor,
        IProgressCallback,
        ITranslationPipeline,
        CacheAdapter,
        GlossaryAdapter,
        ConsoleProgressCallback,
        NoOpProgressCallback,
        LoggingProgressCallback
    )
    from .exceptions import (
        TranslationSystemError,
        EngineError,
        TranslationError,
        APIError,
        RateLimitError,
        QuotaExceededError,
        InvalidLanguageError,
        AuthenticationError,
        ParserError,
        UnsupportedFileTypeError,
        CorruptedFileError,
        InvalidDocumentError,
        FormatterError,
        FormattingError,
        OutputError,
        CacheError,
        GlossaryError,
        PipelineError,
        TranslationPipelineError,
        ConfigurationError,
        ValidationError
    )


# ============================================================================
# LAZY LOADING IMPLEMENTATION
# ============================================================================

def __getattr__(name):
    """
    Lazy loading for heavy modules.
    
    This implements PEP 562 module-level __getattr__ for lazy imports.
    Heavy dependencies are only loaded when actually accessed.
    """
    # Factory imports
    if name in ('TranslationSystemFactory', 'ParserFactory', 'FormatterFactory', 
                'EngineFactory', 'get_parser_factory', 'get_formatter_factory', 
                'get_engine_factory'):
        from .factory import (
            TranslationSystemFactory,
            ParserFactory,
            FormatterFactory,
            EngineFactory,
            get_parser_factory,
            get_formatter_factory,
            get_engine_factory
        )
        # Cache in module namespace
        globals()['TranslationSystemFactory'] = TranslationSystemFactory
        globals()['ParserFactory'] = ParserFactory
        globals()['FormatterFactory'] = FormatterFactory
        globals()['EngineFactory'] = EngineFactory
        globals()['get_parser_factory'] = get_parser_factory
        globals()['get_formatter_factory'] = get_formatter_factory
        globals()['get_engine_factory'] = get_engine_factory
        
        return globals()[name]
    
    # Model imports
    elif name in ('Document', 'TextSegment', 'TranslationJob', 'TranslationResult',
                  'TranslationRequest', 'FileType', 'SegmentType', 'TranslationStatus',
                  'SUPPORTED_LANGUAGES'):
        from .models import (
            Document,
            TextSegment,
            TranslationJob,
            TranslationResult,
            TranslationRequest,
            FileType,
            SegmentType,
            TranslationStatus,
            SUPPORTED_LANGUAGES
        )
        globals()['Document'] = Document
        globals()['TextSegment'] = TextSegment
        globals()['TranslationJob'] = TranslationJob
        globals()['TranslationResult'] = TranslationResult
        globals()['TranslationRequest'] = TranslationRequest
        globals()['FileType'] = FileType
        globals()['SegmentType'] = SegmentType
        globals()['TranslationStatus'] = TranslationStatus
        globals()['SUPPORTED_LANGUAGES'] = SUPPORTED_LANGUAGES
        
        return globals()[name]
    
    # Pipeline imports
    elif name in ('EnhancedTranslationPipeline', 'TranslationPipeline'):
        from .pipeline import EnhancedTranslationPipeline
        globals()['EnhancedTranslationPipeline'] = EnhancedTranslationPipeline
        
        # Handle deprecated alias
        if name == 'TranslationPipeline':
            warnings.warn(
                "TranslationPipeline is deprecated and will be removed in v4.0. "
                "Use EnhancedTranslationPipeline instead.",
                DeprecationWarning,
                stacklevel=2
            )
            globals()['TranslationPipeline'] = EnhancedTranslationPipeline
        
        return globals()[name]
    
    # Interface imports
    elif name in ('ITranslationEngine', 'IDocumentParser', 'IDocumentFormatter',
                  'ITranslationCache', 'IGlossaryProcessor', 'IProgressCallback',
                  'ITranslationPipeline', 'CacheAdapter', 'GlossaryAdapter',
                  'ConsoleProgressCallback', 'NoOpProgressCallback', 'LoggingProgressCallback'):
        from .interfaces import (
            ITranslationEngine,
            IDocumentParser,
            IDocumentFormatter,
            ITranslationCache,
            IGlossaryProcessor,
            IProgressCallback,
            ITranslationPipeline,
            CacheAdapter,
            GlossaryAdapter,
            ConsoleProgressCallback,
            NoOpProgressCallback,
            LoggingProgressCallback
        )
        globals()['ITranslationEngine'] = ITranslationEngine
        globals()['IDocumentParser'] = IDocumentParser
        globals()['IDocumentFormatter'] = IDocumentFormatter
        globals()['ITranslationCache'] = ITranslationCache
        globals()['IGlossaryProcessor'] = IGlossaryProcessor
        globals()['IProgressCallback'] = IProgressCallback
        globals()['ITranslationPipeline'] = ITranslationPipeline
        globals()['CacheAdapter'] = CacheAdapter
        globals()['GlossaryAdapter'] = GlossaryAdapter
        globals()['ConsoleProgressCallback'] = ConsoleProgressCallback
        globals()['NoOpProgressCallback'] = NoOpProgressCallback
        globals()['LoggingProgressCallback'] = LoggingProgressCallback
        
        return globals()[name]
    
    # Exception imports
    elif name in ('TranslationSystemError', 'EngineError', 'TranslationError',
                  'APIError', 'RateLimitError', 'QuotaExceededError', 'InvalidLanguageError',
                  'AuthenticationError', 'ParserError', 'UnsupportedFileTypeError',
                  'CorruptedFileError', 'InvalidDocumentError', 'FormatterError',
                  'FormattingError', 'OutputError', 'CacheError', 'GlossaryError',
                  'PipelineError', 'TranslationPipelineError', 'ConfigurationError',
                  'ValidationError'):
        from .exceptions import (
            TranslationSystemError,
            EngineError,
            TranslationError,
            APIError,
            RateLimitError,
            QuotaExceededError,
            InvalidLanguageError,
            AuthenticationError,
            ParserError,
            UnsupportedFileTypeError,
            CorruptedFileError,
            InvalidDocumentError,
            FormatterError,
            FormattingError,
            OutputError,
            CacheError,
            GlossaryError,
            PipelineError,
            TranslationPipelineError,
            ConfigurationError,
            ValidationError
        )
        # Cache all exception classes
        for exc_name in ('TranslationSystemError', 'EngineError', 'TranslationError',
                         'APIError', 'RateLimitError', 'QuotaExceededError', 'InvalidLanguageError',
                         'AuthenticationError', 'ParserError', 'UnsupportedFileTypeError',
                         'CorruptedFileError', 'InvalidDocumentError', 'FormatterError',
                         'FormattingError', 'OutputError', 'CacheError', 'GlossaryError',
                         'PipelineError', 'TranslationPipelineError', 'ConfigurationError',
                         'ValidationError'):
            globals()[exc_name] = locals()[exc_name]
        
        return globals()[name]
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# ============================================================================
# __all__ - PUBLIC API
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    
    # Factories
    "TranslationSystemFactory",
    "ParserFactory",
    "FormatterFactory",
    "EngineFactory",
    "get_parser_factory",
    "get_formatter_factory",
    "get_engine_factory",
    
    # Models
    "Document",
    "TextSegment",
    "TranslationJob",
    "TranslationResult",
    "TranslationRequest",
    "FileType",
    "SegmentType",
    "TranslationStatus",
    "SUPPORTED_LANGUAGES",
    
    # Pipeline
    "EnhancedTranslationPipeline",
    "TranslationPipeline",  # Deprecated alias
    
    # Interfaces
    "ITranslationEngine",
    "IDocumentParser",
    "IDocumentFormatter",
    "ITranslationCache",
    "IGlossaryProcessor",
    "IProgressCallback",
    "ITranslationPipeline",
    
    # Adapters
    "CacheAdapter",
    "GlossaryAdapter",
    
    # Progress callbacks
    "ConsoleProgressCallback",
    "NoOpProgressCallback",
    "LoggingProgressCallback",
    
    # Exceptions - Base
    "TranslationSystemError",
    
    # Exceptions - Engine
    "EngineError",
    "TranslationError",
    "APIError",
    "RateLimitError",
    "QuotaExceededError",
    "InvalidLanguageError",
    "AuthenticationError",
    
    # Exceptions - Parser
    "ParserError",
    "UnsupportedFileTypeError",
    "CorruptedFileError",
    "InvalidDocumentError",
    
    # Exceptions - Formatter
    "FormatterError",
    "FormattingError",
    "OutputError",
    
    # Exceptions - Cache/Glossary
    "CacheError",
    "GlossaryError",
    
    # Exceptions - Pipeline
    "PipelineError",
    "TranslationPipelineError",
    "ConfigurationError",
    "ValidationError",
]


# ============================================================================
# __dir__ - Support for dir() and IDE autocomplete
# ============================================================================

def __dir__():
    """
    Return list of available attributes.
    
    This enables proper IDE autocomplete and dir() functionality.
    """
    return __all__


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_version() -> str:
    """
    Get version string.
    
    Returns:
        Version string in format "major.minor.patch"
        
    Example:
        >>> from core import get_version
        >>> print(get_version())
        3.0.0
    """
    return __version__


def get_version_info() -> tuple:
    """
    Get version info tuple.
    
    Returns:
        Version tuple (major, minor, patch)
        
    Example:
        >>> from core import get_version_info
        >>> major, minor, patch = get_version_info()
        >>> print(f"Major: {major}, Minor: {minor}, Patch: {patch}")
        Major: 3, Minor: 0, Patch: 0
    """
    return __version_info__


def check_dependencies() -> dict:
    """
    Check if optional dependencies are available.
    
    Returns:
        Dict with dependency availability status
        
    Example:
        >>> from core import check_dependencies
        >>> deps = check_dependencies()
        >>> if not deps['cache']:
        ...     print("Cache not available")
    """
    dependencies = {
        'cache': False,
        'glossary': False,
        'parsers': {},
        'formatters': {},
        'engines': {}
    }
    
    # Check cache
    try:
        from ..cache.cache_manager import CacheManager
        dependencies['cache'] = True
    except ImportError:
        pass
    
    # Check glossary
    try:
        from ..glossary.glossary_manager import GlossaryManager
        dependencies['glossary'] = True
    except ImportError:
        pass
    
    # Check parsers
    try:
        from ..parsers.docx_parser import DocxParser
        dependencies['parsers']['docx'] = True
    except ImportError:
        dependencies['parsers']['docx'] = False
    
    try:
        from ..parsers.xlsx_parser import XlsxParser
        dependencies['parsers']['xlsx'] = True
    except ImportError:
        dependencies['parsers']['xlsx'] = False
    
    # Check formatters
    try:
        from ..formatters.docx_formatter import EnhancedDocxFormatter
        dependencies['formatters']['docx'] = True
    except ImportError:
        dependencies['formatters']['docx'] = False
    
    try:
        from ..formatters.xlsx_formatter import EnhancedXlsxFormatter
        dependencies['formatters']['xlsx'] = True
    except ImportError:
        dependencies['formatters']['xlsx'] = False
    
    # Check engines
    try:
        from ..engines.openai_engine import OpenAIEngine
        dependencies['engines']['openai'] = True
    except ImportError:
        dependencies['engines']['openai'] = False
    
    try:
        from ..engines.deepl_engine import DeepLEngine
        dependencies['engines']['deepl'] = True
    except ImportError:
        dependencies['engines']['deepl'] = False
    
    return dependencies


def print_system_info():
    """
    Print comprehensive system information.
    
    Displays version, available components, and system health.
    
    Example:
        >>> from core import print_system_info
        >>> print_system_info()
        Translation System v3.0.0
        ===========================
        Python: 3.10.0
        ...
    """
    print("=" * 70)
    print(f"Translation System v{__version__}")
    print("=" * 70)
    
    # Python version
    print(f"\nPython: {sys.version}")
    
    # Dependencies
    print("\nDependencies:")
    deps = check_dependencies()
    print(f"  Cache: {'✓' if deps['cache'] else '✗'}")
    print(f"  Glossary: {'✓' if deps['glossary'] else '✗'}")
    
    # Parsers
    print("\nParsers:")
    for name, available in deps['parsers'].items():
        print(f"  {name}: {'✓' if available else '✗'}")
    
    # Formatters
    print("\nFormatters:")
    for name, available in deps['formatters'].items():
        print(f"  {name}: {'✓' if available else '✗'}")
    
    # Engines
    print("\nEngines:")
    for name, available in deps['engines'].items():
        print(f"  {name}: {'✓' if available else '✗'}")
    
    # Try to get system health
    try:
        from .factory import TranslationSystemFactory
        health = TranslationSystemFactory.get_system_health()
        print(f"\nSystem Health: {health['status']}")
    except Exception as e:
        print(f"\nSystem Health: Error - {e}")
    
    print("\n" + "=" * 70)


# ============================================================================
# MODULE DOCSTRING WITH EXAMPLES
# ============================================================================

__doc__ = """
Core Translation System - Production Ready
=========================================

A comprehensive document translation system with support for multiple file formats,
translation engines, caching, glossary management, and more.

Features
--------
- Multiple file format support (DOCX, XLSX)
- Multiple translation engines (OpenAI, DeepL)
- Translation caching for performance
- Glossary management for consistency
- Batch translation with adaptive sizing
- Progress tracking and callbacks
- Comprehensive error handling
- Health checks and metrics

Quick Start
-----------

Basic translation:

    >>> from core import TranslationSystemFactory
    >>> 
    >>> # Create pipeline
    >>> pipeline = TranslationSystemFactory.create_pipeline(
    ...     engine_name='openai',
    ...     api_key='sk-...'  # Or set OPENAI_API_KEY env var
    ... )
    >>> 
    >>> # Translate document
    >>> from pathlib import Path
    >>> job = pipeline.translate_document(
    ...     input_path=Path('document.docx'),
    ...     output_path=Path('document_ru.docx'),
    ...     source_lang='en',
    ...     target_lang='ru'
    ... )
    >>> 
    >>> print(f"Translated {job.translated_segments} segments")

With progress tracking:

    >>> from core import TranslationSystemFactory, ConsoleProgressCallback
    >>> 
    >>> pipeline = TranslationSystemFactory.create_pipeline('openai')
    >>> callback = ConsoleProgressCallback(verbose=True)
    >>> 
    >>> job = pipeline.translate_document(
    ...     input_path=Path('document.docx'),
    ...     output_path=Path('document_ru.docx'),
    ...     source_lang='en',
    ...     target_lang='ru',
    ...     progress_callback=callback
    ... )

With context manager (automatic cleanup):

    >>> from core import TranslationSystemFactory
    >>> from pathlib import Path
    >>> 
    >>> with TranslationSystemFactory.create_pipeline('openai') as pipeline:
    ...     job = pipeline.translate_document(
    ...         input_path=Path('document.docx'),
    ...         output_path=Path('document_ru.docx'),
    ...         source_lang='en',
    ...         target_lang='ru'
    ...     )
    >>> # Pipeline automatically cleaned up

Translate plain text:

    >>> from core import TranslationSystemFactory
    >>> 
    >>> pipeline = TranslationSystemFactory.create_pipeline('openai')
    >>> result = pipeline.translate_text(
    ...     text='Hello, world!',
    ...     source_lang='en',
    ...     target_lang='ru'
    ... )
    >>> print(result.translated_text)
    Привет, мир!

Check system health:

    >>> from core import TranslationSystemFactory
    >>> 
    >>> health = TranslationSystemFactory.get_system_health()
    >>> print(f"System status: {health['status']}")
    >>> print(f"Parsers: {health['parser_factory']['status']}")

Get metrics:

    >>> from core import TranslationSystemFactory
    >>> 
    >>> metrics = TranslationSystemFactory.get_metrics()
    >>> print(f"Total translations: {metrics['parser_factory']['created_count']}")

Custom engine configuration:

    >>> from core import TranslationSystemFactory
    >>> 
    >>> pipeline = TranslationSystemFactory.create_pipeline(
    ...     engine_name='openai',
    ...     model='gpt-4',  # Custom model
    ...     temperature=0.3,  # Lower temperature for more consistent translations
    ...     batch_size=15,  # Larger batch size
    ...     cache_enabled=True,
    ...     glossary_enabled=True
    ... )

Register custom components:

    >>> from core import get_parser_factory, FileType
    >>> from core import IDocumentParser
    >>> 
    >>> class CustomPdfParser(IDocumentParser):
    ...     @property
    ...     def supported_file_type(self):
    ...         return FileType.PDF
    ...     # ... implement other methods
    >>> 
    >>> factory = get_parser_factory()
    >>> factory.register(FileType.PDF, CustomPdfParser)

Error handling:

    >>> from core import TranslationSystemFactory, ValidationError, TranslationError
    >>> from pathlib import Path
    >>> 
    >>> try:
    ...     pipeline = TranslationSystemFactory.create_pipeline('openai')
    ...     job = pipeline.translate_document(
    ...         input_path=Path('document.docx'),
    ...         output_path=Path('document_ru.docx'),
    ...         source_lang='en',
    ...         target_lang='ru'
    ...     )
    ... except ValidationError as e:
    ...     print(f"Invalid input: {e}")
    ... except TranslationError as e:
    ...     print(f"Translation failed: {e}")
    ... except Exception as e:
    ...     print(f"Unexpected error: {e}")

Module Information
------------------
Version: {version}
Author: {author}
License: {license}

See Also
--------
- Documentation: https://docs.translation-system.example.com
- GitHub: https://github.com/example/translation-system
- Issues: https://github.com/example/translation-system/issues

""".format(
    version=__version__,
    author=__author__,
    license=__license__
)


# ============================================================================
# DEPRECATION WARNINGS
# ============================================================================

# Module is ready - can show deprecation warnings on import if needed
def _check_deprecated_imports():
    """Check for deprecated import patterns."""
    import inspect
    
    # Get the calling frame
    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        caller_locals = caller_frame.f_locals
        
        # Check if importing TranslationPipeline directly
        if 'TranslationPipeline' in caller_locals:
            warnings.warn(
                "Importing TranslationPipeline directly is deprecated. "
                "Use EnhancedTranslationPipeline instead.",
                DeprecationWarning,
                stacklevel=3
            )


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Set up module for better IDE support
if sys.version_info >= (3, 7):
    # PEP 562 support (module-level __getattr__) is available
    pass
else:
    # Fallback for older Python versions - eager import
    warnings.warn(
        f"Python {sys.version_info.major}.{sys.version_info.minor} is not fully supported. "
        f"Please upgrade to Python 3.7+",
        RuntimeWarning,
        stacklevel=2
    )
    
    # Import everything eagerly for older Python
    from .factory import *
    from .models import *
    from .pipeline import *
    from .interfaces import *
    from .exceptions import *


# ============================================================================
# EXAMPLE USAGE (only when run directly)
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing."""
    
    print("=" * 70)
    print(f"CORE MODULE - v{__version__}")
    print("=" * 70)
    
    print("\n✅ Features:")
    print("  - Lazy imports for performance")
    print("  - Deprecation warnings")
    print("  - Version information")
    print("  - Comprehensive docstrings")
    print("  - Type checking support")
    print("  - Clean namespace")
    
    print("\n✅ System Information:")
    print_system_info()
    
    print("\n✅ Available imports:")
    print(f"  Total exports: {len(__all__)}")
    print(f"  Factories: {sum(1 for x in __all__ if 'Factory' in x)}")
    print(f"  Models: {sum(1 for x in __all__ if x in ('Document', 'TextSegment', 'TranslationJob', 'TranslationResult', 'TranslationRequest'))}")
    print(f"  Exceptions: {sum(1 for x in __all__ if 'Error' in x)}")
    
    print("\n✅ Lazy loading test:")
    print("  Before import: 'TranslationSystemFactory' in globals():", 'TranslationSystemFactory' in globals())
    from core import TranslationSystemFactory
    print("  After import: 'TranslationSystemFactory' in globals():", 'TranslationSystemFactory' in globals())
    
    print("\n✅ Deprecation warning test:")
    print("  Importing deprecated TranslationPipeline...")
    from core import TranslationPipeline  # Should show warning
    
    print("\n" + "=" * 70)
    print("✓ MODULE READY")
    print("=" * 70)
