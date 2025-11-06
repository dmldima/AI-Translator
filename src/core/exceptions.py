"""
Custom exceptions for the translation system.
Provides clear error hierarchy and meaningful error messages.

Version: 2.1 (Optimized & Production-Ready)
"""

from contextlib import contextmanager
from typing import Optional, Dict, Any, Type
import logging

__all__ = [
    # Base
    'TranslationSystemError',
    # Engine
    'EngineError', 'TranslationError', 'APIError', 'RateLimitError',
    'QuotaExceededError', 'InvalidLanguageError', 'AuthenticationError',
    # Parser
    'ParserError', 'UnsupportedFileTypeError', 'CorruptedFileError',
    'InvalidDocumentError',
    # Formatter
    'FormatterError', 'FormattingError', 'OutputError',
    # Cache
    'CacheError', 'CacheReadError', 'CacheWriteError', 'CacheDatabaseError',
    # Glossary
    'GlossaryError', 'GlossaryReadError', 'GlossaryWriteError', 'InvalidTermError',
    # Pipeline
    'PipelineError', 'TranslationPipelineError', 'ConfigurationError',
    # Validation
    'ValidationError', 'InvalidInputError', 'InvalidConfigError',
    # Batch
    'BatchProcessingError', 'TaskFailedError',
    # Utilities
    'error_context', 'wrap_error', 'is_retryable_error',
]


# ============================================================================
# BASE EXCEPTION
# ============================================================================

class TranslationSystemError(Exception):
    """
    Base exception for all translation system errors.
    
    All custom exceptions inherit from this class for unified error handling.
    """
    
    def __init__(self, message: str, **context: Any) -> None:
        """
        Initialize exception with message and optional context.
        
        Args:
            message: Error message
            **context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.context = context
    
    def __str__(self) -> str:
        """String representation with context."""
        if not self.context:
            return self.message
        
        context_str = ', '.join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({context_str})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'context': self.context
        }


# ============================================================================
# ENGINE EXCEPTIONS
# ============================================================================

class EngineError(TranslationSystemError):
    """Base exception for translation engine errors."""
    pass


class TranslationError(EngineError):
    """Raised when translation operation fails or returns invalid results."""
    pass


class APIError(EngineError):
    """Raised for network issues, timeouts, or API communication problems."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **context: Any) -> None:
        super().__init__(message, status_code=status_code, **context)
        self.status_code = status_code


class RateLimitError(EngineError):
    """Raised when API rate limits are exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **context: Any) -> None:
        super().__init__(message, retry_after=retry_after, **context)
        self.retry_after = retry_after


class QuotaExceededError(EngineError):
    """Raised when API quota/credits are exhausted."""
    
    def __init__(self, message: str, quota_type: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, quota_type=quota_type, **context)
        self.quota_type = quota_type


class InvalidLanguageError(EngineError):
    """Raised when language code is not supported or language pair is invalid."""
    
    def __init__(
        self,
        message: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **context: Any
    ) -> None:
        super().__init__(message, source_lang=source_lang, target_lang=target_lang, **context)
        self.source_lang = source_lang
        self.target_lang = target_lang


class AuthenticationError(EngineError):
    """Raised when API key is invalid, expired, or missing."""
    
    def __init__(self, message: str, engine: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, engine=engine, **context)
        self.engine = engine


# ============================================================================
# PARSER EXCEPTIONS
# ============================================================================

class ParserError(TranslationSystemError):
    """Base exception for document parsing errors."""
    pass


class UnsupportedFileTypeError(ParserError):
    """Raised when attempting to parse an unsupported file format."""
    
    def __init__(self, message: str, file_type: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, file_type=file_type, **context)
        self.file_type = file_type


class CorruptedFileError(ParserError):
    """Raised when file structure is damaged or invalid."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, file_path=file_path, **context)
        self.file_path = file_path


class InvalidDocumentError(ParserError):
    """Raised when document doesn't meet expected structure."""
    pass


# ============================================================================
# FORMATTER EXCEPTIONS
# ============================================================================

class FormatterError(TranslationSystemError):
    """Base exception for document formatting errors."""
    pass


class FormattingError(FormatterError):
    """Raised when formatting fails or produces invalid output."""
    pass


class OutputError(FormatterError):
    """Raised when output file cannot be created or written."""
    
    def __init__(self, message: str, output_path: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, output_path=output_path, **context)
        self.output_path = output_path


# ============================================================================
# CACHE EXCEPTIONS
# ============================================================================

class CacheError(TranslationSystemError):
    """Base exception for cache errors."""
    pass


class CacheReadError(CacheError):
    """Raised when cache lookup fails."""
    pass


class CacheWriteError(CacheError):
    """Raised when cache storage fails."""
    pass


class CacheDatabaseError(CacheError):
    """Raised for SQLite or database errors in cache operations."""
    
    def __init__(self, message: str, db_path: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, db_path=db_path, **context)
        self.db_path = db_path


# ============================================================================
# GLOSSARY EXCEPTIONS
# ============================================================================

class GlossaryError(TranslationSystemError):
    """Base exception for glossary errors."""
    pass


class GlossaryReadError(GlossaryError):
    """Raised when glossary cannot be loaded or read."""
    pass


class GlossaryWriteError(GlossaryError):
    """Raised when glossary cannot be saved or updated."""
    pass


class InvalidTermError(GlossaryError):
    """Raised when term doesn't meet validation requirements."""
    
    def __init__(self, message: str, term: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, term=term, **context)
        self.term = term


# ============================================================================
# PIPELINE EXCEPTIONS
# ============================================================================

class PipelineError(TranslationSystemError):
    """Base exception for pipeline errors."""
    pass


class TranslationPipelineError(PipelineError):
    """Raised for general pipeline failures during document translation."""
    
    def __init__(self, message: str, stage: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, stage=stage, **context)
        self.stage = stage


class ConfigurationError(PipelineError):
    """Raised when pipeline or component configuration is invalid."""
    
    def __init__(self, message: str, component: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, component=component, **context)
        self.component = component


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(TranslationSystemError):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, field: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, field=field, **context)
        self.field = field


class InvalidInputError(ValidationError):
    """Raised when user input doesn't meet requirements."""
    pass


class InvalidConfigError(ValidationError):
    """Raised when configuration values are invalid."""
    pass


# ============================================================================
# BATCH PROCESSING EXCEPTIONS
# ============================================================================

class BatchProcessingError(TranslationSystemError):
    """Raised when batch operation fails."""
    
    def __init__(
        self,
        message: str,
        batch_index: Optional[int] = None,
        total_batches: Optional[int] = None,
        **context: Any
    ) -> None:
        super().__init__(message, batch_index=batch_index, total_batches=total_batches, **context)
        self.batch_index = batch_index
        self.total_batches = total_batches


class TaskFailedError(BatchProcessingError):
    """Raised when a single task in batch processing fails."""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **context: Any) -> None:
        super().__init__(message, task_id=task_id, **context)
        self.task_id = task_id


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def wrap_error(
    error: Exception,
    error_class: Type[TranslationSystemError],
    message: Optional[str] = None
) -> TranslationSystemError:
    """
    Wrap an exception in a custom exception type.
    
    Args:
        error: Original exception
        error_class: Exception class to wrap with
        message: Optional custom message
        
    Returns:
        Wrapped exception
        
    Raises:
        TypeError: If error_class is not a TranslationSystemError subclass
        
    Example:
        >>> try:
        ...     raise ValueError("Invalid")
        ... except ValueError as e:
        ...     wrapped = wrap_error(e, ValidationError, "Input validation failed")
        ...     raise wrapped
    """
    if not issubclass(error_class, TranslationSystemError):
        raise TypeError(
            f"error_class must be subclass of TranslationSystemError, "
            f"got {error_class.__name__}"
        )
    
    error_msg = f"{message}: {error}" if message else str(error)
    return error_class(error_msg) from error


@contextmanager
def error_context(
    operation: str,
    error_class: Type[TranslationSystemError] = TranslationSystemError,
    logger: Optional[logging.Logger] = None
):
    """
    Context manager for consistent error handling and wrapping.
    
    Args:
        operation: Name of operation being performed
        error_class: Exception class to wrap errors with
        logger: Optional logger for error logging
        
    Yields:
        None
        
    Raises:
        error_class: Wrapped exception if error occurs
        
    Example:
        >>> with error_context("parsing document", ParserError):
        ...     parse_document()
    """
    try:
        yield
    except error_class:
        # Already the correct type, just reraise
        raise
    except KeyboardInterrupt:
        # Never catch keyboard interrupt
        raise
    except Exception as e:
        # Log if logger provided
        if logger:
            logger.error(f"Error during {operation}: {e}", exc_info=True)
        
        # Wrap and raise
        raise wrap_error(e, error_class, f"Error during {operation}")


def is_retryable_error(error: Exception) -> bool:
    """
    Check if error is retryable (network, rate limit, transient issues).
    
    Args:
        error: Exception to check
        
    Returns:
        True if error should be retried
        
    Example:
        >>> try:
        ...     api_call()
        ... except Exception as e:
        ...     if is_retryable_error(e):
        ...         retry()
    """
    # Retryable error types
    retryable_types = (
        RateLimitError,
        APIError,
        CacheReadError,
        CacheDatabaseError,
    )
    
    if isinstance(error, retryable_types):
        return True
    
    # Check for transient error indicators in message
    error_msg = str(error).lower()
    transient_keywords = (
        'timeout',
        'connection',
        'network',
        'temporary',
        'unavailable',
        'try again',
    )
    
    return any(keyword in error_msg for keyword in transient_keywords)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

def _run_tests() -> None:
    """Run basic tests for the exception system."""
    
    print("=" * 70)
    print("TESTING EXCEPTION SYSTEM")
    print("=" * 70)
    
    # Test 1: Basic exception with context
    print("\n1. Basic exception with context:")
    try:
        raise APIError("Connection timeout", status_code=504, endpoint="/translate")
    except APIError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Status: {e.status_code}")
        print(f"   ✓ Dict: {e.to_dict()}")
        assert e.status_code == 504
    
    # Test 2: Error wrapping
    print("\n2. Error wrapping:")
    try:
        try:
            raise ValueError("Invalid language code: xyz")
        except ValueError as e:
            raise wrap_error(e, InvalidLanguageError, "Language validation failed")
    except InvalidLanguageError as e:
        print(f"   ✓ Wrapped: {e}")
        print(f"   ✓ Cause: {e.__cause__}")
        assert e.__cause__ is not None
    
    # Test 3: Context manager
    print("\n3. Context manager:")
    try:
        with error_context("test operation", ParserError):
            raise ValueError("Something went wrong")
    except ParserError as e:
        print(f"   ✓ Caught and wrapped: {e}")
        assert "test operation" in str(e)
    
    # Test 4: Retryable errors
    print("\n4. Retryable error detection:")
    errors = [
        (RateLimitError("Rate limit"), True),
        (APIError("API timeout"), True),
        (CorruptedFileError("File corrupted"), False),
        (CacheReadError("Cache miss"), True),
    ]
    
    for error, expected in errors:
        result = is_retryable_error(error)
        status = "✓" if result == expected else "✗"
        print(f"   {status} {error.__class__.__name__}: retryable={result}")
        assert result == expected
    
    # Test 5: RateLimitError with retry_after
    print("\n5. RateLimitError with retry_after:")
    try:
        raise RateLimitError("Too many requests", retry_after=30)
    except RateLimitError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Retry after: {e.retry_after}s")
        assert e.retry_after == 30
    
    # Test 6: InvalidLanguageError with languages
    print("\n6. InvalidLanguageError:")
    try:
        raise InvalidLanguageError(
            "Unsupported language pair",
            source_lang="en",
            target_lang="xyz"
        )
    except InvalidLanguageError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Source: {e.source_lang}, Target: {e.target_lang}")
        assert e.source_lang == "en"
        assert e.target_lang == "xyz"
    
    # Test 7: BatchProcessingError
    print("\n7. BatchProcessingError:")
    try:
        raise BatchProcessingError(
            "Batch failed",
            batch_index=5,
            total_batches=10
        )
    except BatchProcessingError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Progress: {e.batch_index}/{e.total_batches}")
        assert e.batch_index == 5
        assert e.total_batches == 10
    
    # Test 8: to_dict serialization
    print("\n8. Exception serialization:")
    error = TranslationError("Translation failed", model="gpt-4", lang="en")
    error_dict = error.to_dict()
    print(f"   ✓ Serialized: {error_dict}")
    assert error_dict['error_type'] == 'TranslationError'
    assert error_dict['message'] == 'Translation failed'
    assert error_dict['context']['model'] == 'gpt-4'
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    _run_tests()
