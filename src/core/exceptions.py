"""
Custom exceptions for the translation system.
Provides clear error hierarchy and meaningful error messages.

Version: 2.0 (Production-Ready)
"""

from contextlib import contextmanager
from typing import Type, Optional, Dict, Any
import logging


# ============================================================================
# BASE EXCEPTIONS
# ============================================================================

class TranslationSystemError(Exception):
    """
    Base exception for all translation system errors.
    
    All custom exceptions inherit from this class, making it easy to catch
    any system-related error.
    """
    
    def __init__(self, message: str, **kwargs):
        """
        Initialize exception with message and optional context.
        
        Args:
            message: Error message
            **kwargs: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.context = kwargs
    
    def __str__(self) -> str:
        """String representation with context."""
        if self.context:
            context_str = ', '.join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
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
    """
    Error during translation operation.
    
    Raised when the translation engine fails to translate text,
    returns invalid results, or encounters an unexpected error.
    """
    pass


class APIError(EngineError):
    """
    Error communicating with translation API.
    
    Raised for network issues, timeouts, invalid responses,
    or other API communication problems.
    """
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        """
        Initialize with optional HTTP status code.
        
        Args:
            message: Error message
            status_code: HTTP status code if available
            **kwargs: Additional context
        """
        super().__init__(message, status_code=status_code, **kwargs)
        self.status_code = status_code


class RateLimitError(EngineError):
    """
    Rate limit exceeded.
    
    Raised when API rate limits are hit. May include retry information.
    """
    
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        """
        Initialize with optional retry delay.
        
        Args:
            message: Error message
            retry_after: Seconds until retry is allowed
            **kwargs: Additional context
        """
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.retry_after = retry_after


class QuotaExceededError(EngineError):
    """
    API quota exceeded.
    
    Raised when API quota/credits are exhausted.
    """
    
    def __init__(self, message: str, quota_type: Optional[str] = None, **kwargs):
        """
        Initialize with optional quota type.
        
        Args:
            message: Error message
            quota_type: Type of quota exceeded (e.g., 'daily', 'monthly', 'tokens')
            **kwargs: Additional context
        """
        super().__init__(message, quota_type=quota_type, **kwargs)
        self.quota_type = quota_type


class InvalidLanguageError(EngineError):
    """
    Invalid or unsupported language.
    
    Raised when a language code is not supported by the engine
    or language pair is invalid.
    """
    
    def __init__(
        self, 
        message: str, 
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize with language information.
        
        Args:
            message: Error message
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional context
        """
        super().__init__(
            message, 
            source_lang=source_lang, 
            target_lang=target_lang,
            **kwargs
        )
        self.source_lang = source_lang
        self.target_lang = target_lang


class AuthenticationError(EngineError):
    """
    Authentication failed (invalid API key).
    
    Raised when API key is invalid, expired, or missing.
    """
    
    def __init__(self, message: str, engine: Optional[str] = None, **kwargs):
        """
        Initialize with engine information.
        
        Args:
            message: Error message
            engine: Name of the engine
            **kwargs: Additional context
        """
        super().__init__(message, engine=engine, **kwargs)
        self.engine = engine


# ============================================================================
# PARSER EXCEPTIONS
# ============================================================================

class ParserError(TranslationSystemError):
    """Base exception for document parsing errors."""
    pass


class UnsupportedFileTypeError(ParserError):
    """
    File type not supported.
    
    Raised when attempting to parse a file with unsupported format.
    """
    
    def __init__(self, message: str, file_type: Optional[str] = None, **kwargs):
        """
        Initialize with file type information.
        
        Args:
            message: Error message
            file_type: File extension or type
            **kwargs: Additional context
        """
        super().__init__(message, file_type=file_type, **kwargs)
        self.file_type = file_type


class CorruptedFileError(ParserError):
    """
    File is corrupted or cannot be read.
    
    Raised when file structure is damaged or invalid.
    """
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        """
        Initialize with file path.
        
        Args:
            message: Error message
            file_path: Path to corrupted file
            **kwargs: Additional context
        """
        super().__init__(message, file_path=file_path, **kwargs)
        self.file_path = file_path


class InvalidDocumentError(ParserError):
    """
    Document structure is invalid.
    
    Raised when document doesn't meet expected structure or contains
    invalid elements.
    """
    pass


# ============================================================================
# FORMATTER EXCEPTIONS
# ============================================================================

class FormatterError(TranslationSystemError):
    """Base exception for document formatting errors."""
    pass


class FormattingError(FormatterError):
    """
    Error during document formatting.
    
    Raised when formatting fails or produces invalid output.
    """
    pass


class OutputError(FormatterError):
    """
    Error writing output file.
    
    Raised when output file cannot be created or written.
    """
    
    def __init__(self, message: str, output_path: Optional[str] = None, **kwargs):
        """
        Initialize with output path.
        
        Args:
            message: Error message
            output_path: Path where output was attempted
            **kwargs: Additional context
        """
        super().__init__(message, output_path=output_path, **kwargs)
        self.output_path = output_path


# ============================================================================
# CACHE EXCEPTIONS
# ============================================================================

class CacheError(TranslationSystemError):
    """Base exception for cache errors."""
    pass


class CacheReadError(CacheError):
    """
    Error reading from cache.
    
    Raised when cache lookup fails.
    """
    pass


class CacheWriteError(CacheError):
    """
    Error writing to cache.
    
    Raised when cache storage fails.
    """
    pass


class CacheDatabaseError(CacheError):
    """
    Database error in cache.
    
    Raised for SQLite or other database errors in cache operations.
    """
    
    def __init__(self, message: str, db_path: Optional[str] = None, **kwargs):
        """
        Initialize with database path.
        
        Args:
            message: Error message
            db_path: Path to database file
            **kwargs: Additional context
        """
        super().__init__(message, db_path=db_path, **kwargs)
        self.db_path = db_path


# ============================================================================
# GLOSSARY EXCEPTIONS
# ============================================================================

class GlossaryError(TranslationSystemError):
    """Base exception for glossary errors."""
    pass


class GlossaryReadError(GlossaryError):
    """
    Error reading glossary.
    
    Raised when glossary cannot be loaded or read.
    """
    pass


class GlossaryWriteError(GlossaryError):
    """
    Error writing to glossary.
    
    Raised when glossary cannot be saved or updated.
    """
    pass


class InvalidTermError(GlossaryError):
    """
    Invalid glossary term.
    
    Raised when term doesn't meet validation requirements.
    """
    
    def __init__(self, message: str, term: Optional[str] = None, **kwargs):
        """
        Initialize with term information.
        
        Args:
            message: Error message
            term: Invalid term
            **kwargs: Additional context
        """
        super().__init__(message, term=term, **kwargs)
        self.term = term


# ============================================================================
# PIPELINE EXCEPTIONS
# ============================================================================

class PipelineError(TranslationSystemError):
    """Base exception for pipeline errors."""
    pass


class TranslationPipelineError(PipelineError):
    """
    Error in translation pipeline.
    
    Raised for general pipeline failures during document translation.
    """
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs):
        """
        Initialize with pipeline stage.
        
        Args:
            message: Error message
            stage: Pipeline stage where error occurred (e.g., 'parsing', 'translating')
            **kwargs: Additional context
        """
        super().__init__(message, stage=stage, **kwargs)
        self.stage = stage


class ConfigurationError(PipelineError):
    """
    Invalid configuration.
    
    Raised when pipeline or component configuration is invalid or missing.
    """
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        """
        Initialize with component information.
        
        Args:
            message: Error message
            component: Component with invalid configuration
            **kwargs: Additional context
        """
        super().__init__(message, component=component, **kwargs)
        self.component = component


# ============================================================================
# VALIDATION EXCEPTIONS
# ============================================================================

class ValidationError(TranslationSystemError):
    """
    Base exception for validation errors.
    
    Raised when input validation fails.
    """
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        """
        Initialize with field information.
        
        Args:
            message: Error message
            field: Field that failed validation
            **kwargs: Additional context
        """
        super().__init__(message, field=field, **kwargs)
        self.field = field


class InvalidInputError(ValidationError):
    """
    Invalid input provided.
    
    Raised when user input doesn't meet requirements.
    """
    pass


class InvalidConfigError(ValidationError):
    """
    Invalid configuration.
    
    Raised when configuration values are invalid.
    """
    pass


# ============================================================================
# BATCH PROCESSING EXCEPTIONS
# ============================================================================

class BatchProcessingError(TranslationSystemError):
    """
    Error during batch processing.
    
    Raised when batch operation fails.
    """
    
    def __init__(
        self, 
        message: str, 
        batch_index: Optional[int] = None,
        total_batches: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize with batch information.
        
        Args:
            message: Error message
            batch_index: Index of failed batch
            total_batches: Total number of batches
            **kwargs: Additional context
        """
        super().__init__(
            message, 
            batch_index=batch_index,
            total_batches=total_batches,
            **kwargs
        )
        self.batch_index = batch_index
        self.total_batches = total_batches


class TaskFailedError(BatchProcessingError):
    """
    Individual batch task failed.
    
    Raised when a single task in batch processing fails.
    """
    
    def __init__(
        self, 
        message: str, 
        task_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize with task information.
        
        Args:
            message: Error message
            task_id: ID of failed task
            **kwargs: Additional context
        """
        super().__init__(message, task_id=task_id, **kwargs)
        self.task_id = task_id


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_error_message(error: Exception, context: Optional[str] = None) -> str:
    """
    Format error message with context.
    
    Args:
        error: Exception
        context: Optional context information
        
    Returns:
        Formatted error message
        
    Example:
        >>> try:
        ...     raise ValueError("Invalid value")
        ... except ValueError as e:
        ...     msg = format_error_message(e, "validation")
        ...     print(msg)
        ValueError in validation: Invalid value
    """
    error_type = type(error).__name__
    message = str(error)
    
    if context:
        return f"{error_type} in {context}: {message}"
    else:
        return f"{error_type}: {message}"


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
    
    if message:
        return error_class(f"{message}: {error}") from error
    else:
        return error_class(str(error)) from error


@contextmanager
def error_context(
    operation: str, 
    error_class: Type[TranslationSystemError] = TranslationSystemError,
    logger: Optional[logging.Logger] = None,
    reraise: bool = True
):
    """
    Context manager for consistent error handling.
    
    Args:
        operation: Name of operation being performed
        error_class: Exception class to raise on error
        logger: Optional logger for error logging
        reraise: Whether to reraise the exception
        
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
        if reraise:
            raise
    except KeyboardInterrupt:
        # Don't catch keyboard interrupt
        raise
    except Exception as e:
        # Log if logger provided
        if logger:
            logger.error(f"Error during {operation}: {e}", exc_info=True)
        
        # Wrap in appropriate error type
        wrapped = error_class(f"Error during {operation}: {e}") from e
        
        if reraise:
            raise wrapped
        else:
            return wrapped


def is_retryable_error(error: Exception) -> bool:
    """
    Check if error is retryable (network, rate limit, etc.).
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is retryable
        
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
        CacheDatabaseError
    )
    
    if isinstance(error, retryable_types):
        return True
    
    # Check for specific error messages indicating transient issues
    error_msg = str(error).lower()
    transient_keywords = [
        'timeout',
        'connection',
        'network',
        'temporary',
        'unavailable',
        'service temporarily',
        'try again'
    ]
    
    return any(keyword in error_msg for keyword in transient_keywords)


def get_retry_delay(error: Exception, attempt: int, max_delay: float = 60.0) -> float:
    """
    Get retry delay for error with exponential backoff.
    
    Args:
        error: Exception that occurred
        attempt: Attempt number (1-indexed)
        max_delay: Maximum delay in seconds
        
    Returns:
        Delay in seconds
        
    Example:
        >>> for attempt in range(1, 4):
        ...     try:
        ...         api_call()
        ...         break
        ...     except RateLimitError as e:
        ...         delay = get_retry_delay(e, attempt)
        ...         time.sleep(delay)
    """
    if not isinstance(attempt, int) or attempt < 1:
        raise ValueError(f"attempt must be positive integer, got {attempt}")
    
    if isinstance(error, RateLimitError) and error.retry_after:
        # Use provided retry_after if available
        return min(error.retry_after, max_delay)
    
    elif isinstance(error, RateLimitError):
        # Longer delay for rate limits
        base_delay = 2.0
        delay = min(base_delay ** attempt * 2, max_delay)
        
    elif isinstance(error, APIError):
        # Standard exponential backoff for API errors
        base_delay = 2.0
        delay = min(base_delay ** attempt, max_delay)
        
    elif isinstance(error, (CacheReadError, CacheDatabaseError)):
        # Shorter delay for cache errors
        base_delay = 1.5
        delay = min(base_delay ** attempt * 0.5, max_delay * 0.5)
        
    else:
        # Default short delay
        base_delay = 2.0
        delay = min(base_delay ** attempt * 0.5, max_delay * 0.3)
    
    return delay


def categorize_error(error: Exception) -> str:
    """
    Categorize error into high-level category.
    
    Args:
        error: Exception to categorize
        
    Returns:
        Category name ('engine', 'parser', 'formatter', 'cache', 'glossary', 
        'pipeline', 'validation', 'batch', 'unknown')
        
    Example:
        >>> try:
        ...     parse_file()
        ... except Exception as e:
        ...     category = categorize_error(e)
        ...     print(f"Error category: {category}")
    """
    if isinstance(error, EngineError):
        return 'engine'
    elif isinstance(error, ParserError):
        return 'parser'
    elif isinstance(error, FormatterError):
        return 'formatter'
    elif isinstance(error, CacheError):
        return 'cache'
    elif isinstance(error, GlossaryError):
        return 'glossary'
    elif isinstance(error, PipelineError):
        return 'pipeline'
    elif isinstance(error, ValidationError):
        return 'validation'
    elif isinstance(error, BatchProcessingError):
        return 'batch'
    else:
        return 'unknown'


def get_error_severity(error: Exception) -> str:
    """
    Get error severity level.
    
    Args:
        error: Exception to evaluate
        
    Returns:
        Severity level ('critical', 'high', 'medium', 'low')
        
    Example:
        >>> try:
        ...     operation()
        ... except Exception as e:
        ...     severity = get_error_severity(e)
        ...     if severity == 'critical':
        ...         alert_admin()
    """
    # Critical errors
    critical_types = (
        CorruptedFileError,
        CacheDatabaseError,
        ConfigurationError,
        AuthenticationError,
        QuotaExceededError
    )
    
    if isinstance(error, critical_types):
        return 'critical'
    
    # High severity
    high_types = (
        TranslationError,
        OutputError,
        InvalidDocumentError
    )
    
    if isinstance(error, high_types):
        return 'high'
    
    # Medium severity
    medium_types = (
        APIError,
        ParserError,
        FormatterError,
        BatchProcessingError
    )
    
    if isinstance(error, medium_types):
        return 'medium'
    
    # Low severity (retryable)
    low_types = (
        RateLimitError,
        CacheReadError,
        CacheWriteError
    )
    
    if isinstance(error, low_types):
        return 'low'
    
    # Default
    return 'medium'


# ============================================================================
# ERROR RECOVERY STRATEGIES
# ============================================================================

class ErrorRecoveryStrategy:
    """
    Strategy for recovering from errors.
    
    Provides recommendations on how to handle specific errors.
    """
    
    @staticmethod
    def get_recovery_action(error: Exception) -> Dict[str, Any]:
        """
        Get recommended recovery action for error.
        
        Args:
            error: Exception to analyze
            
        Returns:
            Dictionary with recovery information:
            {
                'action': str,  # 'retry', 'skip', 'abort', 'fallback'
                'retry': bool,
                'max_retries': int,
                'delay': float,
                'message': str
            }
        """
        if isinstance(error, RateLimitError):
            return {
                'action': 'retry',
                'retry': True,
                'max_retries': 5,
                'delay': error.retry_after or 60.0,
                'message': 'Rate limit hit, waiting before retry'
            }
        
        elif isinstance(error, APIError):
            return {
                'action': 'retry',
                'retry': True,
                'max_retries': 3,
                'delay': 5.0,
                'message': 'API error, retrying with backoff'
            }
        
        elif isinstance(error, (CacheReadError, CacheWriteError)):
            return {
                'action': 'skip',
                'retry': False,
                'max_retries': 0,
                'delay': 0.0,
                'message': 'Cache error, continuing without cache'
            }
        
        elif isinstance(error, (CorruptedFileError, InvalidDocumentError)):
            return {
                'action': 'abort',
                'retry': False,
                'max_retries': 0,
                'delay': 0.0,
                'message': 'File corrupted or invalid, cannot process'
            }
        
        elif isinstance(error, (AuthenticationError, QuotaExceededError)):
            return {
                'action': 'abort',
                'retry': False,
                'max_retries': 0,
                'delay': 0.0,
                'message': 'Authentication or quota issue, check credentials'
            }
        
        elif isinstance(error, BatchProcessingError):
            return {
                'action': 'fallback',
                'retry': True,
                'max_retries': 1,
                'delay': 1.0,
                'message': 'Batch failed, falling back to individual processing'
            }
        
        else:
            return {
                'action': 'retry',
                'retry': True,
                'max_retries': 2,
                'delay': 2.0,
                'message': 'Unknown error, attempting retry'
            }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing."""
    
    print("=" * 70)
    print("TESTING EXCEPTION SYSTEM")
    print("=" * 70)
    
    # Test 1: Basic exception
    print("\n1. Testing basic exception:")
    try:
        raise APIError("Connection timeout to translation service", status_code=504)
    except APIError as e:
        print(f"   ✓ Caught API error: {e}")
        print(f"   ✓ Status code: {e.status_code}")
        print(f"   ✓ Dict: {e.to_dict()}")
    
    # Test 2: Parser error
    print("\n2. Testing parser error:")
    try:
        raise CorruptedFileError("Document header is malformed", file_path="/path/to/file.docx")
    except ParserError as e:
        print(f"   ✓ Caught parser error: {e}")
        print(f"   ✓ Category: {categorize_error(e)}")
        print(f"   ✓ Severity: {get_error_severity(e)}")
    
    # Test 3: Cache error
    print("\n3. Testing cache error:")
    try:
        raise CacheDatabaseError("Unable to connect to SQLite database", db_path="data/cache.db")
    except CacheError as e:
        print(f"   ✓ Caught cache error: {e}")
        print(f"   ✓ Retryable: {is_retryable_error(e)}")
    
    # Test 4: Error wrapping
    print("\n4. Testing error wrapping:")
    try:
        raise ValueError("Invalid language code")
    except ValueError as e:
        wrapped = wrap_error(e, InvalidLanguageError, "Language validation failed")
        print(f"   ✓ Wrapped error: {wrapped}")
        print(f"   ✓ Original cause: {wrapped.__cause__}")
    
    # Test 5: Error message formatting
    print("\n5. Testing error formatting:")
    try:
        raise TranslationError("Model returned empty response")
    except TranslationError as e:
        formatted = format_error_message(e, context="translate_document")
        print(f"   ✓ Formatted: {formatted}")
    
    # Test 6: Context manager
    print("\n6. Testing error context:")
    try:
        with error_context("test operation", ParserError):
            raise ValueError("Something went wrong")
    except ParserError as e:
        print(f"   ✓ Caught wrapped error: {e}")
    
    # Test 7: Retry logic
    print("\n7. Testing retry logic:")
    try:
        raise RateLimitError("Too many requests", retry_after=30)
    except RateLimitError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Retryable: {is_retryable_error(e)}")
        delay = get_retry_delay(e, attempt=1)
        print(f"   ✓ Retry delay: {delay}s")
    
    # Test 8: Error recovery
    print("\n8. Testing error recovery strategy:")
    errors = [
        RateLimitError("Rate limit"),
        APIError("API error"),
        CorruptedFileError("Corrupted file"),
        CacheReadError("Cache error")
    ]
    
    for error in errors:
        recovery = ErrorRecoveryStrategy.get_recovery_action(error)
        print(f"   ✓ {error.__class__.__name__}: action={recovery['action']}, "
              f"retry={recovery['retry']}, delay={recovery['delay']}s")
    
    # Test 9: ValidationError with field
    print("\n9. Testing ValidationError:")
    try:
        raise ValidationError("Invalid value for field", field="source_lang")
    except ValidationError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Field: {e.field}")
    
    # Test 10: BatchProcessingError
    print("\n10. Testing BatchProcessingError:")
    try:
        raise BatchProcessingError(
            "Batch translation failed", 
            batch_index=5, 
            total_batches=10
        )
    except BatchProcessingError as e:
        print(f"   ✓ Error: {e}")
        print(f"   ✓ Batch: {e.batch_index}/{e.total_batches}")
    
    print("\n" + "=" * 70)
    print("✓ ALL EXCEPTION TESTS PASSED!")
    print("=" * 70)
