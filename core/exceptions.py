"""
Custom exceptions for the translation system.
Provides clear error hierarchy and meaningful error messages.
"""


# ===== Base Exceptions =====

class TranslationSystemError(Exception):
    """Base exception for all translation system errors."""
    pass


# ===== Engine Exceptions =====

class EngineError(TranslationSystemError):
    """Base exception for translation engine errors."""
    pass


class TranslationError(EngineError):
    """Error during translation operation."""
    pass


class APIError(EngineError):
    """Error communicating with translation API."""
    pass


class RateLimitError(EngineError):
    """Rate limit exceeded."""
    pass


class QuotaExceededError(EngineError):
    """API quota exceeded."""
    pass


class InvalidLanguageError(EngineError):
    """Invalid or unsupported language."""
    pass


class AuthenticationError(EngineError):
    """Authentication failed (invalid API key)."""
    pass


# ===== Parser Exceptions =====

class ParserError(TranslationSystemError):
    """Base exception for document parsing errors."""
    pass


class UnsupportedFileTypeError(ParserError):
    """File type not supported."""
    pass


class CorruptedFileError(ParserError):
    """File is corrupted or cannot be read."""
    pass


class InvalidDocumentError(ParserError):
    """Document structure is invalid."""
    pass


# ===== Formatter Exceptions =====

class FormatterError(TranslationSystemError):
    """Base exception for document formatting errors."""
    pass


class FormattingError(FormatterError):
    """Error during document formatting."""
    pass


class OutputError(FormatterError):
    """Error writing output file."""
    pass


# ===== Cache Exceptions =====

class CacheError(TranslationSystemError):
    """Base exception for cache errors."""
    pass


class CacheReadError(CacheError):
    """Error reading from cache."""
    pass


class CacheWriteError(CacheError):
    """Error writing to cache."""
    pass


class CacheDatabaseError(CacheError):
    """Database error in cache."""
    pass


# ===== Glossary Exceptions =====

class GlossaryError(TranslationSystemError):
    """Base exception for glossary errors."""
    pass


class GlossaryReadError(GlossaryError):
    """Error reading glossary."""
    pass


class GlossaryWriteError(GlossaryError):
    """Error writing to glossary."""
    pass


class InvalidTermError(GlossaryError):
    """Invalid glossary term."""
    pass


# ===== Pipeline Exceptions =====

class PipelineError(TranslationSystemError):
    """Base exception for pipeline errors."""
    pass


class TranslationPipelineError(PipelineError):
    """Error in translation pipeline."""
    pass


class ConfigurationError(PipelineError):
    """Invalid configuration."""
    pass


# ===== Validation Exceptions =====

class ValidationError(TranslationSystemError):
    """Base exception for validation errors."""
    pass


class InvalidInputError(ValidationError):
    """Invalid input provided."""
    pass


class InvalidConfigError(ValidationError):
    """Invalid configuration."""
    pass


# ===== Batch Processing Exceptions =====

class BatchProcessingError(TranslationSystemError):
    """Error during batch processing."""
    pass


class TaskFailedError(BatchProcessingError):
    """Individual batch task failed."""
    pass


# ===== Helper Functions =====

def format_error_message(error: Exception, context: str = None) -> str:
    """
    Format error message with context.
    
    Args:
        error: Exception
        context: Optional context information
        
    Returns:
        Formatted error message
    """
    error_type = type(error).__name__
    message = str(error)
    
    if context:
        return f"{error_type} in {context}: {message}"
    else:
        return f"{error_type}: {message}"


def wrap_error(error: Exception, error_class: type, message: str = None) -> Exception:
    """
    Wrap an exception in a custom exception type.
    
    Args:
        error: Original exception
        error_class: Exception class to wrap with
        message: Optional custom message
        
    Returns:
        Wrapped exception
    """
    if message:
        return error_class(f"{message}: {error}") from error
    else:
        return error_class(str(error)) from error


# ===== Example Usage =====

if __name__ == "__main__":
    # Example error handling
    
    try:
        # Simulate API error
        raise APIError("Connection timeout to translation service")
    except APIError as e:
        print(f"Caught API error: {e}")
    
    try:
        # Simulate parsing error
        raise CorruptedFileError("Document header is malformed")
    except ParserError as e:
        print(f"Caught parser error: {e}")
    
    try:
        # Simulate cache error
        raise CacheDatabaseError("Unable to connect to SQLite database")
    except CacheError as e:
        print(f"Caught cache error: {e}")
    
    # Error formatting
    try:
        raise ValueError("Invalid language code")
    except ValueError as e:
        wrapped = wrap_error(e, InvalidLanguageError, "Language validation failed")
        print(f"Wrapped error: {wrapped}")
        
    # Error message formatting
    try:
        raise TranslationError("Model returned empty response")
    except TranslationError as e:
        formatted = format_error_message(e, context="translate_document")
        print(f"Formatted: {formatted}")
