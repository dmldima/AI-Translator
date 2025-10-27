"""
File and input validators.
"""
from pathlib import Path
from typing import Optional, List
import logging

from ..core.models import FileType, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Validation error."""
    pass


def validate_file_exists(file_path: Path) -> None:
    """
    Validate file exists.
    
    Args:
        file_path: Path to file
        
    Raises:
        ValidationError: If file doesn't exist
    """
    if not file_path.exists():
        raise ValidationError(f"File not found: {file_path}")
    
    if not file_path.is_file():
        raise ValidationError(f"Not a file: {file_path}")


def validate_file_size(file_path: Path, max_size_mb: int = 100) -> None:
    """
    Validate file size.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
        
    Raises:
        ValidationError: If file too large
    """
    size_mb = file_path.stat().st_size / (1024 * 1024)
    
    if size_mb > max_size_mb:
        raise ValidationError(
            f"File too large: {size_mb:.1f} MB (max: {max_size_mb} MB)"
        )


def validate_file_type(file_path: Path, allowed_types: Optional[List[FileType]] = None) -> FileType:
    """
    Validate file type.
    
    Args:
        file_path: Path to file
        allowed_types: List of allowed file types (None = all)
        
    Returns:
        FileType
        
    Raises:
        ValidationError: If file type not supported
    """
    try:
        file_type = FileType.from_extension(file_path.suffix)
    except ValueError:
        raise ValidationError(f"Unsupported file type: {file_path.suffix}")
    
    if allowed_types and file_type not in allowed_types:
        allowed = ", ".join(ft.value for ft in allowed_types)
        raise ValidationError(
            f"File type {file_type.value} not allowed. Allowed: {allowed}"
        )
    
    return file_type


def validate_language_code(lang_code: str) -> None:
    """
    Validate language code.
    
    Args:
        lang_code: Language code
        
    Raises:
        ValidationError: If language not supported
    """
    if lang_code not in SUPPORTED_LANGUAGES:
        available = ", ".join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValidationError(
            f"Unsupported language: {lang_code}. "
            f"Available: {available}"
        )


def validate_language_pair(source_lang: str, target_lang: str) -> None:
    """
    Validate language pair.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        
    Raises:
        ValidationError: If language pair invalid
    """
    validate_language_code(source_lang)
    validate_language_code(target_lang)
    
    if source_lang == target_lang:
        raise ValidationError("Source and target languages must be different")


def validate_output_path(output_path: Path, overwrite: bool = False) -> None:
    """
    Validate output path.
    
    Args:
        output_path: Path to output file
        overwrite: Allow overwriting existing files
        
    Raises:
        ValidationError: If output path invalid
    """
    # Check parent directory exists
    if not output_path.parent.exists():
        raise ValidationError(f"Output directory doesn't exist: {output_path.parent}")
    
    # Check if file exists
    if output_path.exists() and not overwrite:
        raise ValidationError(
            f"Output file already exists: {output_path}. "
            f"Use overwrite=True to replace."
        )


def validate_api_key(api_key: Optional[str], engine: str) -> None:
    """
    Validate API key.
    
    Args:
        api_key: API key
        engine: Engine name
        
    Raises:
        ValidationError: If API key invalid
    """
    if not api_key or not api_key.strip():
        raise ValidationError(
            f"API key required for {engine}. "
            f"Set {engine.upper()}_API_KEY environment variable."
        )
    
    # Basic format validation
    if engine == "openai":
        if not api_key.startswith("sk-"):
            logger.warning("OpenAI API key should start with 'sk-'")
    
    if len(api_key) < 20:
        logger.warning(f"API key seems too short: {len(api_key)} characters")


def validate_document(
    file_path: Path,
    max_size_mb: int = 100,
    allowed_types: Optional[List[FileType]] = None
) -> FileType:
    """
    Complete document validation.
    
    Args:
        file_path: Path to document
        max_size_mb: Maximum file size in MB
        allowed_types: List of allowed file types
        
    Returns:
        FileType
        
    Raises:
        ValidationError: If validation fails
    """
    validate_file_exists(file_path)
    validate_file_size(file_path, max_size_mb)
    file_type = validate_file_type(file_path, allowed_types)
    
    logger.info(f"Document validated: {file_path.name} ({file_type.value})")
    
    return file_type


if __name__ == "__main__":
    # Test validators
    test_file = Path("test.docx")
    
    try:
        validate_language_pair("en", "ru")
        print("✓ Language pair valid")
    except ValidationError as e:
        print(f"✗ {e}")
    
    try:
        validate_language_pair("en", "en")
    except ValidationError as e:
        print(f"✓ Caught same language: {e}")
