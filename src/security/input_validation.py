"""
Input validation and sanitization to prevent injection attacks.
"""
import re
import html
import unicodedata
from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class ValidationRules:
    """Validation rules configuration."""
    max_length: int = 10000
    min_length: int = 0
    allow_unicode: bool = True
    allow_html: bool = False
    max_sql_length: int = 1000
    forbidden_patterns: list = None
    
    def __post_init__(self):
        if self.forbidden_patterns is None:
            # Dangerous SQL/script patterns
            self.forbidden_patterns = [
                r';\s*DROP\s+TABLE',
                r';\s*DELETE\s+FROM',
                r';\s*UPDATE\s+',
                r'<script[^>]*>',
                r'javascript:',
                r'on\w+\s*=',  # onclick, onerror, etc.
                r'UNION\s+SELECT',
                r'exec\s*\(',
                r'eval\s*\(',
            ]


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self, rules: Optional[ValidationRules] = None):
        self.rules = rules or ValidationRules()
    
    def validate_text(self, text: str, context: str = "text") -> str:
        """
        Validate and sanitize text input.
        
        Args:
            text: Input text
            context: Context for error messages
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(text, str):
            raise ValueError(f"{context}: Expected string, got {type(text)}")
        
        # Length validation
        if len(text) > self.rules.max_length:
            raise ValueError(
                f"{context}: Text too long ({len(text)} > {self.rules.max_length} chars)"
            )
        
        if len(text) < self.rules.min_length:
            raise ValueError(
                f"{context}: Text too short ({len(text)} < {self.rules.min_length} chars)"
            )
        
        # Check for malicious patterns
        self._check_dangerous_patterns(text, context)
        
        # Normalize unicode
        if self.rules.allow_unicode:
            text = self._normalize_unicode(text)
        else:
            if not text.isascii():
                raise ValueError(f"{context}: Non-ASCII characters not allowed")
        
        # HTML sanitization
        if not self.rules.allow_html:
            text = self._sanitize_html(text)
        
        return text
    
    def validate_sql_string(self, value: str, context: str = "sql_string") -> str:
        """
        Validate string for SQL usage.
        Prevents SQL injection even though we use parameterized queries.
        
        Args:
            value: SQL string value
            context: Context for error messages
            
        Returns:
            Validated string
            
        Raises:
            ValueError: If validation fails
        """
        if not isinstance(value, str):
            raise ValueError(f"{context}: Expected string, got {type(value)}")
        
        # Length check
        if len(value) > self.rules.max_sql_length:
            raise ValueError(
                f"{context}: String too long for SQL "
                f"({len(value)} > {self.rules.max_sql_length})"
            )
        
        # Check for SQL keywords and dangerous characters
        dangerous_sql = [
            '--',  # SQL comments
            '/*',  # Multi-line comments
            '*/',
            ';',   # Statement separator
            '\\x', # Hex escape
            '\\0', # Null byte
        ]
        
        for pattern in dangerous_sql:
            if pattern in value:
                raise ValueError(
                    f"{context}: Potentially dangerous SQL pattern: {pattern}"
                )
        
        # Check for suspicious SQL keywords
        suspicious_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'EXEC',
            'UNION', 'INSERT', 'UPDATE', 'CREATE', 'GRANT'
        ]
        
        value_upper = value.upper()
        for keyword in suspicious_keywords:
            if keyword in value_upper:
                # Allow if it's part of a longer word
                pattern = r'\b' + keyword + r'\b'
                if re.search(pattern, value_upper):
                    raise ValueError(
                        f"{context}: SQL keyword not allowed: {keyword}"
                    )
        
        return value
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate filename to prevent path traversal.
        
        Args:
            filename: Filename to validate
            
        Returns:
            Safe filename
            
        Raises:
            ValueError: If filename is dangerous
        """
        if not isinstance(filename, str):
            raise ValueError(f"Filename must be string, got {type(filename)}")
        
        # Check for path traversal
        if '..' in filename:
            raise ValueError("Path traversal not allowed (..)") 
        
        if filename.startswith('/') or filename.startswith('\\'):
            raise ValueError("Absolute paths not allowed")
        
        if ':' in filename and len(filename) > 1 and filename[1] == ':':
            raise ValueError("Drive letters not allowed")
        
        # Check for dangerous characters
        dangerous_chars = ['<', '>', '|', '\0', '\n', '\r']
        for char in dangerous_chars:
            if char in filename:
                raise ValueError(f"Dangerous character not allowed: {repr(char)}")
        
        # Length check
        if len(filename) > 255:
            raise ValueError(f"Filename too long: {len(filename)} > 255")
        
        if len(filename) == 0:
            raise ValueError("Filename cannot be empty")
        
        return filename
    
    def validate_language_code(self, code: str) -> str:
        """
        Validate language code.
        
        Args:
            code: Language code (e.g., 'en', 'ru')
            
        Returns:
            Validated code
            
        Raises:
            ValueError: If code is invalid
        """
        if not isinstance(code, str):
            raise ValueError(f"Language code must be string, got {type(code)}")
        
        # ISO 639-1 format: 2-3 lowercase letters
        if not re.match(r'^[a-z]{2,3}$', code):
            raise ValueError(
                f"Invalid language code format: {code}. "
                f"Expected 2-3 lowercase letters (ISO 639-1)"
            )
        
        return code
    
    def validate_domain(self, domain: str) -> str:
        """
        Validate domain name.
        
        Args:
            domain: Domain name (e.g., 'legal', 'medical')
            
        Returns:
            Validated domain
            
        Raises:
            ValueError: If domain is invalid
        """
        if not isinstance(domain, str):
            raise ValueError(f"Domain must be string, got {type(domain)}")
        
        # Alphanumeric, hyphen, underscore only
        if not re.match(r'^[a-zA-Z0-9_-]+$', domain):
            raise ValueError(
                f"Invalid domain format: {domain}. "
                f"Only alphanumeric, hyphen, underscore allowed"
            )
        
        if len(domain) > 100:
            raise ValueError(f"Domain too long: {len(domain)} > 100")
        
        return domain
    
    def sanitize_for_log(self, text: str, max_length: int = 200) -> str:
        """
        Sanitize text for safe logging.
        Prevents log injection and truncates long text.
        
        Args:
            text: Text to log
            max_length: Maximum length
            
        Returns:
            Safe log text
        """
        # Remove newlines to prevent log injection
        text = text.replace('\n', ' ').replace('\r', '')
        
        # Truncate
        if len(text) > max_length:
            text = text[:max_length] + '...'
        
        # Remove control characters
        text = ''.join(
            char for char in text 
            if unicodedata.category(char)[0] != 'C' or char in ('\t', ' ')
        )
        
        return text
    
    def _check_dangerous_patterns(self, text: str, context: str):
        """Check for dangerous patterns."""
        for pattern in self.rules.forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(
                    f"{context}: Forbidden pattern detected: {pattern}"
                )
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode to NFC form."""
        return unicodedata.normalize('NFC', text)
    
    def _sanitize_html(self, text: str) -> str:
        """Remove/escape HTML tags."""
        # Escape HTML entities
        text = html.escape(text)
        
        # Remove any remaining tags (shouldn't happen after escape)
        text = re.sub(r'<[^>]+>', '', text)
        
        return text


# Specialized validators for different contexts
class StrictValidator(InputValidator):
    """Extra strict validation for critical inputs."""
    
    def __init__(self):
        rules = ValidationRules(
            max_length=5000,
            allow_unicode=True,
            allow_html=False,
            max_sql_length=500
        )
        super().__init__(rules)


class LenientValidator(InputValidator):
    """Lenient validation for user content."""
    
    def __init__(self):
        rules = ValidationRules(
            max_length=50000,
            allow_unicode=True,
            allow_html=False,  # Still no HTML
            max_sql_length=5000
        )
        super().__init__(rules)


# Global instances
_strict_validator: Optional[StrictValidator] = None
_lenient_validator: Optional[LenientValidator] = None


def get_strict_validator() -> StrictValidator:
    """Get strict validator instance."""
    global _strict_validator
    if _strict_validator is None:
        _strict_validator = StrictValidator()
    return _strict_validator


def get_lenient_validator() -> LenientValidator:
    """Get lenient validator instance."""
    global _lenient_validator
    if _lenient_validator is None:
        _lenient_validator = LenientValidator()
    return _lenient_validator


# Convenience functions
def validate_translation_text(text: str) -> str:
    """Validate text for translation (lenient)."""
    return get_lenient_validator().validate_text(text, "translation_text")


def validate_glossary_term(term: str) -> str:
    """Validate glossary term (strict)."""
    return get_strict_validator().validate_text(term, "glossary_term")


def validate_cache_key(key: str) -> str:
    """Validate cache key (strict)."""
    validator = get_strict_validator()
    key = validator.validate_text(key, "cache_key")
    return validator.validate_sql_string(key, "cache_key")


# Example usage and tests
if __name__ == "__main__":
    validator = InputValidator()
    
    # Test valid inputs
    try:
        print("✓ Valid text:", validator.validate_text("Hello, world!"))
        print("✓ Valid filename:", validator.validate_filename("document.docx"))
        print("✓ Valid language:", validator.validate_language_code("en"))
        print("✓ Valid domain:", validator.validate_domain("legal"))
    except ValueError as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test malicious inputs
    malicious_tests = [
        ("SQL injection", "; DROP TABLE users--", "validate_sql_string"),
        ("XSS", "<script>alert('xss')</script>", "validate_text"),
        ("Path traversal", "../../../etc/passwd", "validate_filename"),
        ("Long input", "A" * 20000, "validate_text"),
        ("Invalid language", "eng123", "validate_language_code"),
    ]
    
    print("\nMalicious input tests:")
    for name, input_val, method_name in malicious_tests:
        try:
            method = getattr(validator, method_name)
            method(input_val)
            print(f"✗ {name}: FAILED TO BLOCK")
        except ValueError as e:
            print(f"✓ {name}: Blocked - {str(e)[:50]}")
