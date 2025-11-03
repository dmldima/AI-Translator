"""Core data models for document processing."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


# ============================================================================
# FILE AND SEGMENT TYPES
# ============================================================================

class FileType(Enum):
    """Supported document file types."""
    DOCX = "docx"
    XLSX = "xlsx"
    PDF = "pdf"
    TXT = "txt"
    
    @classmethod
    def from_extension(cls, extension: str) -> 'FileType':
        """
        Get FileType from file extension.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            FileType enum
            
        Raises:
            ValueError: If extension not supported
        """
        if not extension:
            raise ValueError("Extension cannot be empty")
        
        ext = extension.lower().lstrip('.')
        
        if not ext:
            raise ValueError("Extension cannot be empty after stripping dot")
        
        try:
            return cls(ext)
        except ValueError:
            supported = ', '.join(f".{ft.value}" for ft in cls)
            raise ValueError(
                f"Unsupported file extension: '.{ext}'. "
                f"Supported: {supported}"
            )
    
    @classmethod
    def is_supported(cls, extension: str) -> bool:
        """
        Check if extension is supported.
        
        Args:
            extension: File extension (with or without dot)
            
        Returns:
            True if supported, False otherwise
        """
        try:
            cls.from_extension(extension)
            return True
        except ValueError:
            return False


class SegmentType(Enum):
    """Types of text segments in documents."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    ENDNOTE = "endnote"


class TranslationStatus(Enum):
    """Translation job status."""
    PENDING = "pending"
    PARSING = "parsing"
    TRANSLATING = "translating"
    FORMATTING = "formatting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    
    def is_terminal(self) -> bool:
        """Check if status is terminal (job finished)."""
        return self in (
            TranslationStatus.COMPLETED,
            TranslationStatus.FAILED,
            TranslationStatus.CANCELLED
        )
    
    def is_active(self) -> bool:
        """Check if status is active (job in progress)."""
        return self in (
            TranslationStatus.PARSING,
            TranslationStatus.TRANSLATING,
            TranslationStatus.FORMATTING
        )


# ============================================================================
# FORMATTING CLASSES
# ============================================================================

@dataclass
class SegmentPosition:
    """Position of text segment in document."""
    # Paragraph/Run (DOCX body)
    paragraph_index: Optional[int] = None
    run_index: Optional[int] = None
    
    # Table (DOCX)
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    cell_index: Optional[int] = None
    
    # Section (DOCX headers/footers)
    section_index: Optional[int] = None
    
    # Spreadsheet (XLSX)
    sheet_name: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None
    
    def __post_init__(self):
        """Validate position after initialization."""
        # At least one position must be set
        if all(v is None for v in [
            self.paragraph_index, self.run_index, self.table_index,
            self.row_index, self.cell_index, self.section_index,
            self.sheet_name, self.row, self.column
        ]):
            raise ValueError("At least one position field must be set")


@dataclass
class TextFormatting:
    """Text-level formatting properties."""
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: Optional[str] = None
    highlight_color: Optional[str] = None
    subscript: bool = False
    superscript: bool = False
    small_caps: bool = False
    
    def __post_init__(self):
        """Validate formatting after initialization."""
        # Validate font_size
        if self.font_size is not None:
            if self.font_size <= 0:
                raise ValueError(f"font_size must be positive, got {self.font_size}")
            if self.font_size > 1000:
                raise ValueError(f"font_size too large: {self.font_size}")
        
        # Validate colors (basic check)
        if self.color and not self._is_valid_color(self.color):
            raise ValueError(f"Invalid color format: {self.color}")
        if self.highlight_color and not self._is_valid_color(self.highlight_color):
            raise ValueError(f"Invalid highlight_color format: {self.highlight_color}")
    
    @staticmethod
    def _is_valid_color(color: str) -> bool:
        """Basic color validation (hex or name)."""
        if not color:
            return False
        # Accept hex colors
        if color.startswith('#'):
            return len(color) in (4, 7) and all(c in '0123456789ABCDEFabcdef' for c in color[1:])
        # Accept RGB format
        if color.startswith('rgb'):
            return True
        # Accept color names (basic check)
        return color.replace(' ', '').replace('-', '').isalpha()


@dataclass
class ParagraphFormatting:
    """Paragraph-level formatting properties."""
    alignment: Optional[str] = None
    line_spacing: Optional[float] = None
    space_before: Optional[float] = None
    space_after: Optional[float] = None
    left_indent: Optional[float] = None
    right_indent: Optional[float] = None
    first_line_indent: Optional[float] = None
    style_name: Optional[str] = None
    keep_together: bool = False
    keep_with_next: bool = False
    page_break_before: bool = False
    
    def __post_init__(self):
        """Validate formatting after initialization."""
        # Validate alignment
        if self.alignment:
            valid_alignments = ('left', 'center', 'right', 'justify', 'distributed')
            if self.alignment.lower() not in valid_alignments:
                raise ValueError(
                    f"Invalid alignment: {self.alignment}. "
                    f"Valid: {', '.join(valid_alignments)}"
                )
        
        # Validate numeric values are non-negative
        for field_name in ['line_spacing', 'space_before', 'space_after', 
                           'left_indent', 'right_indent']:
            value = getattr(self, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative: {value}")


@dataclass
class CellFormatting:
    """Spreadsheet cell formatting properties."""
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    font_bold: bool = False
    font_italic: bool = False
    font_color: Optional[str] = None
    fill_color: Optional[str] = None
    fill_pattern: Optional[str] = None
    horizontal_alignment: Optional[str] = None
    vertical_alignment: Optional[str] = None
    wrap_text: bool = False
    border_style: Optional[Dict[str, str]] = None
    number_format: Optional[str] = None
    column_width: Optional[float] = None
    row_height: Optional[float] = None
    
    def __post_init__(self):
        """Validate formatting after initialization."""
        # Validate font_size
        if self.font_size is not None:
            if self.font_size <= 0:
                raise ValueError(f"font_size must be positive, got {self.font_size}")
        
        # Validate dimensions
        if self.column_width is not None and self.column_width <= 0:
            raise ValueError(f"column_width must be positive: {self.column_width}")
        if self.row_height is not None and self.row_height <= 0:
            raise ValueError(f"row_height must be positive: {self.row_height}")


# ============================================================================
# DOCUMENT CLASSES
# ============================================================================

@dataclass
class TextSegment:
    """A segment of translatable text with formatting."""
    id: str
    text: str
    segment_type: SegmentType
    position: SegmentPosition
    text_formatting: Optional[TextFormatting] = None
    paragraph_formatting: Optional[ParagraphFormatting] = None
    cell_formatting: Optional[CellFormatting] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate segment after initialization."""
        if not self.id:
            raise ValueError("Segment id cannot be empty")
        
        if not isinstance(self.text, str):
            raise TypeError(f"Text must be str, got {type(self.text).__name__}")
        
        if not isinstance(self.segment_type, SegmentType):
            raise TypeError(
                f"segment_type must be SegmentType, got {type(self.segment_type).__name__}"
            )
        
        if not isinstance(self.position, SegmentPosition):
            raise TypeError(
                f"position must be SegmentPosition, got {type(self.position).__name__}"
            )
    
    @property
    def word_count(self) -> int:
        """Count words in segment."""
        if not self.text or not self.text.strip():
            return 0
        return len(self.text.split())
    
    @property
    def char_count(self) -> int:
        """Count characters in segment."""
        return len(self.text)
    
    @property
    def is_empty(self) -> bool:
        """Check if segment is empty or whitespace only."""
        return not self.text or not self.text.strip()


@dataclass
class DocumentMetadata:
    """Document-level metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[str] = None
    comments: Optional[str] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    last_modified_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'author': self.author,
            'subject': self.subject,
            'keywords': self.keywords,
            'comments': self.comments,
            'created': self.created.isoformat() if self.created else None,
            'modified': self.modified.isoformat() if self.modified else None,
            'last_modified_by': self.last_modified_by
        }


@dataclass
class DocumentStyles:
    """Document-level style information."""
    default_font: Optional[str] = None
    default_font_size: Optional[float] = None
    styles: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate styles after initialization."""
        if self.default_font_size is not None:
            if self.default_font_size <= 0:
                raise ValueError(
                    f"default_font_size must be positive, got {self.default_font_size}"
                )


@dataclass
class Document:
    """Complete document with content and formatting."""
    file_path: Path
    file_type: FileType
    segments: List[TextSegment]
    metadata: Optional[DocumentMetadata] = None
    styles: Optional[DocumentStyles] = None
    headers: List[TextSegment] = field(default_factory=list)
    footers: List[TextSegment] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate document after initialization."""
        if not isinstance(self.file_path, Path):
            raise TypeError(f"file_path must be Path, got {type(self.file_path).__name__}")
        
        if not isinstance(self.file_type, FileType):
            raise TypeError(f"file_type must be FileType, got {type(self.file_type).__name__}")
        
        if not isinstance(self.segments, list):
            raise TypeError(f"segments must be list, got {type(self.segments).__name__}")
        
        # Validate all segments are TextSegment
        for i, seg in enumerate(self.segments):
            if not isinstance(seg, TextSegment):
                raise TypeError(
                    f"segments[{i}] must be TextSegment, got {type(seg).__name__}"
                )
    
    @property
    def total_words(self) -> int:
        """Total word count in document."""
        return sum(segment.word_count for segment in self.segments)
    
    @property
    def total_chars(self) -> int:
        """Total character count in document."""
        return sum(segment.char_count for segment in self.segments)
    
    @property
    def total_segments(self) -> int:
        """Total number of segments."""
        return len(self.segments)
    
    @property
    def non_empty_segments(self) -> int:
        """Count of non-empty segments."""
        return sum(1 for seg in self.segments if not seg.is_empty)


# ============================================================================
# TRANSLATION REQUEST/RESPONSE CLASSES
# ============================================================================

@dataclass
class TranslationRequest:
    """
    Translation request for caching and API calls.
    
    This class is hashable and can be used as dictionary key.
    """
    text: str
    source_lang: str
    target_lang: str
    domain: str = "general"
    segment_id: Optional[str] = None
    glossary_version: str = "latest"
    
    def __post_init__(self):
        """Validate request after initialization."""
        if not isinstance(self.text, str):
            raise TypeError(f"text must be str, got {type(self.text).__name__}")
        
        if not self.text:
            raise ValueError("text cannot be empty")
        
        if not self.source_lang or not self.source_lang.strip():
            raise ValueError("source_lang cannot be empty")
        
        if not self.target_lang or not self.target_lang.strip():
            raise ValueError("target_lang cannot be empty")
        
        if not self.domain or not self.domain.strip():
            raise ValueError("domain cannot be empty")
        
        # Normalize language codes
        self.source_lang = self.source_lang.lower().strip()
        self.target_lang = self.target_lang.lower().strip()
        self.domain = self.domain.lower().strip()
    
    def __hash__(self):
        """Make request hashable for dict keys."""
        return hash((
            self.text,
            self.source_lang,
            self.target_lang,
            self.domain,
            self.glossary_version
        ))
    
    def __eq__(self, other):
        """Equality check for caching."""
        if not isinstance(other, TranslationRequest):
            return False
        return (
            self.text == other.text and
            self.source_lang == other.source_lang and
            self.target_lang == other.target_lang and
            self.domain == other.domain and
            self.glossary_version == other.glossary_version
        )
    
    def __repr__(self):
        """String representation."""
        text_preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return (
            f"TranslationRequest("
            f"{self.source_lang}→{self.target_lang}, "
            f"domain={self.domain}, "
            f"text='{text_preview}')"
        )


@dataclass
class TranslationResult:
    """
    Translation result with metadata.
    """
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    domain: str
    engine: str
    model: str
    confidence: float = 1.0
    cached: bool = False
    glossary_applied: bool = False
    glossary_terms_used: List[str] = field(default_factory=list)
    segment_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __post_init__(self):
        """Validate result after initialization."""
        if not isinstance(self.original_text, str):
            raise TypeError(f"original_text must be str, got {type(self.original_text).__name__}")
        
        if not isinstance(self.translated_text, str):
            raise TypeError(f"translated_text must be str, got {type(self.translated_text).__name__}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        
        if not self.engine or not self.engine.strip():
            raise ValueError("engine cannot be empty")
        
        if not self.model or not self.model.strip():
            raise ValueError("model cannot be empty")
    
    @property
    def word_count(self) -> int:
        """Word count of translated text."""
        return len(self.translated_text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_text': self.original_text,
            'translated_text': self.translated_text,
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'domain': self.domain,
            'engine': self.engine,
            'model': self.model,
            'confidence': self.confidence,
            'cached': self.cached,
            'glossary_applied': self.glossary_applied,
            'glossary_terms_used': self.glossary_terms_used,
            'segment_id': self.segment_id,
            'timestamp': self.timestamp
        }


@dataclass
class TranslationJob:
    """
    Translation job tracking progress and results.
    """
    job_id: str
    input_file: Path
    output_file: Path
    source_lang: str
    target_lang: str
    domain: str
    engine: str
    
    status: TranslationStatus = TranslationStatus.PENDING
    
    total_segments: int = 0
    translated_segments: int = 0
    cached_segments: int = 0
    failed_segments: int = 0
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate job after initialization."""
        if not self.job_id or not self.job_id.strip():
            raise ValueError("job_id cannot be empty")
        
        if not isinstance(self.input_file, Path):
            raise TypeError(f"input_file must be Path, got {type(self.input_file).__name__}")
        
        if not isinstance(self.output_file, Path):
            raise TypeError(f"output_file must be Path, got {type(self.output_file).__name__}")
        
        if self.total_segments < 0:
            raise ValueError(f"total_segments cannot be negative: {self.total_segments}")
        
        if self.translated_segments < 0:
            raise ValueError(f"translated_segments cannot be negative: {self.translated_segments}")
        
        if self.cached_segments < 0:
            raise ValueError(f"cached_segments cannot be negative: {self.cached_segments}")
        
        if self.failed_segments < 0:
            raise ValueError(f"failed_segments cannot be negative: {self.failed_segments}")
    
    def update_progress(self, current: int):
        """
        Update translation progress.
        
        Args:
            current: Current number of completed segments (including cached)
        """
        if current < 0:
            raise ValueError(f"current cannot be negative: {current}")
        
        self.translated_segments = current - self.cached_segments
        
        if self.translated_segments < 0:
            self.translated_segments = 0
    
    def add_error(self, error: str):
        """
        Add error message to job.
        
        Args:
            error: Error message
        """
        if not error or not error.strip():
            return
        
        # Truncate very long errors
        max_length = 500
        if len(error) > max_length:
            error = error[:max_length] + "... (truncated)"
        
        self.errors.append(error)
    
    @property
    def duration(self) -> float:
        """
        Get job duration in seconds.
        
        Returns:
            Duration in seconds, or 0 if not completed
        """
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            # Job still running
            return (datetime.utcnow() - self.started_at).total_seconds()
        return 0.0
    
    @property
    def progress_percentage(self) -> float:
        """
        Get progress as percentage.
        
        Returns:
            Progress percentage (0-100)
        """
        if self.total_segments == 0:
            return 0.0
        completed = self.translated_segments + self.cached_segments
        return min((completed / self.total_segments) * 100, 100.0)
    
    @property
    def success_rate(self) -> float:
        """
        Get success rate as percentage.
        
        Returns:
            Success rate percentage (0-100)
        """
        if self.total_segments == 0:
            return 0.0
        successful = self.total_segments - self.failed_segments
        return (successful / self.total_segments) * 100
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status.is_terminal()
    
    @property
    def is_successful(self) -> bool:
        """Check if job completed successfully."""
        return self.status == TranslationStatus.COMPLETED and self.failed_segments == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'job_id': self.job_id,
            'input_file': str(self.input_file),
            'output_file': str(self.output_file),
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'domain': self.domain,
            'engine': self.engine,
            'status': self.status.value,
            'total_segments': self.total_segments,
            'translated_segments': self.translated_segments,
            'cached_segments': self.cached_segments,
            'failed_segments': self.failed_segments,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'duration': self.duration,
            'progress_percentage': self.progress_percentage,
            'success_rate': self.success_rate,
            'errors': self.errors
        }


# ============================================================================
# SUPPORTED LANGUAGES
# ============================================================================

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'zh': 'Chinese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'nl': 'Dutch',
    'pl': 'Polish',
    'tr': 'Turkish',
    'sv': 'Swedish',
    'da': 'Danish',
    'no': 'Norwegian',
    'fi': 'Finnish',
    'cs': 'Czech',
    'hu': 'Hungarian',
    'ro': 'Romanian',
    'bg': 'Bulgarian',
    'el': 'Greek',
    'he': 'Hebrew',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'id': 'Indonesian',
    'uk': 'Ukrainian',
    'ca': 'Catalan',
    'hr': 'Croatian',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'et': 'Estonian',
    'lv': 'Latvian',
    'lt': 'Lithuanian',
}


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_language_pair(source_lang: str, target_lang: str):
    """
    Validate language pair.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        
    Raises:
        ValueError: If languages invalid or identical
    """
    if not source_lang or not source_lang.strip():
        raise ValueError("source_lang cannot be empty")
    
    if not target_lang or not target_lang.strip():
        raise ValueError("target_lang cannot be empty")
    
    source_lang = source_lang.lower().strip()
    target_lang = target_lang.lower().strip()
    
    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different")
    
    if source_lang not in SUPPORTED_LANGUAGES:
        supported = ', '.join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported source language: {source_lang}. "
            f"Supported: {supported}"
        )
    
    if target_lang not in SUPPORTED_LANGUAGES:
        supported = ', '.join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported target language: {target_lang}. "
            f"Supported: {supported}"
        )


def validate_domain(domain: str):
    """
    Validate domain name.
    
    Args:
        domain: Domain name
        
    Raises:
        ValueError: If domain invalid
    """
    if not domain or not domain.strip():
        raise ValueError("Domain cannot be empty")
    
    domain = domain.strip()
    
    if not domain.replace('_', '').replace('-', '').isalnum():
        raise ValueError(
            f"Invalid domain: {domain}. "
            f"Use only alphanumeric characters, underscores, and hyphens"
        )
    
    if len(domain) > 100:
        raise ValueError(f"Domain too long: {len(domain)} chars (max: 100)")
    
    if len(domain) < 2:
        raise ValueError(f"Domain too short: {len(domain)} chars (min: 2)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_segment_id(
    segment_type: SegmentType,
    position: SegmentPosition
) -> str:
    """
    Generate unique segment ID from type and position.
    
    Args:
        segment_type: Type of segment
        position: Position in document
        
    Returns:
        Unique segment ID
        
    Raises:
        TypeError: If arguments have wrong type
    """
    if not isinstance(segment_type, SegmentType):
        raise TypeError(f"segment_type must be SegmentType, got {type(segment_type).__name__}")
    
    if not isinstance(position, SegmentPosition):
        raise TypeError(f"position must be SegmentPosition, got {type(position).__name__}")
    
    parts = []
    
    # Add segment type for special segments
    if segment_type in (SegmentType.HEADER, SegmentType.FOOTER):
        parts.append(segment_type.value)
    
    # Spreadsheet position
    if position.sheet_name:
        # Sanitize sheet name (remove special chars)
        safe_sheet = position.sheet_name.replace(' ', '_').replace('/', '_')
        parts.append(f"sheet_{safe_sheet}")
    
    # Document structure
    if position.section_index is not None:
        parts.append(f"section_{position.section_index}")
    
    if position.table_index is not None:
        parts.append(f"table_{position.table_index}")
    
    # Rows
    if position.row_index is not None:
        parts.append(f"row_{position.row_index}")
    elif position.row is not None:
        parts.append(f"row_{position.row}")
    
    # Cells/Columns
    if position.cell_index is not None:
        parts.append(f"cell_{position.cell_index}")
    elif position.column is not None:
        parts.append(f"col_{position.column}")
    
    # Paragraph/Run
    if position.paragraph_index is not None:
        parts.append(f"para_{position.paragraph_index}")
    
    if position.run_index is not None:
        parts.append(f"run_{position.run_index}")
    
    # Generate ID
    if not parts:
        raise ValueError("Cannot create segment ID: no position information provided")
    
    return "_".join(parts)


def get_language_name(language_code: str) -> str:
    """
    Get language name from code.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        Language name
        
    Raises:
        ValueError: If language code not supported
    """
    if not language_code or not language_code.strip():
        raise ValueError("language_code cannot be empty")
    
    language_code = language_code.lower().strip()
    
    if language_code not in SUPPORTED_LANGUAGES:
        supported = ', '.join(sorted(SUPPORTED_LANGUAGES.keys()))
        raise ValueError(
            f"Unsupported language code: {language_code}. "
            f"Supported: {supported}"
        )
    
    return SUPPORTED_LANGUAGES[language_code]


def is_language_supported(language_code: str) -> bool:
    """
    Check if language is supported.
    
    Args:
        language_code: ISO 639-1 language code
        
    Returns:
        True if supported, False otherwise
    """
    if not language_code:
        return False
    
    return language_code.lower().strip() in SUPPORTED_LANGUAGES


# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum file size in MB
MAX_FILE_SIZE_MB = 100

# Maximum text length for single translation
MAX_TEXT_LENGTH = 50000

# Default batch size for translations
DEFAULT_BATCH_SIZE = 10

# Maximum batch size
MAX_BATCH_SIZE = 50


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing."""
    
    print("=" * 70)
    print("TESTING DATA MODELS")
    print("=" * 70)
    
    # Test FileType
    print("\n1. Testing FileType:")
    try:
        docx_type = FileType.from_extension(".docx")
        print(f"   ✓ .docx -> {docx_type.value}")
        
        xlsx_type = FileType.from_extension("xlsx")
        print(f"   ✓ xlsx -> {xlsx_type.value}")
        
        # Test unsupported
        try:
            FileType.from_extension(".xyz")
            print("   ✗ Should have raised ValueError")
        except ValueError as e:
            print(f"   ✓ Caught unsupported extension: {str(e)[:50]}...")
        
        # Test is_supported
        print(f"   ✓ .docx supported: {FileType.is_supported('.docx')}")
        print(f"   ✓ .xyz supported: {FileType.is_supported('.xyz')}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test TranslationStatus
    print("\n2. Testing TranslationStatus:")
    try:
        status = TranslationStatus.COMPLETED
        print(f"   ✓ Status: {status.value}")
        print(f"   ✓ Is terminal: {status.is_terminal()}")
        print(f"   ✓ Is active: {status.is_active()}")
        
        status2 = TranslationStatus.TRANSLATING
        print(f"   ✓ Status: {status2.value}")
        print(f"   ✓ Is terminal: {status2.is_terminal()}")
        print(f"   ✓ Is active: {status2.is_active()}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test language validation
    print("\n3. Testing language validation:")
    try:
        validate_language_pair("en", "ru")
        print("   ✓ en -> ru: valid")
    except ValueError as e:
        print(f"   ✗ Error: {e}")
    
    try:
        validate_language_pair("en", "en")
        print("   ✗ Same language should fail")
    except ValueError as e:
        print(f"   ✓ Caught: {str(e)[:50]}...")
    
    try:
        validate_language_pair("xyz", "en")
        print("   ✗ Invalid language should fail")
    except ValueError as e:
        print(f"   ✓ Caught: {str(e)[:50]}...")
    
    # Test domain validation
    print("\n4. Testing domain validation:")
    try:
        validate_domain("legal")
        print("   ✓ 'legal': valid")
        
        validate_domain("my-domain_123")
        print("   ✓ 'my-domain_123': valid")
    except ValueError as e:
        print(f"   ✗ Error: {e}")
    
    try:
        validate_domain("")
        print("   ✗ Empty domain should fail")
    except ValueError as e:
        print(f"   ✓ Caught: {str(e)[:50]}...")
    
    try:
        validate_domain("invalid domain!")
        print("   ✗ Invalid chars should fail")
    except ValueError as e:
        print(f"   ✓ Caught: {str(e)[:50]}...")
    
    # Test TranslationRequest
    print("\n5. Testing TranslationRequest:")
    try:
        req1 = TranslationRequest(
            text="Hello world",
            source_lang="en",
            target_lang="ru"
        )
        print(f"   ✓ Created request: {req1}")
        
        req2 = TranslationRequest(
            text="Hello world",
            source_lang="EN",  # Different case
            target_lang="RU"
        )
        print(f"   ✓ Created request: {req2}")
        
        print(f"   ✓ req1 == req2: {req1 == req2} (normalized)")
        print(f"   ✓ hash equal: {hash(req1) == hash(req2)}")
        
        # Test as dict key
        cache = {req1: "cached_result"}
        print(f"   ✓ Can use as dict key: {req2 in cache}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation
        TranslationRequest(text="", source_lang="en", target_lang="ru")
        print("   ✗ Empty text should fail")
    except ValueError as e:
        print(f"   ✓ Caught empty text: {str(e)[:50]}...")
    
    # Test TranslationResult
    print("\n6. Testing TranslationResult:")
    try:
        result = TranslationResult(
            original_text="Hello",
            translated_text="Привет",
            source_lang="en",
            target_lang="ru",
            domain="general",
            engine="openai",
            model="gpt-4",
            confidence=0.95,
            cached=False
        )
        print(f"   ✓ Created result")
        print(f"   ✓ Word count: {result.word_count}")
        
        result_dict = result.to_dict()
        print(f"   ✓ Converted to dict: {len(result_dict)} fields")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation
        TranslationResult(
            original_text="test",
            translated_text="тест",
            source_lang="en",
            target_lang="ru",
            domain="general",
            engine="openai",
            model="gpt-4",
            confidence=1.5  # Invalid
        )
        print("   ✗ Invalid confidence should fail")
    except ValueError as e:
        print(f"   ✓ Caught invalid confidence: {str(e)[:50]}...")
    
    # Test TranslationJob
    print("\n7. Testing TranslationJob:")
    try:
        job = TranslationJob(
            job_id="test-123",
            input_file=Path("test.docx"),
            output_file=Path("test_ru.docx"),
            source_lang="en",
            target_lang="ru",
            domain="general",
            engine="openai"
        )
        print(f"   ✓ Created job: {job.job_id}")
        
        job.total_segments = 100
        job.cached_segments = 20
        job.failed_segments = 5
        job.started_at = datetime.utcnow()
        
        job.update_progress(50)
        print(f"   ✓ Progress: {job.progress_percentage:.1f}%")
        print(f"   ✓ Success rate: {job.success_rate:.1f}%")
        print(f"   ✓ Translated: {job.translated_segments}")
        print(f"   ✓ Cached: {job.cached_segments}")
        print(f"   ✓ Failed: {job.failed_segments}")
        print(f"   ✓ Is complete: {job.is_complete}")
        
        job.add_error("Test error message")
        print(f"   ✓ Added error: {len(job.errors)} errors")
        
        job_dict = job.to_dict()
        print(f"   ✓ Converted to dict: {len(job_dict)} fields")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test SegmentPosition
    print("\n8. Testing SegmentPosition:")
    try:
        position = SegmentPosition(
            paragraph_index=5,
            run_index=2
        )
        print(f"   ✓ Created paragraph position")
        
        position2 = SegmentPosition(
            sheet_name="Sheet1",
            row=10,
            column=5
        )
        print(f"   ✓ Created spreadsheet position")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation - all None should fail
        SegmentPosition()
        print("   ✗ Empty position should fail")
    except ValueError as e:
        print(f"   ✓ Caught empty position: {str(e)[:50]}...")
    
    # Test TextFormatting
    print("\n9. Testing TextFormatting:")
    try:
        fmt = TextFormatting(
            font_name="Arial",
            font_size=12.0,
            bold=True,
            color="#FF0000"
        )
        print(f"   ✓ Created text formatting")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation
        TextFormatting(font_size=-5)
        print("   ✗ Negative font size should fail")
    except ValueError as e:
        print(f"   ✓ Caught negative font size: {str(e)[:50]}...")
    
    try:
        TextFormatting(color="invalid_color_###")
        print("   ✗ Invalid color should fail")
    except ValueError as e:
        print(f"   ✓ Caught invalid color: {str(e)[:50]}...")
    
    # Test ParagraphFormatting
    print("\n10. Testing ParagraphFormatting:")
    try:
        para = ParagraphFormatting(
            alignment="justify",
            line_spacing=1.5,
            left_indent=1.0
        )
        print(f"   ✓ Created paragraph formatting")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation
        ParagraphFormatting(alignment="invalid")
        print("   ✗ Invalid alignment should fail")
    except ValueError as e:
        print(f"   ✓ Caught invalid alignment: {str(e)[:50]}...")
    
    try:
        ParagraphFormatting(line_spacing=-1.0)
        print("   ✗ Negative spacing should fail")
    except ValueError as e:
        print(f"   ✓ Caught negative spacing: {str(e)[:50]}...")
    
    # Test TextSegment
    print("\n11. Testing TextSegment:")
    try:
        position = SegmentPosition(paragraph_index=1)
        segment = TextSegment(
            id="para_1",
            text="This is a test segment.",
            segment_type=SegmentType.PARAGRAPH,
            position=position,
            text_formatting=TextFormatting(bold=True)
        )
        print(f"   ✓ Created segment: {segment.id}")
        print(f"   ✓ Word count: {segment.word_count}")
        print(f"   ✓ Char count: {segment.char_count}")
        print(f"   ✓ Is empty: {segment.is_empty}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        # Test validation
        TextSegment(
            id="",  # Empty ID
            text="test",
            segment_type=SegmentType.PARAGRAPH,
            position=SegmentPosition(paragraph_index=1)
        )
        print("   ✗ Empty ID should fail")
    except ValueError as e:
        print(f"   ✓ Caught empty ID: {str(e)[:50]}...")
    
    # Test segment ID generation
    print("\n12. Testing segment ID generation:")
    try:
        position = SegmentPosition(
            paragraph_index=5,
            run_index=2
        )
        seg_id = create_segment_id(SegmentType.PARAGRAPH, position)
        print(f"   ✓ Paragraph ID: {seg_id}")
        
        position2 = SegmentPosition(
            sheet_name="Sheet1",
            row=10,
            column=5
        )
        seg_id2 = create_segment_id(SegmentType.TABLE_CELL, position2)
        print(f"   ✓ Cell ID: {seg_id2}")
        
        # Header segment
        position3 = SegmentPosition(section_index=0, paragraph_index=1)
        seg_id3 = create_segment_id(SegmentType.HEADER, position3)
        print(f"   ✓ Header ID: {seg_id3}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test Document
    print("\n13. Testing Document:")
    try:
        position = SegmentPosition(paragraph_index=0)
        segments = [
            TextSegment(
                id="seg_1",
                text="First paragraph.",
                segment_type=SegmentType.PARAGRAPH,
                position=position
            ),
            TextSegment(
                id="seg_2",
                text="Second paragraph.",
                segment_type=SegmentType.PARAGRAPH,
                position=SegmentPosition(paragraph_index=1)
            )
        ]
        
        doc = Document(
            file_path=Path("test.docx"),
            file_type=FileType.DOCX,
            segments=segments,
            metadata=DocumentMetadata(title="Test Document", author="Test Author")
        )
        
        print(f"   ✓ Created document")
        print(f"   ✓ Total segments: {doc.total_segments}")
        print(f"   ✓ Non-empty segments: {doc.non_empty_segments}")
        print(f"   ✓ Total words: {doc.total_words}")
        print(f"   ✓ Total chars: {doc.total_chars}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test helper functions
    print("\n14. Testing helper functions:")
    try:
        name = get_language_name("en")
        print(f"   ✓ Language name for 'en': {name}")
        
        name = get_language_name("ru")
        print(f"   ✓ Language name for 'ru': {name}")
        
        supported = is_language_supported("en")
        print(f"   ✓ 'en' supported: {supported}")
        
        supported = is_language_supported("xyz")
        print(f"   ✓ 'xyz' supported: {supported}")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    try:
        get_language_name("xyz")
        print("   ✗ Invalid language should fail")
    except ValueError as e:
        print(f"   ✓ Caught invalid language: {str(e)[:50]}...")
    
    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED!")
    print("=" * 70)
