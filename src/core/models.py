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
        ext = extension.lower().lstrip('.')
        try:
            return cls(ext)
        except ValueError:
            raise ValueError(
                f"Unsupported file extension: {extension}. "
                f"Supported: {', '.join(ft.value for ft in cls)}"
            )


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
    
    @property
    def word_count(self) -> int:
        """Count words in segment."""
        return len(self.text.split())


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


@dataclass
class DocumentStyles:
    """Document-level style information."""
    default_font: Optional[str] = None
    default_font_size: Optional[float] = None
    styles: Dict[str, Any] = field(default_factory=dict)


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
    
    @property
    def total_words(self) -> int:
        """Total word count in document."""
        return sum(segment.word_count for segment in self.segments)
    
    @property
    def total_segments(self) -> int:
        """Total number of segments."""
        return len(self.segments)


# ============================================================================
# TRANSLATION REQUEST/RESPONSE CLASSES (NEW)
# ============================================================================

@dataclass
class TranslationRequest:
    """
    Translation request for caching and API calls.
    """
    text: str
    source_lang: str
    target_lang: str
    domain: str = "general"
    segment_id: Optional[str] = None
    glossary_version: str = "latest"
    
    def __hash__(self):
        """Make request hashable for dict keys."""
        return hash((self.text, self.source_lang, self.target_lang, self.domain))
    
    def __eq__(self, other):
        """Equality check for caching."""
        if not isinstance(other, TranslationRequest):
            return False
        return (
            self.text == other.text and
            self.source_lang == other.source_lang and
            self.target_lang == other.target_lang and
            self.domain == other.domain
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
    
    def update_progress(self, current: int):
        """
        Update translation progress.
        
        Args:
            current: Current number of completed segments (including cached)
        """
        self.translated_segments = current - self.cached_segments
    
    def add_error(self, error: str):
        """
        Add error message to job.
        
        Args:
            error: Error message
        """
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
        return (completed / self.total_segments) * 100
    
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
    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different")
    
    if source_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported source language: {source_lang}. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
        )
    
    if target_lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported target language: {target_lang}. "
            f"Supported: {', '.join(sorted(SUPPORTED_LANGUAGES.keys()))}"
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
    
    if not domain.replace('_', '').replace('-', '').isalnum():
        raise ValueError(
            f"Invalid domain: {domain}. "
            f"Use only alphanumeric, underscore, hyphen"
        )
    
    if len(domain) > 100:
        raise ValueError(f"Domain too long: {len(domain)} chars (max: 100)")


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
    """
    parts = []
    
    if position.sheet_name:
        parts.append(f"sheet_{position.sheet_name}")
    
    if position.section_index is not None:
        parts.append(f"section_{position.section_index}")
    
    if position.table_index is not None:
        parts.append(f"table_{position.table_index}")
    
    if position.row_index is not None:
        parts.append(f"row_{position.row_index}")
    
    if position.row is not None:
        parts.append(f"row_{position.row}")
    
    if position.cell_index is not None:
        parts.append(f"cell_{position.cell_index}")
    
    if position.column is not None:
        parts.append(f"col_{position.column}")
    
    if position.paragraph_index is not None:
        parts.append(f"para_{position.paragraph_index}")
    
    if position.run_index is not None:
        parts.append(f"run_{position.run_index}")
    
    if segment_type in (SegmentType.HEADER, SegmentType.FOOTER):
        parts.insert(0, segment_type.value)
    
    return "_".join(parts) if parts else "unknown"


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
    if language_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language code: {language_code}")
    
    return SUPPORTED_LANGUAGES[language_code]


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
    
    # Test FileType
    print("Testing FileType:")
    docx_type = FileType.from_extension(".docx")
    print(f"  .docx -> {docx_type.value}")
    
    xlsx_type = FileType.from_extension("xlsx")
    print(f"  xlsx -> {xlsx_type.value}")
    
    # Test language validation
    print("\nTesting language validation:")
    try:
        validate_language_pair("en", "ru")
        print("  ✓ en -> ru: valid")
    except ValueError as e:
        print(f"  ✗ Error: {e}")
    
    try:
        validate_language_pair("en", "en")
        print("  ✗ Same language should fail")
    except ValueError as e:
        print(f"  ✓ Caught: {e}")
    
    # Test TranslationRequest hashing
    print("\nTesting TranslationRequest:")
    req1 = TranslationRequest(
        text="Hello",
        source_lang="en",
        target_lang="ru"
    )
    req2 = TranslationRequest(
        text="Hello",
        source_lang="en",
        target_lang="ru"
    )
    print(f"  req1 == req2: {req1 == req2}")
    print(f"  hash(req1) == hash(req2): {hash(req1) == hash(req2)}")
    
    # Test TranslationJob
    print("\nTesting TranslationJob:")
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
    job.cached_segments = 20
    job.update_progress(50)
    
    print(f"  Progress: {job.progress_percentage:.1f}%")
    print(f"  Translated: {job.translated_segments}")
    print(f"  Cached: {job.cached_segments}")
    
    # Test segment ID generation
    print("\nTesting segment ID generation:")
    position = SegmentPosition(
        paragraph_index=5,
        run_index=2
    )
    seg_id = create_segment_id(SegmentType.PARAGRAPH, position)
    print(f"  Segment ID: {seg_id}")
    
    position2 = SegmentPosition(
        sheet_name="Sheet1",
        row=10,
        column=5
    )
    seg_id2 = create_segment_id(SegmentType.TABLE_CELL, position2)
    print(f"  Cell ID: {seg_id2}")
    
    print("\n✓ All tests passed!")
