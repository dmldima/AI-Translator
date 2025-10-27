"""
Core data models for the translation system.
Defines all data structures used throughout the application.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


# ===== Language Support =====

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
    'uk': 'Ukrainian'
}


# ===== Enums =====

class FileType(Enum):
    """Supported file types."""
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PDF = "pdf"
    TXT = "txt"
    
    @classmethod
    def from_extension(cls, extension: str) -> 'FileType':
        """Create FileType from file extension."""
        ext = extension.lower().lstrip('.')
        try:
            return cls(ext)
        except ValueError:
            raise ValueError(f"Unsupported file type: {extension}")


class SegmentType(Enum):
    """Types of text segments."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    HEADER = "header"
    FOOTER = "footer"
    FOOTNOTE = "footnote"
    CAPTION = "caption"


class TranslationStatus(Enum):
    """Translation job status."""
    PENDING = "pending"
    PARSING = "parsing"
    TRANSLATING = "translating"
    FORMATTING = "formatting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ===== Formatting Models =====

@dataclass
class TextFormatting:
    """Text-level formatting (runs/spans)."""
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikethrough: bool = False
    color: Optional[str] = None  # Hex color #RRGGBB
    highlight_color: Optional[str] = None
    subscript: bool = False
    superscript: bool = False
    small_caps: bool = False


@dataclass
class ParagraphFormatting:
    """Paragraph-level formatting."""
    alignment: Optional[str] = "left"  # left, center, right, justify
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
    """Spreadsheet cell formatting."""
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


@dataclass
class SegmentPosition:
    """Position of segment in document."""
    paragraph_index: Optional[int] = None
    run_index: Optional[int] = None
    table_index: Optional[int] = None
    row_index: Optional[int] = None
    cell_index: Optional[int] = None
    sheet_name: Optional[str] = None
    row: Optional[int] = None
    column: Optional[int] = None


# ===== Document Models =====

@dataclass
class DocumentMetadata:
    """Document metadata."""
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
    """Document-level styles."""
    default_font: Optional[str] = None
    default_font_size: Optional[float] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    margin_top: Optional[float] = None
    margin_bottom: Optional[float] = None
    margin_left: Optional[float] = None
    margin_right: Optional[float] = None


@dataclass
class TextSegment:
    """Text segment for translation."""
    id: str
    text: str
    segment_type: SegmentType
    position: SegmentPosition
    text_formatting: Optional[TextFormatting] = None
    paragraph_formatting: Optional[ParagraphFormatting] = None
    cell_formatting: Optional[CellFormatting] = None
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        """Count words in segment."""
        return len(self.text.split())


@dataclass
class Document:
    """Complete document with segments."""
    file_path: Path
    file_type: FileType
    segments: List[TextSegment]
    metadata: Optional[DocumentMetadata] = None
    styles: Optional[DocumentStyles] = None
    headers: List[TextSegment] = field(default_factory=list)
    footers: List[TextSegment] = field(default_factory=list)
    
    @property
    def total_segments(self) -> int:
        """Total number of segments."""
        return len(self.segments)
    
    @property
    def translatable_segments(self) -> int:
        """Number of non-empty segments."""
        return len([s for s in self.segments if s.text.strip()])
    
    @property
    def total_words(self) -> int:
        """Total word count."""
        return sum(s.word_count for s in self.segments)


# ===== Translation Models =====

@dataclass
class TranslationRequest:
    """Request for translation."""
    text: str
    source_lang: str
    target_lang: str
    domain: str = "general"
    segment_id: Optional[str] = None
    context: Optional[str] = None
    glossary_version: str = "latest"
    
    def __post_init__(self):
        if not all([self.text, self.source_lang, self.target_lang]):
            raise ValueError("text, source_lang, target_lang are required")


@dataclass
class TranslationResult:
    """Result of translation."""
    original_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    domain: str
    engine: str
    model: str
    segment_id: Optional[str] = None
    cached: bool = False
    confidence: float = 1.0
    glossary_applied: bool = False
    glossary_terms_used: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


@dataclass
class TranslationJob:
    """Translation job tracking."""
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """Job duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def progress_percentage(self) -> float:
        """Progress as percentage."""
        if self.total_segments == 0:
            return 0.0
        return (self.translated_segments / self.total_segments) * 100
    
    def update_progress(self, segments_completed: int):
        """Update progress."""
        self.translated_segments = segments_completed
    
    def add_error(self, error: str):
        """Add error message."""
        self.errors.append(error)
    
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
            'progress': f"{self.progress_percentage:.1f}%",
            'duration': f"{self.duration:.2f}s" if self.duration else None,
            'errors': self.errors[:5]  # First 5 errors
        }


# ===== Validation =====

def validate_language_pair(source_lang: str, target_lang: str) -> bool:
    """Validate language pair."""
    return (source_lang in SUPPORTED_LANGUAGES and 
            target_lang in SUPPORTED_LANGUAGES and 
            source_lang != target_lang)


def validate_file_type(file_path: Path) -> FileType:
    """Validate and return file type."""
    try:
        return FileType.from_extension(file_path.suffix)
    except ValueError as e:
        raise ValueError(f"Unsupported file: {file_path}") from e


# ===== Example Usage =====

if __name__ == "__main__":
    # Create a text segment
    segment = TextSegment(
        id="seg_1",
        text="Hello, world!",
        segment_type=SegmentType.PARAGRAPH,
        position=SegmentPosition(paragraph_index=0, run_index=0),
        text_formatting=TextFormatting(
            font_name="Arial",
            font_size=12.0,
            bold=True
        )
    )
    
    print(f"Segment: {segment.text}")
    print(f"Words: {segment.word_count}")
    
    # Create a translation request
    request = TranslationRequest(
        text="Hello, world!",
        source_lang="en",
        target_lang="es"
    )
    
    print(f"\nRequest: {request.source_lang} -> {request.target_lang}")
    
    # Create a translation result
    result = TranslationResult(
        original_text="Hello, world!",
        translated_text="¡Hola, mundo!",
        source_lang="en",
        target_lang="es",
        domain="general",
        engine="openai",
        model="gpt-4o-mini"
    )
    
    print(f"Result: {result.translated_text}")
    
    # Supported languages
    print(f"\nSupported languages: {len(SUPPORTED_LANGUAGES)}")
    for code, name in list(SUPPORTED_LANGUAGES.items())[:5]:
        print(f"  {code}: {name}")
