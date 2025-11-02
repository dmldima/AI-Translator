"""Core data models for document processing."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any


class FileType(Enum):
    """Supported document file types."""
    DOCX = "docx"
    XLSX = "xlsx"
    PDF = "pdf"
    TXT = "txt"


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
    
    # Section (DOCX headers/footers) - NEW
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
