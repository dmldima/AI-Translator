"""
XLSX Document Parser.
Extracts text and formatting from Excel .xlsx files.
"""
from pathlib import Path
from typing import List, Dict
import logging
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border

from ..core.interfaces import IDocumentParser
from ..core.models import (
    Document,
    TextSegment,
    FileType,
    SegmentType,
    SegmentPosition,
    CellFormatting,
    DocumentMetadata,
    DocumentStyles
)


logger = logging.getLogger(__name__)


class XlsxParser(IDocumentParser):
    """Parser for Microsoft Excel .xlsx files."""
    
    @property
    def supported_file_type(self) -> FileType:
        """Supported file type."""
        return FileType.XLSX
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if file can be parsed."""
        return file_path.suffix.lower() == '.xlsx'
    
    def validate_document(self, file_path: Path) -> bool:
        """Validate document before parsing."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False
        
        if not self.can_parse(file_path):
            logger.error(f"Unsupported file type: {file_path.suffix}")
            return False
        
        try:
            load_workbook(file_path, read_only=True, data_only=True)
            return True
        except Exception as e:
            logger.error(f"Cannot open workbook: {e}")
            return False
    
    def parse(self, file_path: Path) -> Document:
        """
        Parse XLSX document.
        
        Args:
            file_path: Path to .xlsx file
            
        Returns:
            Document with extracted content and formatting
        """
        if not self.validate_document(file_path):
            raise ParsingError(f"Invalid document: {file_path}")
        
        try:
            # Load workbook (not read-only to access formatting)
            wb = load_workbook(file_path, data_only=False)
            
            # Extract metadata
            metadata = self._extract_metadata(wb)
            
            # Extract styles
            styles = self._extract_styles(wb)
            
            # Extract segments from all sheets
            segments = []
            
            for sheet in wb.worksheets:
                sheet_segments = self._extract_sheet_segments(sheet)
                segments.extend(sheet_segments)
            
            wb.close()
            
            document = Document(
                file_path=file_path,
                file_type=FileType.XLSX,
                segments=segments,
                metadata=metadata,
                styles=styles
            )
            
            logger.info(
                f"Parsed {file_path.name}: "
                f"{len(wb.worksheets)} sheets, "
                f"{len(segments)} segments, "
                f"{document.total_words} words"
            )
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to parse document: {e}")
            raise ParsingError(f"Parsing failed: {e}")
    
    def extract_segments(self, document: Document) -> List[TextSegment]:
        """Extract translatable segments."""
        # Filter out empty segments and numeric-only cells
        segments = [
            s for s in document.segments 
            if s.text.strip() and not self._is_numeric_only(s.text)
        ]
        
        logger.info(f"Extracted {len(segments)} translatable segments")
        return segments
    
    def _extract_metadata(self, wb) -> DocumentMetadata:
        """Extract workbook metadata."""
        props = wb.properties
        
        return DocumentMetadata(
            title=props.title,
            author=props.creator,
            subject=props.subject,
            keywords=props.keywords,
            comments=props.description,
            created=props.created,
            modified=props.modified,
            last_modified_by=props.lastModifiedBy
        )
    
    def _extract_styles(self, wb) -> DocumentStyles:
        """Extract workbook-level styles."""
        styles = DocumentStyles()
        
        # Extract default font from first cell
        for sheet in wb.worksheets:
            for row in sheet.iter_rows(min_row=1, max_row=1, max_col=1):
                for cell in row:
                    if cell.font:
                        styles.default_font = cell.font.name
                        return styles
        
        return styles
    
    def _extract_sheet_segments(self, sheet) -> List[TextSegment]:
        """Extract segments from a single sheet."""
        segments = []
        sheet_name = sheet.title
        
        for row in sheet.iter_rows():
            for cell in row:
                # Skip empty cells
                if cell.value is None or str(cell.value).strip() == "":
                    continue
                
                # Skip formulas (we want calculated values)
                text = str(cell.value)
                
                # Create segment
                segment = TextSegment(
                    id=f"sheet_{sheet_name}_row_{cell.row}_col_{cell.column}",
                    text=text,
                    segment_type=SegmentType.TABLE_CELL,
                    position=SegmentPosition(
                        sheet_name=sheet_name,
                        row=cell.row,
                        column=cell.column
                    ),
                    cell_formatting=self._extract_cell_formatting(cell, sheet)
                )
                segments.append(segment)
        
        return segments
    
    def _extract_cell_formatting(self, cell, sheet) -> CellFormatting:
        """Extract formatting from cell."""
        font = cell.font
        fill = cell.fill
        alignment = cell.alignment
        border = cell.border
        
        # Font color
        font_color = None
        if font and font.color and hasattr(font.color, 'rgb'):
            rgb = font.color.rgb
            if rgb and len(rgb) >= 6:
                # Remove alpha channel if present (ARGB -> RGB)
                if len(rgb) == 8:
                    rgb = rgb[2:]
                font_color = f"#{rgb}"
        
        # Fill color
        fill_color = None
        if fill and fill.start_color and hasattr(fill.start_color, 'rgb'):
            rgb = fill.start_color.rgb
            if rgb and len(rgb) >= 6:
                if len(rgb) == 8:
                    rgb = rgb[2:]
                fill_color = f"#{rgb}"
        
        # Alignment
        h_align = None
        v_align = None
        wrap = False
        if alignment:
            h_align = alignment.horizontal
            v_align = alignment.vertical
            wrap = alignment.wrap_text or False
        
        # Border
        border_style = None
        if border:
            border_style = {
                'top': border.top.style if border.top else None,
                'bottom': border.bottom.style if border.bottom else None,
                'left': border.left.style if border.left else None,
                'right': border.right.style if border.right else None
            }
        
        # Column width and row height
        col_letter = cell.column_letter
        column_width = sheet.column_dimensions[col_letter].width if col_letter in sheet.column_dimensions else None
        row_height = sheet.row_dimensions[cell.row].height if cell.row in sheet.row_dimensions else None
        
        return CellFormatting(
            font_name=font.name if font else None,
            font_size=font.size if font else None,
            font_bold=font.bold if font else False,
            font_italic=font.italic if font else False,
            font_color=font_color,
            fill_color=fill_color,
            fill_pattern=fill.patternType if fill else None,
            horizontal_alignment=h_align,
            vertical_alignment=v_align,
            wrap_text=wrap,
            border_style=border_style,
            number_format=cell.number_format if cell.number_format != 'General' else None,
            column_width=column_width,
            row_height=row_height
        )
    
    def _is_numeric_only(self, text: str) -> bool:
        """Check if text is numeric only (don't translate numbers)."""
        try:
            # Try to convert to number
            float(text.replace(',', '').replace(' ', ''))
            return True
        except:
            return False


class ParsingError(Exception):
    """Exception raised when parsing fails."""
    pass


# ===== Example Usage =====

if __name__ == "__main__":
    from pathlib import Path
    
    parser = XlsxParser()
    
    test_file = Path("test_spreadsheet.xlsx")
    if test_file.exists():
        document = parser.parse(test_file)
        
        print(f"Document: {document.file_path.name}")
        print(f"Total segments: {len(document.segments)}")
        print(f"Total words: {document.total_words}")
        print(f"Translatable segments: {document.translatable_segments}")
        
        print(f"\nMetadata:")
        print(f"  Title: {document.metadata.title}")
        print(f"  Author: {document.metadata.author}")
        
        print(f"\nFirst 3 segments:")
        for segment in document.segments[:3]:
            print(f"  [{segment.id}] {segment.text}")
            if segment.cell_formatting:
                fmt = segment.cell_formatting
                print(f"    Font: {fmt.font_name}, Size: {fmt.font_size}, Bold: {fmt.font_bold}")
                print(f"    Position: Sheet={segment.position.sheet_name}, Row={segment.position.row}, Col={segment.position.column}")
    else:
        print(f"Test file not found: {test_file}")
