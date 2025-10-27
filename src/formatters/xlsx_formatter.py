"""
XLSX Document Formatter.
Creates .xlsx files with preserved formatting.
"""
from pathlib import Path
from typing import Dict
import logging
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from ..core.interfaces import IDocumentFormatter
from ..core.models import (
    Document,
    TextSegment,
    FileType
)


logger = logging.getLogger(__name__)


class XlsxFormatter(IDocumentFormatter):
    """Formatter for Microsoft Excel .xlsx files."""
    
    @property
    def supported_file_type(self) -> FileType:
        """Supported file type."""
        return FileType.XLSX
    
    def format(
        self,
        document: Document,
        output_path: Path,
        preserve_formatting: bool = True
    ) -> Path:
        """
        Format and save translated spreadsheet.
        
        Args:
            document: Document with translated segments
            output_path: Where to save
            preserve_formatting: Whether to preserve formatting
            
        Returns:
            Path to saved document
        """
        try:
            # Create new workbook
            wb = Workbook()
            
            # Remove default sheet
            if 'Sheet' in wb.sheetnames:
                wb.remove(wb['Sheet'])
            
            # Apply metadata
            self._apply_metadata(wb, document)
            
            # Group segments by sheet
            sheets = self._group_by_sheet(document.segments)
            
            # Create sheets and write data
            for sheet_name, segments in sheets.items():
                ws = wb.create_sheet(title=sheet_name)
                
                # Write cells
                for segment in segments:
                    row = segment.position.row
                    col = segment.position.column
                    
                    cell = ws.cell(row=row, column=col, value=segment.text)
                    
                    # Apply formatting
                    if preserve_formatting and segment.cell_formatting:
                        self._apply_cell_formatting(cell, ws, segment.cell_formatting)
            
            # Save workbook
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)
            
            logger.info(f"Saved formatted spreadsheet to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to format spreadsheet: {e}")
            raise FormattingError(f"Formatting failed: {e}")
    
    def preserve_styles(
        self,
        original: Document,
        translated: Document
    ) -> Document:
        """Copy styles from original to translated."""
        # Copy metadata
        translated.metadata = original.metadata
        translated.styles = original.styles
        
        # Copy formatting from original segments
        original_by_id = {s.id: s for s in original.segments}
        
        for segment in translated.segments:
            if segment.id in original_by_id:
                original_segment = original_by_id[segment.id]
                segment.cell_formatting = original_segment.cell_formatting
                segment.position = original_segment.position
        
        return translated
    
    def validate_output(self, output_path: Path) -> bool:
        """Validate output document."""
        if not output_path.exists():
            logger.error(f"Output file not found: {output_path}")
            return False
        
        try:
            from openpyxl import load_workbook
            load_workbook(output_path, read_only=True)
            return True
        except Exception as e:
            logger.error(f"Invalid output spreadsheet: {e}")
            return False
    
    def _apply_metadata(self, wb: Workbook, document: Document):
        """Apply metadata to workbook."""
        props = wb.properties
        meta = document.metadata
        
        if meta.title:
            props.title = meta.title
        if meta.author:
            props.creator = meta.author
        if meta.subject:
            props.subject = meta.subject
        if meta.keywords:
            props.keywords = meta.keywords
        if meta.comments:
            props.description = meta.comments
    
    def _group_by_sheet(self, segments: list) -> Dict[str, list]:
        """Group segments by sheet name."""
        sheets = {}
        
        for segment in segments:
            sheet_name = segment.position.sheet_name or "Sheet1"
            
            if sheet_name not in sheets:
                sheets[sheet_name] = []
            
            sheets[sheet_name].append(segment)
        
        return sheets
    
    def _apply_cell_formatting(self, cell, ws, formatting):
        """Apply formatting to cell."""
        # Font
        if any([formatting.font_name, formatting.font_size, formatting.font_bold, formatting.font_italic, formatting.font_color]):
            font_kwargs = {}
            
            if formatting.font_name:
                font_kwargs['name'] = formatting.font_name
            if formatting.font_size:
                font_kwargs['size'] = formatting.font_size
            if formatting.font_bold:
                font_kwargs['bold'] = True
            if formatting.font_italic:
                font_kwargs['italic'] = True
            if formatting.font_color:
                # Remove # from color
                color = formatting.font_color.lstrip('#')
                font_kwargs['color'] = color
            
            cell.font = Font(**font_kwargs)
        
        # Fill
        if formatting.fill_color:
            color = formatting.fill_color.lstrip('#')
            pattern = formatting.fill_pattern or 'solid'
            cell.fill = PatternFill(
                start_color=color,
                end_color=color,
                fill_type=pattern
            )
        
        # Alignment
        if any([formatting.horizontal_alignment, formatting.vertical_alignment, formatting.wrap_text]):
            alignment_kwargs = {}
            
            if formatting.horizontal_alignment:
                alignment_kwargs['horizontal'] = formatting.horizontal_alignment
            if formatting.vertical_alignment:
                alignment_kwargs['vertical'] = formatting.vertical_alignment
            if formatting.wrap_text:
                alignment_kwargs['wrap_text'] = True
            
            cell.alignment = Alignment(**alignment_kwargs)
        
        # Border
        if formatting.border_style:
            sides = {}
            for position in ['top', 'bottom', 'left', 'right']:
                style = formatting.border_style.get(position)
                if style:
                    sides[position] = Side(style=style)
            
            if sides:
                cell.border = Border(**sides)
        
        # Number format
        if formatting.number_format:
            cell.number_format = formatting.number_format
        
        # Column width
        if formatting.column_width:
            col_letter = get_column_letter(cell.column)
            ws.column_dimensions[col_letter].width = formatting.column_width
        
        # Row height
        if formatting.row_height:
            ws.row_dimensions[cell.row].height = formatting.row_height


class FormattingError(Exception):
    """Exception raised when formatting fails."""
    pass


# ===== Example Usage =====

if __name__ == "__main__":
    from pathlib import Path
    from ..parsers.xlsx_parser import XlsxParser
    from ..core.models import TextSegment, SegmentType, SegmentPosition
    
    parser = XlsxParser()
    formatter = XlsxFormatter()
    
    test_file = Path("test_spreadsheet.xlsx")
    if test_file.exists():
        # Parse
        original_doc = parser.parse(test_file)
        print(f"Parsed: {len(original_doc.segments)} segments")
        
        # Simulate translation
        translated_doc = Document(
            file_path=Path("test_translated.xlsx"),
            file_type=FileType.XLSX,
            segments=[],
            metadata=original_doc.metadata,
            styles=original_doc.styles
        )
        
        for segment in original_doc.segments:
            translated_segment = TextSegment(
                id=segment.id,
                text=segment.text + " [TRANSLATED]",
                segment_type=segment.segment_type,
                position=segment.position,
                cell_formatting=segment.cell_formatting
            )
            translated_doc.segments.append(translated_segment)
        
        # Format and save
        output_path = Path("test_translated.xlsx")
        formatter.format(translated_doc, output_path, preserve_formatting=True)
        
        # Validate
        if formatter.validate_output(output_path):
            print(f"✓ Successfully created: {output_path}")
        else:
            print(f"✗ Failed to create valid spreadsheet")
    else:
        print(f"Test file not found: {test_file}")
