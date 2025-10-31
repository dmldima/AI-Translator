"""
Enhanced XLSX Document Formatter with Complete Formatting Preservation.
Handles merged cells, conditional formatting, and all cell properties.
FIXED: Proper merged cell handling
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, Protection
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from ..core.interfaces import IDocumentFormatter
from ..core.models import Document, TextSegment, FileType


logger = logging.getLogger(__name__)


class EnhancedXlsxFormatter(IDocumentFormatter):
    """
    Enhanced XLSX formatter with complete formatting preservation.
    
    Features:
    - Preserves all cell formatting (font, fill, border, alignment)
    - Preserves merged cells
    - Preserves column widths and row heights
    - Preserves conditional formatting
    - Preserves cell protection
    - Preserves formulas (where appropriate)
    - Validates formatting preservation
    """
    
    @property
    def supported_file_type(self) -> FileType:
        return FileType.XLSX
    
    def format(
        self,
        document: Document,
        output_path: Path,
        preserve_formatting: bool = True
    ) -> Path:
        """
        Format and save translated spreadsheet with complete formatting preservation.
        
        Args:
            document: Document with translated segments
            output_path: Where to save
            preserve_formatting: Whether to preserve formatting
            
        Returns:
            Path to saved document
        """
        try:
            logger.info(f"Formatting spreadsheet: {output_path.name}")
            
            if preserve_formatting:
                # Load original workbook and replace text in-place
                wb = load_workbook(
                    document.file_path,
                    data_only=False,  # Keep formulas
                    keep_vba=True     # Keep macros
                )
                self._replace_text_in_place(wb, document.segments)
            else:
                # Create new workbook from scratch
                wb = self._create_new_workbook(document)
            
            # Save workbook
            output_path.parent.mkdir(parents=True, exist_ok=True)
            wb.save(output_path)
            
            # Validate formatting preservation
            if preserve_formatting:
                self._validate_formatting(document.file_path, output_path)
            
            logger.info(f"✓ Spreadsheet saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            raise FormattingError(f"Failed to format spreadsheet: {e}")
    
    def _replace_text_in_place(
        self,
        wb: Workbook,
        segments: List[TextSegment]
    ) -> None:
        """
        Replace text in original workbook while preserving all formatting.
        ENHANCED: Proper merged cell handling.
        """
        segment_map = self._build_segment_map(segments)
        
        # Process all worksheets
        for sheet in wb.worksheets:
            sheet_name = sheet.title
            
            # Get merged cell ranges for this sheet
            merged_ranges = list(sheet.merged_cells.ranges)
            
            # Create mapping of merged cells
            merged_cells_info = self._build_merged_cells_map(merged_ranges)
            
            for row in sheet.iter_rows():
                for cell in row:
                    segment_id = f"sheet_{sheet_name}_row_{cell.row}_col_{cell.column}"
                    
                    if segment_id in segment_map:
                        segment = segment_map[segment_id]
                        
                        # Check if this is a merged cell
                        if cell.coordinate in merged_cells_info:
                            top_left_coord = merged_cells_info[cell.coordinate]
                            
                            # Only update if this is the top-left cell
                            if cell.coordinate == top_left_coord:
                                cell.value = segment.text
                                logger.debug(f"Updated merged cell {segment_id}")
                        else:
                            # Regular cell - just replace value
                            cell.value = segment.text
                            logger.debug(f"Updated cell {segment_id}")
    
    def _build_merged_cells_map(self, merged_ranges: List) -> Dict[str, str]:
        """
        Build a map of all merged cell coordinates to their top-left coordinate.
        
        Args:
            merged_ranges: List of merged cell ranges
            
        Returns:
            Dict mapping any merged cell coordinate to its top-left coordinate
        """
        merged_cells_map = {}
        
        for merged_range in merged_ranges:
            # Get top-left coordinate
            top_left = f"{get_column_letter(merged_range.min_col)}{merged_range.min_row}"
            
            # Map all cells in this range to the top-left
            for row in range(merged_range.min_row, merged_range.max_row + 1):
                for col in range(merged_range.min_col, merged_range.max_col + 1):
                    coord = f"{get_column_letter(col)}{row}"
                    merged_cells_map[coord] = top_left
        
        return merged_cells_map
    
    def _build_segment_map(
        self,
        segments: List[TextSegment]
    ) -> Dict[str, TextSegment]:
        """Build lookup map of segments by ID."""
        return {segment.id: segment for segment in segments}
    
    def _create_new_workbook(self, document: Document) -> Workbook:
        """
        Create new workbook from scratch (no formatting preservation).
        Used only when preserve_formatting=False.
        """
        wb = Workbook()
        
        # Remove default sheet
        if 'Sheet' in wb.sheetnames:
            wb.remove(wb['Sheet'])
        
        # Apply metadata
        if document.metadata:
            self._apply_metadata(wb, document.metadata)
        
        # Group segments by sheet
        sheets = self._group_by_sheet(document.segments)
        
        # Create sheets and write data
        for sheet_name, segments in sheets.items():
            ws = wb.create_sheet(title=sheet_name)
            
            for segment in segments:
                row = segment.position.row
                col = segment.position.column
                
                cell = ws.cell(row=row, column=col, value=segment.text)
                
                # Apply formatting if available
                if segment.cell_formatting:
                    self._apply_cell_formatting(
                        cell,
                        ws,
                        segment.cell_formatting
                    )
        
        return wb
    
    def _apply_metadata(self, wb: Workbook, metadata) -> None:
        """Apply workbook metadata."""
        props = wb.properties
        
        if metadata.title:
            props.title = metadata.title
        if metadata.author:
            props.creator = metadata.author
        if metadata.subject:
            props.subject = metadata.subject
        if metadata.keywords:
            props.keywords = metadata.keywords
        if metadata.comments:
            props.description = metadata.comments
    
    def _group_by_sheet(
        self,
        segments: List[TextSegment]
    ) -> Dict[str, List[TextSegment]]:
        """Group segments by sheet name."""
        sheets = {}
        
        for segment in segments:
            sheet_name = segment.position.sheet_name or "Sheet1"
            
            if sheet_name not in sheets:
                sheets[sheet_name] = []
            
            sheets[sheet_name].append(segment)
        
        return sheets
    
    def _apply_cell_formatting(self, cell, ws: Worksheet, formatting) -> None:
        """Apply comprehensive cell formatting."""
        # Font
        if any([
            formatting.font_name,
            formatting.font_size,
            formatting.font_bold,
            formatting.font_italic,
            formatting.font_color
        ]):
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
                color = formatting.font_color.lstrip('#')
                font_kwargs['color'] = color
            
            cell.font = Font(**font_kwargs)
        
        # Fill (background color)
        if formatting.fill_color:
            color = formatting.fill_color.lstrip('#')
            pattern = formatting.fill_pattern or 'solid'
            cell.fill = PatternFill(
                start_color=color,
                end_color=color,
                fill_type=pattern
            )
        
        # Alignment
        if any([
            formatting.horizontal_alignment,
            formatting.vertical_alignment,
            formatting.wrap_text
        ]):
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
    
    def _validate_formatting(
        self,
        original_path: Path,
        output_path: Path
    ) -> bool:
        """
        Validate that formatting was preserved.
        
        Returns:
            True if validation passed
        """
        try:
            original_wb = load_workbook(original_path, data_only=False)
            output_wb = load_workbook(output_path, data_only=False)
            
            # Compare structure
            if len(original_wb.worksheets) != len(output_wb.worksheets):
                logger.warning(
                    f"Worksheet count mismatch: "
                    f"{len(original_wb.worksheets)} vs {len(output_wb.worksheets)}"
                )
                return False
            
            validation_passed = True
            
            # Sample formatting checks on first sheet
            if original_wb.worksheets and output_wb.worksheets:
                orig_sheet = original_wb.worksheets[0]
                out_sheet = output_wb.worksheets[0]
                
                # Check merged cells
                orig_merged = set(str(r) for r in orig_sheet.merged_cells.ranges)
                out_merged = set(str(r) for r in out_sheet.merged_cells.ranges)
                
                if orig_merged != out_merged:
                    logger.warning(
                        f"Merged cells mismatch: "
                        f"{len(orig_merged)} vs {len(out_merged)}"
                    )
                    validation_passed = False
                
                # Sample cell formatting checks (first 5 rows, 5 cols)
                for row in range(1, min(6, orig_sheet.max_row + 1)):
                    for col in range(1, min(6, orig_sheet.max_column + 1)):
                        orig_cell = orig_sheet.cell(row, col)
                        out_cell = out_sheet.cell(row, col)
                        
                        # Check font
                        if orig_cell.font.name != out_cell.font.name:
                            logger.warning(
                                f"Font mismatch at ({row},{col}): "
                                f"{orig_cell.font.name} vs {out_cell.font.name}"
                            )
                            validation_passed = False
                        
                        # Check fill
                        if orig_cell.fill.start_color != out_cell.fill.start_color:
                            logger.warning(
                                f"Fill color mismatch at ({row},{col})"
                            )
                            validation_passed = False
                
                # Check column widths
                for col_letter in ['A', 'B', 'C', 'D', 'E']:
                    orig_width = orig_sheet.column_dimensions[col_letter].width
                    out_width = out_sheet.column_dimensions[col_letter].width
                    
                    if orig_width != out_width:
                        logger.warning(
                            f"Column width mismatch for {col_letter}: "
                            f"{orig_width} vs {out_width}"
                        )
                        validation_passed = False
            
            if validation_passed:
                logger.info("✓ Formatting validation passed")
            else:
                logger.warning("⚠ Formatting validation found differences")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Formatting validation failed: {e}")
            return False
    
    def preserve_styles(
        self,
        original: Document,
        translated: Document
    ) -> Document:
        """
        Copy styles from original to translated.
        Used when creating a new document.
        """
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
        """
        Validate output spreadsheet can be opened.
        
        Args:
            output_path: Path to output document
            
        Returns:
            True if valid
        """
        if not output_path.exists():
            logger.error(f"Output file not found: {output_path}")
            return False
        
        try:
            load_workbook(output_path, read_only=True)
            return True
        except Exception as e:
            logger.error(f"Invalid output spreadsheet: {e}")
            return False


class FormattingError(Exception):
    """Exception raised when formatting fails."""
    pass


# Compatibility alias for existing code
XlsxFormatter = EnhancedXlsxFormatter
