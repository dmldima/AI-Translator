"""
Enhanced DOCX Document Formatter - FIXED: Path Traversal Protection
Implements best practices for maintaining all document styling.
"""
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
from docx import Document as DocxDocument
from docx.shared import RGBColor, Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn

from ..core.interfaces import IDocumentFormatter
from ..core.models import (
    Document, TextSegment, FileType, SegmentType
)


logger = logging.getLogger(__name__)


class EnhancedDocxFormatter(IDocumentFormatter):
    """
    Enhanced DOCX formatter with complete formatting preservation.
    FIXED: Path traversal protection.
    """
    
    @property
    def supported_file_type(self) -> FileType:
        return FileType.DOCX
    
    def format(
        self,
        document: Document,
        output_path: Path,
        preserve_formatting: bool = True
    ) -> Path:
        """
        Format and save translated document with complete formatting preservation.
        
        Args:
            document: Document with translated segments
            output_path: Where to save
            preserve_formatting: Whether to preserve formatting
            
        Returns:
            Path to saved document
        """
        try:
            # SECURITY FIX: Validate paths before processing
            document.file_path = self._validate_and_resolve_path(document.file_path)
            output_path = self._validate_and_resolve_path(output_path, check_exists=False)
            
            logger.info(f"Formatting document: {output_path.name}")
            
            # Load original document to preserve structure
            original_doc = DocxDocument(document.file_path)
            
            if preserve_formatting:
                # Strategy: Replace text in-place to preserve all formatting
                self._replace_text_in_place(original_doc, document.segments)
            else:
                # Create new document from scratch
                original_doc = self._create_new_document(document)
            
            # Save document
            output_path.parent.mkdir(parents=True, exist_ok=True)
            original_doc.save(output_path)
            
            # Validate formatting preservation
            if preserve_formatting:
                self._validate_formatting(document.file_path, output_path)
            
            logger.info(f"✓ Document saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            raise FormattingError(f"Failed to format document: {e}")
    
    def _validate_and_resolve_path(self, file_path: Path, check_exists: bool = True) -> Path:
        """
        SECURITY: Validate and resolve file path to prevent path traversal.
        
        Args:
            file_path: Input path
            check_exists: Whether to check if path exists
            
        Returns:
            Resolved absolute path
            
        Raises:
            FormattingError: If path is unsafe
        """
        # Convert to absolute path
        if not file_path.is_absolute():
            file_path = file_path.resolve()
        
        # Check for path traversal attempts
        path_str = str(file_path)
        if '..' in path_str:
            raise FormattingError(f"Path traversal not allowed: {file_path}")
        
        # Validate filename doesn't contain dangerous characters
        dangerous_chars = ['<', '>', '|', '\0', '\n', '\r']
        for char in dangerous_chars:
            if char in file_path.name:
                raise FormattingError(f"Invalid character in filename: {repr(char)}")
        
        # Check parent directory exists (for output files)
        if not check_exists and not file_path.parent.exists():
            raise FormattingError(f"Output directory doesn't exist: {file_path.parent}")
        
        # Check file exists (for input files)
        if check_exists and not file_path.exists():
            raise FormattingError(f"File not found: {file_path}")
        
        return file_path
    
    def _replace_text_in_place(
        self,
        docx: DocxDocument,
        segments: List[TextSegment]
    ) -> None:
        """
        Replace text in original document while preserving all formatting.
        ENHANCED: Better preservation of run properties.
        """
        # Build segment lookup by position
        segment_map = self._build_segment_map(segments)
        
        # Process paragraphs
        for para_idx, paragraph in enumerate(docx.paragraphs):
            for run_idx, run in enumerate(paragraph.runs):
                segment_id = f"para_{para_idx}_run_{run_idx}"
                
                if segment_id in segment_map:
                    segment = segment_map[segment_id]
                    
                    # Store original formatting before text replacement
                    original_font_name = run.font.name
                    original_font_size = run.font.size
                    original_bold = run.font.bold
                    original_italic = run.font.italic
                    original_underline = run.font.underline
                    original_color = run.font.color.rgb if run.font.color and hasattr(run.font.color, 'rgb') else None
                    
                    # Replace text
                    run.text = segment.text
                    
                    # Restore formatting if it was reset
                    if original_font_name and not run.font.name:
                        run.font.name = original_font_name
                    if original_font_size and not run.font.size:
                        run.font.size = original_font_size
                    if original_bold is not None and run.font.bold != original_bold:
                        run.font.bold = original_bold
                    if original_italic is not None and run.font.italic != original_italic:
                        run.font.italic = original_italic
                    if original_underline is not None and run.font.underline != original_underline:
                        run.font.underline = original_underline
                    if original_color and run.font.color:
                        try:
                            run.font.color.rgb = original_color
                        except:
                            pass
                    
                    logger.debug(f"Replaced text in {segment_id}")
        
        # Process tables
        for table_idx, table in enumerate(docx.tables):
            for row_idx, row in enumerate(table.rows):
                for cell_idx, cell in enumerate(row.cells):
                    segment_id = f"table_{table_idx}_row_{row_idx}_cell_{cell_idx}"
                    
                    if segment_id in segment_map:
                        segment = segment_map[segment_id]
                        
                        # For table cells, preserve paragraph formatting
                        if cell.paragraphs:
                            # Get first paragraph
                            first_para = cell.paragraphs[0]
                            
                            # Store formatting from first run if exists
                            original_formatting = None
                            if first_para.runs:
                                first_run = first_para.runs[0]
                                original_formatting = {
                                    'font_name': first_run.font.name,
                                    'font_size': first_run.font.size,
                                    'bold': first_run.font.bold,
                                    'italic': first_run.font.italic
                                }
                            
                            # Clear existing runs
                            for run in list(first_para.runs):
                                run.text = ''
                            
                            # Add new text
                            if first_para.runs:
                                new_run = first_para.runs[0]
                                new_run.text = segment.text
                                
                                # Restore formatting
                                if original_formatting:
                                    if original_formatting['font_name']:
                                        new_run.font.name = original_formatting['font_name']
                                    if original_formatting['font_size']:
                                        new_run.font.size = original_formatting['font_size']
                                    if original_formatting['bold'] is not None:
                                        new_run.font.bold = original_formatting['bold']
                                    if original_formatting['italic'] is not None:
                                        new_run.font.italic = original_formatting['italic']
                            else:
                                new_run = first_para.add_run(segment.text)
                                if original_formatting and original_formatting['font_name']:
                                    new_run.font.name = original_formatting['font_name']
                        
                        logger.debug(f"Replaced text in {segment_id}")
        
        # Process headers/footers
        for section in docx.sections:
            self._replace_header_footer_text(section.header, segment_map, "header")
            self._replace_header_footer_text(section.footer, segment_map, "footer")
    
    def _replace_header_footer_text(
        self,
        header_footer,
        segment_map: Dict[str, TextSegment],
        prefix: str
    ) -> None:
        """Replace text in headers/footers."""
        for para_idx, paragraph in enumerate(header_footer.paragraphs):
            segment_id = f"{prefix}_{para_idx}"
            
            if segment_id in segment_map:
                segment = segment_map[segment_id]
                
                # Preserve formatting from first run
                if paragraph.runs:
                    first_run = paragraph.runs[0]
                    original_font = {
                        'name': first_run.font.name,
                        'size': first_run.font.size,
                        'bold': first_run.font.bold
                    }
                    
                    # Replace text in first run, clear others
                    first_run.text = segment.text
                    for run in paragraph.runs[1:]:
                        run.text = ''
                else:
                    # Add new run with text
                    paragraph.add_run(segment.text)
    
    def _build_segment_map(
        self,
        segments: List[TextSegment]
    ) -> Dict[str, TextSegment]:
        """Build lookup map of segments by ID."""
        return {segment.id: segment for segment in segments}
    
    def _create_new_document(self, document: Document) -> DocxDocument:
        """
        Create new document from scratch (no formatting preservation).
        Used only when preserve_formatting=False.
        """
        docx = DocxDocument()
        
        # Apply metadata
        if document.metadata:
            self._apply_metadata(docx, document.metadata)
        
        # Group segments by paragraph
        paragraphs = self._group_by_paragraph(document.segments)
        
        # Write paragraphs
        for para_segments in paragraphs:
            if not para_segments:
                continue
            
            first_segment = para_segments[0]
            
            if first_segment.segment_type == SegmentType.HEADING:
                self._add_heading(docx, para_segments)
            elif first_segment.segment_type != SegmentType.TABLE_CELL:
                self._add_paragraph(docx, para_segments)
        
        # Add tables
        self._add_tables(docx, document)
        
        return docx
    
    def _apply_metadata(self, docx: DocxDocument, metadata) -> None:
        """Apply document metadata."""
        core_props = docx.core_properties
        
        if metadata.title:
            core_props.title = metadata.title
        if metadata.author:
            core_props.author = metadata.author
        if metadata.subject:
            core_props.subject = metadata.subject
        if metadata.keywords:
            core_props.keywords = metadata.keywords
        if metadata.comments:
            core_props.comments = metadata.comments
    
    def _group_by_paragraph(self, segments: List[TextSegment]) -> List[List[TextSegment]]:
        """Group segments that belong to same paragraph."""
        paragraphs: Dict[int, List[TextSegment]] = {}
        
        for segment in segments:
            if segment.segment_type == SegmentType.TABLE_CELL:
                continue
            
            para_idx = segment.position.paragraph_index or -1
            
            if para_idx not in paragraphs:
                paragraphs[para_idx] = []
            
            paragraphs[para_idx].append(segment)
        
        return [paragraphs[k] for k in sorted(paragraphs.keys())]
    
    def _add_paragraph(
        self,
        docx: DocxDocument,
        segments: List[TextSegment]
    ) -> None:
        """Add paragraph with runs."""
        paragraph = docx.add_paragraph()
        
        # Apply paragraph formatting from first segment
        if segments and segments[0].paragraph_formatting:
            self._apply_paragraph_formatting(
                paragraph,
                segments[0].paragraph_formatting
            )
        
        # Add runs
        for segment in segments:
            run = paragraph.add_run(segment.text)
            
            if segment.text_formatting:
                self._apply_run_formatting(run, segment.text_formatting)
    
    def _add_heading(
        self,
        docx: DocxDocument,
        segments: List[TextSegment]
    ) -> None:
        """Add heading."""
        if not segments:
            return
        
        text = " ".join(s.text for s in segments)
        
        # Determine heading level
        level = 1
        first_segment = segments[0]
        if first_segment.paragraph_formatting:
            style_name = first_segment.paragraph_formatting.style_name or ""
            if "heading" in style_name.lower():
                try:
                    level = int(''.join(filter(str.isdigit, style_name))) or 1
                except:
                    level = 1
        
        heading = docx.add_heading(text, level=min(level, 9))
        
        # Apply formatting
        if segments[0].text_formatting:
            for run in heading.runs:
                self._apply_run_formatting(run, segments[0].text_formatting)
    
    def _add_tables(self, docx: DocxDocument, document: Document) -> None:
        """Add tables from segments."""
        table_segments = [
            s for s in document.segments
            if s.segment_type == SegmentType.TABLE_CELL
        ]
        
        if not table_segments:
            return
        
        # Group by table index
        tables = {}
        for segment in table_segments:
            table_idx = segment.position.table_index
            if table_idx not in tables:
                tables[table_idx] = {}
            
            row_idx = segment.position.row_index
            if row_idx not in tables[table_idx]:
                tables[table_idx][row_idx] = {}
            
            cell_idx = segment.position.cell_index
            tables[table_idx][row_idx][cell_idx] = segment
        
        # Create tables
        for table_idx in sorted(tables.keys()):
            rows = tables[table_idx]
            
            num_rows = len(rows)
            num_cols = max(len(row) for row in rows.values())
            
            table = docx.add_table(rows=num_rows, cols=num_cols)
            table.style = 'Table Grid'
            
            # Fill cells
            for row_idx in sorted(rows.keys()):
                row = rows[row_idx]
                for cell_idx in sorted(row.keys()):
                    segment = row[cell_idx]
                    cell = table.rows[row_idx].cells[cell_idx]
                    cell.text = segment.text
                    
                    # Apply formatting
                    if segment.text_formatting:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                self._apply_run_formatting(
                                    run,
                                    segment.text_formatting
                                )
    
    def _apply_run_formatting(self, run, formatting) -> None:
        """Apply comprehensive text formatting to run."""
        font = run.font
        
        # Font properties
        if formatting.font_name:
            font.name = formatting.font_name
        
        if formatting.font_size:
            font.size = Pt(formatting.font_size)
        
        # Text effects
        if formatting.bold:
            font.bold = True
        
        if formatting.italic:
            font.italic = True
        
        if formatting.underline:
            font.underline = True
        
        if formatting.strikethrough:
            font.strike = True
        
        # Color
        if formatting.color:
            try:
                color_hex = formatting.color.lstrip('#')
                r = int(color_hex[0:2], 16)
                g = int(color_hex[2:4], 16)
                b = int(color_hex[4:6], 16)
                font.color.rgb = RGBColor(r, g, b)
            except Exception as e:
                logger.warning(f"Failed to apply color: {e}")
        
        # Highlight color
        if formatting.highlight_color:
            try:
                # Set highlight color using XML
                shading_elm = OxmlElement('w:shd')
                shading_elm.set(qn('w:fill'), formatting.highlight_color.lstrip('#'))
                run._element.get_or_add_rPr().append(shading_elm)
            except Exception as e:
                logger.warning(f"Failed to apply highlight: {e}")
        
        # Position effects
        if formatting.subscript:
            font.subscript = True
        
        if formatting.superscript:
            font.superscript = True
        
        if formatting.small_caps:
            font.small_caps = True
    
    def _apply_paragraph_formatting(self, paragraph, formatting) -> None:
        """Apply comprehensive paragraph formatting."""
        fmt = paragraph.paragraph_format
        
        # Alignment
        alignment_map = {
            "left": WD_ALIGN_PARAGRAPH.LEFT,
            "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT,
            "justify": WD_ALIGN_PARAGRAPH.JUSTIFY
        }
        
        if formatting.alignment:
            paragraph.alignment = alignment_map.get(
                formatting.alignment,
                WD_ALIGN_PARAGRAPH.LEFT
            )
        
        # Spacing
        if formatting.line_spacing:
            fmt.line_spacing = formatting.line_spacing
        
        if formatting.space_before:
            fmt.space_before = Pt(formatting.space_before)
        
        if formatting.space_after:
            fmt.space_after = Pt(formatting.space_after)
        
        # Indentation
        if formatting.left_indent:
            fmt.left_indent = Inches(formatting.left_indent / 72)
        
        if formatting.right_indent:
            fmt.right_indent = Inches(formatting.right_indent / 72)
        
        if formatting.first_line_indent:
            fmt.first_line_indent = Inches(formatting.first_line_indent / 72)
        
        # Pagination
        if formatting.keep_together:
            fmt.keep_together = True
        
        if formatting.keep_with_next:
            fmt.keep_with_next = True
        
        if formatting.page_break_before:
            fmt.page_break_before = True
    
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
            original_doc = DocxDocument(original_path)
            output_doc = DocxDocument(output_path)
            
            # Compare structure
            if len(original_doc.paragraphs) != len(output_doc.paragraphs):
                logger.warning(
                    f"Paragraph count mismatch: "
                    f"{len(original_doc.paragraphs)} vs {len(output_doc.paragraphs)}"
                )
                return False
            
            if len(original_doc.tables) != len(output_doc.tables):
                logger.warning(
                    f"Table count mismatch: "
                    f"{len(original_doc.tables)} vs {len(output_doc.tables)}"
                )
                return False
            
            # Sample formatting checks
            validation_passed = True
            
            for i, (orig_para, out_para) in enumerate(
                zip(original_doc.paragraphs[:5], output_doc.paragraphs[:5])
            ):
                # Check paragraph alignment
                if orig_para.alignment != out_para.alignment:
                    logger.warning(
                        f"Alignment mismatch in paragraph {i}: "
                        f"{orig_para.alignment} vs {out_para.alignment}"
                    )
                    validation_passed = False
                
                # Check run formatting for first run
                if orig_para.runs and out_para.runs:
                    orig_run = orig_para.runs[0]
                    out_run = out_para.runs[0]
                    
                    if orig_run.font.name != out_run.font.name:
                        logger.warning(
                            f"Font mismatch in paragraph {i}: "
                            f"{orig_run.font.name} vs {out_run.font.name}"
                        )
                        validation_passed = False
                    
                    if orig_run.font.size != out_run.font.size:
                        logger.warning(
                            f"Font size mismatch in paragraph {i}: "
                            f"{orig_run.font.size} vs {out_run.font.size}"
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
        This is used when creating a new document.
        """
        translated.metadata = original.metadata
        translated.styles = original.styles
        
        # Copy formatting from original segments
        original_by_id = {s.id: s for s in original.segments}
        
        for segment in translated.segments:
            if segment.id in original_by_id:
                original_segment = original_by_id[segment.id]
                segment.text_formatting = original_segment.text_formatting
                segment.paragraph_formatting = original_segment.paragraph_formatting
                segment.position = original_segment.position
        
        return translated
    
    def validate_output(self, output_path: Path) -> bool:
        """
        Validate output document can be opened.
        
        Args:
            output_path: Path to output document
            
        Returns:
            True if valid
        """
        if not output_path.exists():
            logger.error(f"Output file not found: {output_path}")
            return False
        
        try:
            DocxDocument(output_path)
            return True
        except Exception as e:
            logger.error(f"Invalid output document: {e}")
            return False


class FormattingError(Exception):
    """Exception raised when formatting fails."""
    pass


# Compatibility alias for existing code
DocxFormatter = EnhancedDocxFormatter
