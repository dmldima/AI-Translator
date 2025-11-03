"""
Enhanced DOCX Document Formatter - FULLY FIXED & IMPROVED
Implements best practices for maintaining all document styling.

FIXES:
- Path traversal protection
- Memory management with proper resource cleanup
- Atomic saves with correct temp file tracking
- Multi-run formatting preservation in table cells
- Extended font properties preservation
- Better header/footer handling
- Comprehensive validation with detailed reporting
- Segment mapping validation and coverage tracking

Version: 2.0 (Production Ready)
"""
from pathlib import Path
from typing import Dict, List, Optional, Set
import logging
import os
import uuid
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
    FULLY FIXED: All critical issues resolved.
    """
    
    def __init__(self, workspace_root: Path = None):
        """
        Initialize formatter.
        
        Args:
            workspace_root: Root directory for allowed file operations.
                           If None, uses current working directory.
        """
        self.workspace_root = (workspace_root or Path.cwd()).resolve()
    
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
            
        Raises:
            FormattingError: If formatting fails
        """
        doc = None
        temp_path = None  # Track temp file for cleanup
        
        try:
            # SECURITY FIX: Validate paths before processing
            document.file_path = self._validate_and_resolve_path(document.file_path)
            output_path = self._validate_and_resolve_path(output_path, check_exists=False)
            
            logger.info(f"Formatting document: {output_path.name}")
            
            # Load original document to preserve structure
            doc = DocxDocument(document.file_path)
            
            if preserve_formatting:
                # Strategy: Replace text in-place to preserve all formatting
                self._replace_text_in_place(doc, document.segments)
            else:
                # Create new document from scratch
                doc = self._create_new_document(document)
            
            # Atomic save using temp file - FIXED: capture temp_path
            temp_path = self._save_atomic(doc, output_path)
            
            # Validate formatting preservation
            if preserve_formatting:
                validation_passed = self._validate_formatting(document.file_path, output_path)
                if not validation_passed:
                    logger.warning("Formatting validation failed, but document was saved")
            
            logger.info(f"✓ Document saved: {output_path}")
            return output_path
            
        except FileNotFoundError as e:
            raise FormattingError(
                f"Input file not found: {document.file_path}\n"
                f"Please verify the file path and try again."
            ) from e
        
        except PermissionError as e:
            raise FormattingError(
                f"Permission denied: Cannot write to {output_path}\n"
                f"Please check file permissions or choose a different location."
            ) from e
        
        except ValueError as e:
            if "workspace" in str(e).lower():
                raise FormattingError(
                    f"Security error: File path outside allowed workspace.\n"
                    f"File: {document.file_path}\n"
                    f"Workspace: {self.workspace_root}"
                ) from e
            raise
        
        except Exception as e:
            logger.error(f"Formatting failed: {e}")
            raise FormattingError(
                f"Failed to format DOCX document.\n"
                f"Error: {e}\n"
                f"Please verify the file is not corrupted."
            ) from e
        
        finally:
            # IMPROVED: Better resource cleanup
            if doc:
                try:
                    # python-docx doesn't have explicit close, but help GC
                    if hasattr(doc, '_part'):
                        doc._part = None
                except Exception as e:
                    logger.debug(f"Error during document cleanup: {e}")
            
            # Cleanup temp file if exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug(f"Cleaned up temp file: {temp_path.name}")
                except OSError as e:
                    logger.warning(f"Could not cleanup temp file: {e}")
    
    def _save_atomic(self, doc, output_path: Path) -> Path:
        """
        Atomic save using temp file to prevent corruption.
        
        Args:
            doc: Document to save
            output_path: Target path
            
        Returns:
            Path to temp file (for cleanup tracking)
            
        Raises:
            FormattingError: If save fails
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in same directory (ensures same filesystem for atomic rename)
        temp_path = output_path.with_name(
            f'.tmp_{uuid.uuid4().hex}_{output_path.name}'
        )
        
        try:
            doc.save(temp_path)
            # Atomic rename (POSIX) / replace (Windows)
            temp_path.replace(output_path)
            logger.debug(f"Atomic save completed: {output_path.name}")
            return temp_path  # FIXED: Return for cleanup tracking
        except Exception as e:
            # Cleanup temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise FormattingError(f"Failed to save document: {e}") from e
    
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
        # Resolve to absolute path
        file_path = file_path.resolve()
        
        # Check workspace boundary
        try:
            file_path.relative_to(self.workspace_root)
        except ValueError:
            raise FormattingError(
                f"Access denied: Path outside workspace.\n"
                f"File: {file_path}\n"
                f"Workspace: {self.workspace_root}"
            )
        
        # Validate filename
        self._validate_filename(file_path.name)
        
        # Check file exists (for input files)
        if check_exists and not file_path.exists():
            raise FormattingError(f"File not found: {file_path}")
        
        return file_path
    
    def _validate_filename(self, filename: str) -> None:
        """
        Validate filename for cross-platform compatibility.
        
        Args:
            filename: Filename to validate
            
        Raises:
            FormattingError: If filename is invalid
        """
        # OS-specific dangerous characters
        if os.name == 'nt':  # Windows
            dangerous = ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\0']
        else:
            dangerous = ['/', '\0']
        
        for char in dangerous:
            if char in filename:
                raise FormattingError(f"Invalid character in filename: {repr(char)}")
        
        # Windows reserved names
        if os.name == 'nt':
            reserved = {
                'CON', 'PRN', 'AUX', 'NUL',
                'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
                'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
            }
            name_without_ext = filename.rsplit('.', 1)[0].upper()
            if name_without_ext in reserved:
                raise FormattingError(f"Reserved Windows filename: {filename}")
            
            # No trailing spaces or dots
            if filename.rstrip() != filename or filename.rstrip('.') != filename:
                raise FormattingError("Filename cannot end with space or dot on Windows")
        
        # Length check
        if len(filename) > 255:
            raise FormattingError("Filename too long (max 255 characters)")
    
    def _replace_text_in_place(
        self,
        docx: DocxDocument,
        segments: List[TextSegment]
    ) -> None:
        """
        Replace text in original document while preserving all formatting.
        
        IMPROVED: Better error handling, coverage tracking, and reporting.
        """
        # Build segment lookup by position
        segment_map = self._build_segment_map(segments)
        
        # IMPROVED: Track coverage for reporting
        segments_replaced = set()
        segments_not_found = []
        
        # Process paragraphs
        for para_idx, paragraph in enumerate(docx.paragraphs):
            for run_idx, run in enumerate(paragraph.runs):
                segment_id = f"para_{para_idx}_run_{run_idx}"
                
                if segment_id in segment_map:
                    segment = segment_map[segment_id]
                    self._replace_text_preserve_formatting(run, segment.text)
                    segments_replaced.add(segment_id)
                    logger.debug(f"✓ Replaced {segment_id}")
        
        # Process tables
        for table_idx, table in enumerate(docx.tables):
            for row_idx, row in enumerate(table.rows):
                for cell_idx, cell in enumerate(row.cells):
                    segment_id = f"table_{table_idx}_row_{row_idx}_cell_{cell_idx}"
                    
                    if segment_id in segment_map:
                        segment = segment_map[segment_id]
                        
                        # IMPROVED: Use smart multi-run preservation
                        if cell.paragraphs:
                            self._replace_cell_text(cell.paragraphs[0], segment.text)
                        
                        segments_replaced.add(segment_id)
                        logger.debug(f"✓ Replaced {segment_id}")
        
        # Process headers/footers
        for section_idx, section in enumerate(docx.sections):
            self._replace_header_footer_text(
                section.header, 
                segment_map, 
                "header",
                section_idx,
                segments_replaced
            )
            self._replace_header_footer_text(
                section.footer, 
                segment_map, 
                "footer",
                section_idx,
                segments_replaced
            )
        
        # IMPROVED: Report missing segments
        for segment_id in segment_map.keys():
            if segment_id not in segments_replaced:
                segments_not_found.append(segment_id)
        
        if segments_not_found:
            logger.warning(
                f"⚠ {len(segments_not_found)} segments not found in document: "
                f"{segments_not_found[:5]}..."
            )
        
        logger.info(
            f"✓ Replaced {len(segments_replaced)}/{len(segment_map)} segments "
            f"({len(segments_replaced)/len(segment_map)*100:.1f}%)"
        )
    
    def _replace_text_preserve_formatting(self, run, new_text: str) -> None:
        """
        Replace text in run while preserving ALL font properties.
        
        IMPROVED: Saves extended font attributes including advanced typography.
        
        Args:
            run: Run object to modify
            new_text: New text to set
        """
        font = run.font
        
        # IMPROVED: Comprehensive font backup with extended properties
        font_backup = {
            # Basic properties
            'name': font.name,
            'size': font.size,
            'bold': font.bold,
            'italic': font.italic,
            'underline': font.underline,
            
            # Extended properties
            'strike': font.strike,
            'double_strike': getattr(font, 'double_strike', None),
            'outline': getattr(font, 'outline', None),
            'shadow': getattr(font, 'shadow', None),
            'emboss': getattr(font, 'emboss', None),
            'imprint': getattr(font, 'imprint', None),
            
            # Color properties
            'color_rgb': font.color.rgb if (font.color and hasattr(font.color, 'rgb')) else None,
            'color_theme': getattr(font.color, 'theme_color', None) if font.color else None,
            'highlight_color': font.highlight_color,
            
            # Position effects
            'subscript': font.subscript,
            'superscript': font.superscript,
            
            # Typography
            'all_caps': getattr(font, 'all_caps', None),
            'small_caps': font.small_caps,
            'hidden': getattr(font, 'hidden', None),
            
            # Spacing
            'spacing': getattr(font, 'spacing', None),
            'kerning': getattr(font, 'kerning', None),
        }
        
        # Replace text
        run.text = new_text
        
        # IMPROVED: Restore ALL properties with better error handling
        for prop, value in font_backup.items():
            if value is None:
                continue
                
            try:
                if prop == 'color_rgb' and value:
                    font.color.rgb = value
                elif prop == 'color_theme' and value:
                    font.color.theme_color = value
                elif prop in ['spacing', 'kerning'] and value:
                    setattr(font, prop, value)
                else:
                    setattr(font, prop, value)
            except (AttributeError, TypeError, ValueError) as e:
                # Some properties may be read-only or version-specific
                logger.debug(f"Could not restore font property '{prop}': {e}")
    
    def _replace_cell_text(self, paragraph, new_text: str) -> None:
        """
        Replace text in table cell paragraph while preserving multi-run formatting.
        
        IMPROVED: Handles cells with multiple differently-formatted runs intelligently.
        
        Args:
            paragraph: Paragraph in cell
            new_text: New text
        """
        if not paragraph.runs:
            # No existing runs - just add text
            paragraph.add_run(new_text)
            return
        
        # STRATEGY 1: Single run - simple replacement
        if len(paragraph.runs) == 1:
            self._replace_text_preserve_formatting(paragraph.runs[0], new_text)
            return
        
        # STRATEGY 2: Multiple runs - distribute text proportionally
        # This preserves the intent of having different formatting for different parts
        
        # Calculate original text distribution
        original_texts = [run.text for run in paragraph.runs]
        original_lengths = [len(t) for t in original_texts]
        total_original = sum(original_lengths)
        
        if total_original == 0:
            # Fallback: all runs are empty, use first run
            self._replace_text_preserve_formatting(paragraph.runs[0], new_text)
            for run in paragraph.runs[1:]:
                run.text = ''
            return
        
        # Distribute new text proportionally to preserve formatting boundaries
        new_length = len(new_text)
        char_index = 0
        
        for i, run in enumerate(paragraph.runs):
            if i == len(paragraph.runs) - 1:
                # Last run gets all remaining text
                run_text = new_text[char_index:]
            else:
                # Calculate proportional share
                proportion = original_lengths[i] / total_original
                chars_for_run = int(new_length * proportion)
                
                # Ensure we don't go out of bounds
                chars_for_run = min(chars_for_run, new_length - char_index)
                
                run_text = new_text[char_index:char_index + chars_for_run]
                char_index += chars_for_run
            
            # Replace text while preserving formatting
            self._replace_text_preserve_formatting(run, run_text)
        
        logger.debug(f"Distributed text across {len(paragraph.runs)} runs proportionally")
    
    def _replace_header_footer_text(
        self,
        header_footer,
        segment_map: Dict[str, TextSegment],
        prefix: str,
        section_idx: int,
        segments_replaced: Set[str]
    ) -> None:
        """
        Replace text in headers/footers with better multi-run handling.
        
        IMPROVED: Uses smart cell replacement logic for multi-run preservation.
        FIXED: Added section_idx to prevent ID collisions.
        
        Args:
            header_footer: Header or footer object
            segment_map: Map of segment IDs to segments
            prefix: "header" or "footer"
            section_idx: Section index for unique IDs
            segments_replaced: Set to track replaced segments
        """
        for para_idx, paragraph in enumerate(header_footer.paragraphs):
            segment_id = f"section_{section_idx}_{prefix}_{para_idx}"
            
            if segment_id not in segment_map:
                continue
            
            segment = segment_map[segment_id]
            
            if not paragraph.runs:
                # No runs - add new one
                paragraph.add_run(segment.text)
                segments_replaced.add(segment_id)
                continue
            
            # IMPROVED: Use smart multi-run replacement
            self._replace_cell_text(paragraph, segment.text)
            segments_replaced.add(segment_id)
            logger.debug(f"✓ Replaced {segment_id}")
    
    def _build_segment_map(
        self,
        segments: List[TextSegment]
    ) -> Dict[str, TextSegment]:
        """
        Build lookup map of segments by ID.
        
        IMPROVED: Better duplicate detection and reporting.
        """
        segment_map = {}
        duplicates = []
        
        for segment in segments:
            if segment.id in segment_map:
                duplicates.append(segment.id)
                logger.warning(f"Duplicate segment ID: {segment.id}")
            segment_map[segment.id] = segment
        
        if duplicates:
            logger.warning(
                f"⚠ Found {len(duplicates)} duplicate segment IDs. "
                f"Later segments will overwrite earlier ones."
            )
        
        return segment_map
    
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
        
        IMPROVED: More comprehensive validation with detailed reporting.
        
        Returns:
            True if validation passed (no critical errors)
        """
        try:
            original_doc = DocxDocument(original_path)
            output_doc = DocxDocument(output_path)
            
            # IMPROVED: Structured validation results
            validation_results = {
                'critical_errors': [],
                'warnings': [],
                'checks_passed': 0,
                'checks_total': 0
            }
            
            # CHECK 1: Paragraph count (CRITICAL)
            validation_results['checks_total'] += 1
            if len(original_doc.paragraphs) != len(output_doc.paragraphs):
                validation_results['critical_errors'].append(
                    f"Paragraph count: {len(original_doc.paragraphs)} → {len(output_doc.paragraphs)}"
                )
            else:
                validation_results['checks_passed'] += 1
            
            # CHECK 2: Table count (CRITICAL)
            validation_results['checks_total'] += 1
            if len(original_doc.tables) != len(output_doc.tables):
                validation_results['critical_errors'].append(
                    f"Table count: {len(original_doc.tables)} → {len(output_doc.tables)}"
                )
            else:
                validation_results['checks_passed'] += 1
            
            # CHECK 3: Section count
            validation_results['checks_total'] += 1
            if len(original_doc.sections) != len(output_doc.sections):
                validation_results['warnings'].append(
                    f"Section count: {len(original_doc.sections)} → {len(output_doc.sections)}"
                )
            else:
                validation_results['checks_passed'] += 1
            
            # CHECK 4: Sample paragraph formatting (first 10 paragraphs)
            sample_size = min(10, len(original_doc.paragraphs))
            for i in range(sample_size):
                validation_results['checks_total'] += 1
                
                orig_para = original_doc.paragraphs[i]
                out_para = output_doc.paragraphs[i]
                
                formatting_match = True
                
                # Alignment
                if orig_para.alignment != out_para.alignment:
                    validation_results['warnings'].append(
                        f"Para {i}: alignment {orig_para.alignment} → {out_para.alignment}"
                    )
                    formatting_match = False
                
                # Style
                orig_style = orig_para.style.name if orig_para.style else None
                out_style = out_para.style.name if out_para.style else None
                if orig_style != out_style:
                    validation_results['warnings'].append(
                        f"Para {i}: style '{orig_style}' → '{out_style}'"
                    )
                    formatting_match = False
                
                # Run formatting (first 3 runs)
                min_runs = min(3, len(orig_para.runs), len(out_para.runs))
                for j in range(min_runs):
                    orig_run = orig_para.runs[j]
                    out_run = out_para.runs[j]
                    
                    # Font name
                    if orig_run.font.name != out_run.font.name:
                        validation_results['warnings'].append(
                            f"Para {i} Run {j}: font '{orig_run.font.name}' → '{out_run.font.name}'"
                        )
                        formatting_match = False
                    
                    # Font size
                    if orig_run.font.size != out_run.font.size:
                        validation_results['warnings'].append(
                            f"Para {i} Run {j}: size {orig_run.font.size} → {out_run.font.size}"
                        )
                        formatting_match = False
                    
                    # Bold/Italic
                    if orig_run.font.bold != out_run.font.bold:
                        validation_results['warnings'].append(
                            f"Para {i} Run {j}: bold changed"
                        )
                        formatting_match = False
                    
                    if orig_run.font.italic != out_run.font.italic:
                        validation_results['warnings'].append(
                            f"Para {i} Run {j}: italic changed"
                        )
                        formatting_match = False
                
                if formatting_match:
                    validation_results['checks_passed'] += 1
            
            # CHECK 5: Table structure
            for table_idx, (orig_table, out_table) in enumerate(
                zip(original_doc.tables, output_doc.tables)
            ):
                validation_results['checks_total'] += 1
                
                table_match = True
                
                if len(orig_table.rows) != len(out_table.rows):
                    validation_results['critical_errors'].append(
                        f"Table {table_idx}: row count {len(orig_table.rows)} → {len(out_table.rows)}"
                    )
                    table_match = False
                
                if len(orig_table.columns) != len(out_table.columns):
                    validation_results['critical_errors'].append(
                        f"Table {table_idx}: column count {len(orig_table.columns)} → {len(out_table.columns)}"
                    )
                    table_match = False
                
                if table_match:
                    validation_results['checks_passed'] += 1
            
            # Generate report
            has_critical_errors = len(validation_results['critical_errors']) > 0
            
            if has_critical_errors:
                logger.error("❌ CRITICAL FORMATTING ERRORS:")
                for error in validation_results['critical_errors']:
                    logger.error(f"  • {error}")
            
            if validation_results['warnings']:
                logger.warning(f"⚠ {len(validation_results['warnings'])} formatting warnings:")
                # Show first 10 warnings
                for warning in validation_results['warnings'][:10]:
                    logger.warning(f"  • {warning}")
                if len(validation_results['warnings']) > 10:
                    logger.warning(f"  ... and {len(validation_results['warnings']) - 10} more warnings")
            
            logger.info(
                f"✓ Validation: {validation_results['checks_passed']}/{validation_results['checks_total']} checks passed "
                f"({validation_results['checks_passed']/validation_results['checks_total']*100:.1f}%)"
            )
            
            # Return True if no critical errors (warnings are acceptable)
            return not has_critical_errors
            
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
            logger.info(f"✓ Output document validated: {output_path.name}")
            return True
        except Exception as e:
            logger.error(f"Invalid output document: {e}")
            return False


class FormattingError(Exception):
    """Exception raised when formatting fails."""
    pass


# Compatibility alias for existing code
DocxFormatter = EnhancedDocxFormatter
