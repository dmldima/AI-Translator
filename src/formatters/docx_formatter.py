"""
Enhanced DOCX Document Formatter - FIXED: Path Traversal + Memory Management + Atomic Saves
Implements best practices for maintaining all document styling.
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
    FIXED: Path traversal protection, memory management, atomic saves.
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
        temp_path = None
        
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
            
            # Atomic save using temp file
            self._save_atomic(doc, output_path)
            
            # Validate formatting preservation
            if preserve_formatting:
                self._validate_formatting(document.file_path, output_path)
            
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
            # Cleanup temp file if exists
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            
            # Help garbage collector
            del doc
    
    def _save_atomic(self, doc, output_path: Path) -> None:
        """
        Atomic save using temp file to prevent corruption.
        
        Args:
            doc: Document to save
            output_path: Target path
            
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
                    self._replace_text_preserve_formatting(run, segment.text)
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
                            self._replace_cell_text(cell.paragraphs[0], segment.text)
                        
                        logger.debug(f"Replaced text in {segment_id}")
        
        # Process headers/footers
        for section_idx, section in enumerate(docx.sections):
            self._replace_header_footer_text(
                section.header, 
                segment_map, 
                "header",
                section_idx
            )
            self._replace_header_footer_text(
                section.footer, 
                segment_map, 
                "footer",
                section_idx
            )
    
    def _replace_text_preserve_formatting(self, run, new_text: str) -> None:
        """
        Replace text in run while preserving ALL font properties.
        
        Args:
            run: Run object to modify
            new_text: New text to set
        """
        # Save ALL font properties
        font_backup = {
            'name': run.font.name,
            'size': run.font.size,
            'bold': run.font.bold,
            'italic': run.font.italic,
            'underline': run.font.underline,
            'strike': run.font.strike,
            'color': run.font.color.rgb if (run.font.color and hasattr(run.font.color, 'rgb')) else None,
            'highlight_color': run.font.highlight_color,
            'subscript': run.font.subscript,
            'superscript': run.font.superscript,
            'small_caps': run.font.small_caps,
        }
        
        # Replace text
        run.text = new_text
        
        # Restore ALL properties
        for prop, value in font_backup.items():
            if value is not None:
                try:
                    if prop == 'color' and value:
                        run.font.color.rgb = value
                    else:
                        setattr(run.font, prop, value)
                except (AttributeError, TypeError):
                    # Some properties may be read-only
                    pass
    
    def _replace_cell_text(self, paragraph, new_text: str) -> None:
        """
        Replace text in table cell paragraph.
        
        Args:
            paragraph: Paragraph in cell
            new_text: New text
        """
        if paragraph.runs:
            # Save formatting from first run
            template_run = paragraph.runs[0]
            
            # Remove all runs via XML (cleaner than setting text to '')
            for _ in range(len(paragraph.runs)):
                paragraph._element.remove(paragraph.runs[0]._element)
            
            # Create new run with template formatting
            new_run = paragraph.add_run(new_text)
            
            # Copy formatting
            if template_run.font.name:
                new_run.font.name = template_run.font.name
            if template_run.font.size:
                new_run.font.size = template_run.font.size
            if template_run.font.bold is not None:
                new_run.font.bold = template_run.font.bold
            if template_run.font.italic is not None:
                new_run.font.italic = template_run.font.italic
            if template_run.font.color and hasattr(template_run.font.color, 'rgb'):
                try:
                    new_run.font.color.rgb = template_run.font.color.rgb
                except:
                    pass
        else:
            # No existing runs - just add text
            paragraph.add_run(new_text)
    
    def _replace_header_footer_text(
        self,
        header_footer,
        segment_map: Dict[str, TextSegment],
        prefix: str,
        section_idx: int
    ) -> None:
        """
        Replace text in headers/footers.
        
        FIXED: Added section_idx to prevent ID collisions.
        """
        for para_idx, paragraph in enumerate(header_footer.paragraphs):
            segment_id = f"section_{section_idx}_{prefix}_{para_idx}"
            
            if segment_id in segment_map:
                segment = segment_map[segment_id]
                
                # Preserve formatting from first run
                if paragraph.runs:
                    self._replace_text_preserve_formatting(
                        paragraph.runs[0],
                        segment.text
                    )
                    # Clear other runs
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
        segment_map = {}
        duplicates = []
        
        for segment in segments:
            if segment.id in segment_map:
                duplicates.append(segment.id)
            segment_map[segment.id] = segment
        
        if duplicates:
            logger.warning(f"Duplicate segment IDs found: {len(duplicates)}")
        
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
        
        Returns:
            True if validation passed (warnings don't fail)
        """
        try:
            original_doc = DocxDocument(original_path)
            output_doc = DocxDocument(output_path)
            
            # Compare structure (critical checks)
            if len(original_doc.paragraphs) != len(output_doc.paragraphs):
                logger.error(
                    f"CRITICAL: Paragraph count mismatch: "
                    f"{len(original_doc.paragraphs)} → {len(output_doc.paragraphs)}"
                )
                return False
            
            if len(original_doc.tables) != len(output_doc.tables):
                logger.error(
                    f"CRITICAL: Table count mismatch: "
                    f"{len(original_doc.tables)} → {len(output_doc.tables)}"
                )
                return False
            
            # Sample formatting checks (warnings only)
            warnings = 0
            
            for i, (orig_para, out_para) in enumerate(
                zip(original_doc.paragraphs[:5], output_doc.paragraphs[:5])
            ):
                # Check paragraph alignment
                if orig_para.alignment != out_para.alignment:
                    logger.warning(f"Para {i}: alignment changed")
                    warnings += 1
                
                # Check run count (may legitimately change)
                if len(orig_para.runs) != len(out_para.runs):
                    logger.debug(
                        f"Para {i}: run count changed "
                        f"({len(orig_para.runs)} → {len(out_para.runs)})"
                    )
                
                # Check formatting of common runs
                min_runs = min(len(orig_para.runs), len(out_para.runs))
                for j in range(min_runs):
                    orig_run = orig_para.runs[j]
                    out_run = out_para.runs[j]
                    
                    if orig_run.font.name != out_run.font.name:
                        logger.warning(f"Para {i} Run {j}: font name changed")
                        warnings += 1
                    
                    if orig_run.font.size != out_run.font.size:
                        logger.warning(f"Para {i} Run {j}: font size changed")
                        warnings += 1
            
            if warnings > 0:
                logger.warning(f"⚠ Formatting validation: {warnings} warnings")
            else:
                logger.info("✓ Formatting validation: perfect match")
            
            # Warnings don't fail validation
            return True
            
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
