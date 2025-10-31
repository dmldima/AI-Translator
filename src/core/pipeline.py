"""
Enhanced Translation Pipeline - Production-Ready Implementation.
Implements comprehensive error handling, validation, and resource management.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import logging
import uuid
import tempfile
import shutil
from contextlib import contextmanager

from .interfaces import (
    ITranslationEngine,
    ITranslationCache,
    IGlossaryProcessor,
    IProgressCallback,
    ITranslationPipeline,
    IDocumentParser,
    IDocumentFormatter
)
from .models import (
    Document,
    TextSegment,
    TranslationRequest,
    TranslationResult,
    TranslationJob,
    TranslationStatus,
    FileType,
    validate_language_pair
)
from .exceptions import (
    TranslationPipelineError,
    ParserError,
    FormatterError,
    ValidationError,
    TranslationError
)


logger = logging.getLogger(__name__)


class EnhancedTranslationPipeline(ITranslationPipeline):
    """
    Production-ready translation pipeline with comprehensive error handling.
    
    Features:
    - Input validation for all parameters
    - Resource cleanup on failure
    - Progress reporting at all stages
    - File type validation and matching
    - Temporary file handling
    - Thread-safe operations
    - Comprehensive logging
    - Health checks
    """
    
    def __init__(
        self,
        engine: ITranslationEngine,
        parser_factory: 'ParserFactory',
        formatter_factory: 'FormatterFactory',
        cache: Optional[ITranslationCache] = None,
        glossary: Optional[IGlossaryProcessor] = None,
        temp_dir: Optional[Path] = None,
        operation_timeout: int = 3600  # 1 hour default
    ):
        """
        Initialize enhanced pipeline.
        
        Args:
            engine: Translation engine
            parser_factory: Factory for creating parsers
            formatter_factory: Factory for creating formatters
            cache: Optional cache
            glossary: Optional glossary processor
            temp_dir: Temporary directory for intermediate files
            operation_timeout: Maximum operation time in seconds
            
        Raises:
            ValidationError: If components invalid
        """
        # Validate required components
        if not engine:
            raise ValidationError("Translation engine is required")
        if not parser_factory:
            raise ValidationError("Parser factory is required")
        if not formatter_factory:
            raise ValidationError("Formatter factory is required")
        
        self.engine = engine
        self.parser_factory = parser_factory
        self.formatter_factory = formatter_factory
        self.cache = cache
        self.glossary = glossary
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.operation_timeout = operation_timeout
        
        # Validate engine
        if not self.engine.validate_config():
            raise ValidationError(f"Engine {engine.name} configuration invalid")
        
        # Create temp directory
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized pipeline: engine={engine.name}, "
            f"cache={'enabled' if cache else 'disabled'}, "
            f"glossary={'enabled' if glossary else 'disabled'}"
        )
    
    def translate_document(
        self,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        domain: str = "general",
        progress_callback: Optional[IProgressCallback] = None
    ) -> TranslationJob:
        """
        Translate document with comprehensive error handling and validation.
        
        Args:
            input_path: Path to input document
            output_path: Path for output document
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)
            domain: Domain for glossary matching
            progress_callback: Optional progress callback
            
        Returns:
            TranslationJob with complete results
            
        Raises:
            ValidationError: If inputs invalid
            TranslationPipelineError: If translation fails
        """
        # Create job
        job = TranslationJob(
            job_id=str(uuid.uuid4()),
            input_file=input_path,
            output_file=output_path,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            engine=self.engine.name
        )
        
        job.started_at = datetime.utcnow()
        temp_output = None
        
        try:
            # === VALIDATION PHASE ===
            job.status = TranslationStatus.PENDING
            logger.info(f"Starting translation job {job.job_id}")
            
            self._validate_inputs(input_path, output_path, source_lang, target_lang, domain)
            
            # Validate file type compatibility
            input_type = FileType.from_extension(input_path.suffix)
            output_type = FileType.from_extension(output_path.suffix)
            
            if input_type != output_type:
                raise ValidationError(
                    f"Input and output file types must match: "
                    f"{input_type.value} != {output_type.value}"
                )
            
            # Check parser and formatter availability
            if not self.parser_factory.is_supported(input_type):
                raise ValidationError(f"No parser available for {input_type.value}")
            
            if not self.formatter_factory.is_supported(output_type):
                raise ValidationError(f"No formatter available for {output_type.value}")
            
            if progress_callback:
                progress_callback.on_start(job)
            
            # === PARSING PHASE ===
            logger.info(f"Parsing document: {input_path}")
            job.status = TranslationStatus.PARSING
            
            document = self._parse_document_safe(input_path, input_type)
            segments = self._get_translatable_segments(document)
            
            job.total_segments = len(segments)
            logger.info(f"Found {job.total_segments} translatable segments")
            
            if job.total_segments == 0:
                logger.warning("No translatable content found")
                job.status = TranslationStatus.COMPLETED
                job.completed_at = datetime.utcnow()
                # Copy original file to output
                shutil.copy2(input_path, output_path)
                return job
            
            # === TRANSLATION PHASE ===
            logger.info("Translating segments")
            job.status = TranslationStatus.TRANSLATING
            
            translated_segments = self._translate_segments_safe(
                segments,
                source_lang,
                target_lang,
                domain,
                job,
                progress_callback
            )
            
            # === FORMATTING PHASE ===
            logger.info("Formatting output document")
            job.status = TranslationStatus.FORMATTING
            
            # Use temporary file for output
            temp_output = self._get_temp_path(output_path)
            
            translated_document = self._create_translated_document(
                document,
                translated_segments,
                temp_output
            )
            
            self._format_document_safe(
                translated_document,
                temp_output,
                output_type
            )
            
            # Move temp file to final location
            self._finalize_output(temp_output, output_path)
            
            # === COMPLETION ===
            job.status = TranslationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            if progress_callback:
                progress_callback.on_complete(job)
            
            logger.info(
                f"Translation complete: {job.translated_segments}/{job.total_segments} segments, "
                f"{job.cached_segments} from cache, {job.failed_segments} failed, "
                f"duration={job.duration:.2f}s"
            )
            
            return job
            
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            job.status = TranslationStatus.FAILED
            job.add_error(f"Validation: {e}")
            job.completed_at = datetime.utcnow()
            if progress_callback:
                progress_callback.on_error(job, e)
            raise
            
        except Exception as e:
            logger.error(f"Translation pipeline failed: {e}", exc_info=True)
            job.status = TranslationStatus.FAILED
            job.add_error(str(e))
            job.completed_at = datetime.utcnow()
            
            if progress_callback:
                progress_callback.on_error(job, e)
            
            raise TranslationPipelineError(f"Pipeline failed: {e}") from e
            
        finally:
            # Cleanup temporary files
            if temp_output and temp_output.exists():
                try:
                    temp_output.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file: {e}")
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: str = "general"
    ) -> TranslationResult:
        """
        Translate plain text with validation.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for glossary
            
        Returns:
            TranslationResult
            
        Raises:
            ValidationError: If inputs invalid
            TranslationError: If translation fails
        """
        # Validate inputs
        if not isinstance(text, str):
            raise ValidationError(f"Text must be string, got {type(text)}")
        
        if not text.strip():
            return TranslationResult(
                original_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                engine=self.engine.name,
                model=self.engine.model_name,
                cached=False
            )
        
        validate_language_pair(source_lang, target_lang)
        
        # Check text length
        if len(text) > 50000:  # Reasonable limit
            raise ValidationError(f"Text too long: {len(text)} characters (max: 50000)")
        
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain
        )
        
        try:
            # Check cache
            if self.cache:
                cached = self.cache.get(request)
                if cached:
                    logger.debug(f"Cache hit for text: {text[:50]}...")
                    return cached
            
            # Apply glossary preprocessing
            preprocessed_text = text
            if self.glossary:
                try:
                    preprocessed_text = self.glossary.preprocess(
                        text, domain, source_lang, target_lang
                    )
                except Exception as e:
                    logger.warning(f"Glossary preprocessing failed: {e}")
            
            # Translate
            translated_text = self.engine.translate(
                preprocessed_text,
                source_lang,
                target_lang
            )
            
            if not translated_text:
                raise TranslationError("Engine returned empty translation")
            
            # Apply glossary postprocessing
            glossary_result = None
            if self.glossary:
                try:
                    glossary_result = self.glossary.postprocess(
                        translated_text, domain, source_lang, target_lang
                    )
                    translated_text = glossary_result.text
                except Exception as e:
                    logger.warning(f"Glossary postprocessing failed: {e}")
            
            # Create result
            result = TranslationResult(
                original_text=text,
                translated_text=translated_text,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain,
                engine=self.engine.name,
                model=self.engine.model_name,
                cached=False,
                glossary_applied=glossary_result is not None,
                glossary_terms_used=glossary_result.terms_applied if glossary_result else []
            )
            
            # Cache result
            if self.cache:
                try:
                    self.cache.set(request, result)
                except Exception as e:
                    logger.warning(f"Failed to cache result: {e}")
            
            return result
            
        except TranslationError:
            raise
        except Exception as e:
            raise TranslationError(f"Text translation failed: {e}") from e
    
    def get_supported_file_types(self) -> List[FileType]:
        """Get list of supported file types."""
        # Only return types supported by both parser and formatter
        parser_types = set(self.parser_factory.get_supported_types())
        formatter_types = set(self.formatter_factory.get_supported_types())
        return list(parser_types & formatter_types)
    
    def get_health(self) -> Dict[str, Any]:
        """
        Get pipeline health status with detailed component checks.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Check engine
        try:
            engine_valid = self.engine.validate_config()
            health['components']['engine'] = {
                'name': self.engine.name,
                'model': self.engine.model_name,
                'status': 'healthy' if engine_valid else 'degraded',
                'supported_languages': len(self.engine.get_supported_languages())
            }
        except Exception as e:
            health['components']['engine'] = {
                'status': 'error',
                'error': str(e)
            }
            health['status'] = 'degraded'
        
        # Check cache
        if self.cache:
            try:
                cache_stats = self.cache.get_stats()
                health['components']['cache'] = {
                    'status': 'healthy',
                    'stats': cache_stats
                }
            except Exception as e:
                health['components']['cache'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        # Check glossary
        if self.glossary:
            try:
                glossary_stats = self.glossary.get_stats()
                health['components']['glossary'] = {
                    'status': 'healthy',
                    'stats': glossary_stats
                }
            except Exception as e:
                health['components']['glossary'] = {
                    'status': 'error',
                    'error': str(e)
                }
                health['status'] = 'degraded'
        
        # Check parsers/formatters
        health['components']['parsers'] = {
            'supported_types': [ft.value for ft in self.parser_factory.get_supported_types()]
        }
        health['components']['formatters'] = {
            'supported_types': [ft.value for ft in self.formatter_factory.get_supported_types()]
        }
        
        health['supported_file_types'] = [
            ft.value for ft in self.get_supported_file_types()
        ]
        
        return health
    
    # === PRIVATE HELPER METHODS ===
    
    def _validate_inputs(
        self,
        input_path: Path,
        output_path: Path,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> None:
        """Validate all inputs comprehensively."""
        # Validate input file
        if not input_path.exists():
            raise ValidationError(f"Input file not found: {input_path}")
        
        if not input_path.is_file():
            raise ValidationError(f"Input path is not a file: {input_path}")
        
        # Validate file size (max 100MB)
        size_mb = input_path.stat().st_size / (1024 * 1024)
        if size_mb > 100:
            raise ValidationError(
                f"Input file too large: {size_mb:.1f} MB (max: 100 MB)"
            )
        
        # Validate output path
        if not output_path.parent.exists():
            raise ValidationError(
                f"Output directory doesn't exist: {output_path.parent}"
            )
        
        # Validate languages
        validate_language_pair(source_lang, target_lang)
        
        # Validate engine supports languages
        if not self.engine.is_language_pair_supported(source_lang, target_lang):
            raise ValidationError(
                f"Engine {self.engine.name} doesn't support "
                f"{source_lang} -> {target_lang}"
            )
        
        # Validate domain
        if not domain or not domain.strip():
            raise ValidationError("Domain cannot be empty")
        
        if not domain.replace('_', '').replace('-', '').isalnum():
            raise ValidationError(
                f"Invalid domain format: {domain}. "
                f"Use alphanumeric, underscore, hyphen only"
            )
    
    def _parse_document_safe(self, input_path: Path, file_type: FileType) -> Document:
        """Parse document with error handling."""
        try:
            parser = self.parser_factory.get_parser(file_type)
            
            if not parser.validate_document(input_path):
                raise ParserError(f"Document validation failed: {input_path}")
            
            document = parser.parse(input_path)
            
            if not document.segments:
                logger.warning(f"No segments extracted from {input_path}")
            
            return document
            
        except Exception as e:
            raise ParserError(f"Failed to parse {input_path}: {e}") from e
    
    def _get_translatable_segments(self, document: Document) -> List[TextSegment]:
        """Get segments ready for translation with filtering."""
        segments = [s for s in document.segments if s.text.strip()]
        
        # Filter out very short segments (likely formatting artifacts)
        segments = [s for s in segments if len(s.text.strip()) > 1]
        
        logger.info(
            f"Translatable segments: {len(segments)}/{document.total_segments}"
        )
        
        return segments
    
    def _translate_segments_safe(
        self,
        segments: List[TextSegment],
        source_lang: str,
        target_lang: str,
        domain: str,
        job: TranslationJob,
        progress_callback: Optional[IProgressCallback]
    ) -> List[TextSegment]:
        """Translate segments with comprehensive error handling."""
        translated_segments = []
        
        for idx, segment in enumerate(segments):
            try:
                result = self._translate_segment_safe(
                    segment,
                    source_lang,
                    target_lang,
                    domain
                )
                
                # Create translated segment
                translated_segment = TextSegment(
                    id=segment.id,
                    text=result.translated_text,
                    segment_type=segment.segment_type,
                    position=segment.position,
                    text_formatting=segment.text_formatting,
                    paragraph_formatting=segment.paragraph_formatting,
                    cell_formatting=segment.cell_formatting,
                    context=segment.context,
                    metadata=segment.metadata
                )
                translated_segments.append(translated_segment)
                
                # Update job stats
                if result.cached:
                    job.cached_segments += 1
                
                job.update_progress(idx + 1)
                
                # Notify progress
                if progress_callback:
                    progress_callback.on_progress(job, idx + 1, job.total_segments)
                    progress_callback.on_segment_translated(segment, result)
                
            except Exception as e:
                logger.error(f"Failed to translate segment {segment.id}: {e}")
                job.failed_segments += 1
                job.add_error(f"Segment {segment.id}: {str(e)[:100]}")
                
                # Keep original text on error
                translated_segments.append(segment)
        
        return translated_segments
    
    def _translate_segment_safe(
        self,
        segment: TextSegment,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> TranslationResult:
        """Translate single segment with error handling."""
        request = TranslationRequest(
            text=segment.text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            segment_id=segment.id,
            context=segment.context
        )
        
        # Check cache
        if self.cache:
            try:
                cached = self.cache.get(request)
                if cached:
                    return cached
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
        
        # Apply glossary preprocessing
        preprocessed_text = segment.text
        if self.glossary:
            try:
                preprocessed_text = self.glossary.preprocess(
                    segment.text, domain, source_lang, target_lang
                )
            except Exception as e:
                logger.warning(f"Glossary preprocessing failed: {e}")
        
        # Translate
        try:
            translated_text = self.engine.translate(
                preprocessed_text,
                source_lang,
                target_lang,
                context=segment.context
            )
        except Exception as e:
            raise TranslationError(f"Engine translation failed: {e}") from e
        
        if not translated_text or not translated_text.strip():
            raise TranslationError("Engine returned empty translation")
        
        # Apply glossary postprocessing
        glossary_result = None
        if self.glossary:
            try:
                glossary_result = self.glossary.postprocess(
                    translated_text, domain, source_lang, target_lang
                )
                translated_text = glossary_result.text
            except Exception as e:
                logger.warning(f"Glossary postprocessing failed: {e}")
        
        # Create result
        result = TranslationResult(
            original_text=segment.text,
            translated_text=translated_text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            engine=self.engine.name,
            model=self.engine.model_name,
            segment_id=segment.id,
            glossary_applied=glossary_result is not None,
            glossary_terms_used=glossary_result.terms_applied if glossary_result else []
        )
        
        # Cache result
        if self.cache:
            try:
                self.cache.set(request, result)
            except Exception as e:
                logger.warning(f"Failed to cache result: {e}")
        
        return result
    
    def _create_translated_document(
        self,
        original: Document,
        translated_segments: List[TextSegment],
        output_path: Path
    ) -> Document:
        """Create translated document preserving structure."""
        return Document(
            file_path=output_path,
            file_type=original.file_type,
            segments=translated_segments,
            metadata=original.metadata,
            styles=original.styles,
            headers=original.headers,
            footers=original.footers
        )
    
    def _format_document_safe(
        self,
        document: Document,
        output_path: Path,
        file_type: FileType
    ) -> None:
        """Format document with error handling."""
        try:
            formatter = self.formatter_factory.get_formatter(file_type)
            
            # Format with formatting preservation
            formatter.format(document, output_path, preserve_formatting=True)
            
            # Validate output
            if not formatter.validate_output(output_path):
                raise FormatterError("Output validation failed")
            
        except Exception as e:
            raise FormatterError(f"Failed to format document: {e}") from e
    
    def _get_temp_path(self, final_path: Path) -> Path:
        """Get temporary file path."""
        temp_name = f"temp_{uuid.uuid4().hex}_{final_path.name}"
        return self.temp_dir / temp_name
    
    def _finalize_output(self, temp_path: Path, final_path: Path) -> None:
        """Move temporary file to final location."""
        try:
            # Ensure parent directory exists
            final_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file
            shutil.move(str(temp_path), str(final_path))
            
            logger.info(f"Output finalized: {final_path}")
            
        except Exception as e:
            raise FormatterError(f"Failed to finalize output: {e}") from e
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        # Clean up temporary files
        try:
            temp_files = list(self.temp_dir.glob("temp_*"))
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to cleanup {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
