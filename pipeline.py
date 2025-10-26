"""
Translation Pipeline - Main orchestrator.
Combines parser, engine, cache, glossary, and formatter.
"""
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import logging
import uuid

from .interfaces import (
    ITranslationEngine,
    ITranslationCache,
    IGlossaryProcessor,
    IProgressCallback,
    ITranslationPipeline
)
from .models import (
    Document,
    TextSegment,
    TranslationRequest,
    TranslationResult,
    TranslationJob,
    TranslationStatus,
    FileType
)
from ..parsers.docx_parser import DocxParser
from ..formatters.docx_formatter import DocxFormatter


logger = logging.getLogger(__name__)


class TranslationPipeline(ITranslationPipeline):
    """
    Main translation pipeline.
    Orchestrates all components to translate documents.
    """
    
    def __init__(
        self,
        engine: ITranslationEngine,
        cache: Optional[ITranslationCache] = None,
        glossary: Optional[IGlossaryProcessor] = None
    ):
        """
        Initialize pipeline.
        
        Args:
            engine: Translation engine
            cache: Optional cache
            glossary: Optional glossary processor
        """
        self.engine = engine
        self.cache = cache
        self.glossary = glossary
        
        # Parsers and formatters
        self.parsers = {
            FileType.DOCX: DocxParser()
        }
        self.formatters = {
            FileType.DOCX: DocxFormatter()
        }
        
        logger.info(f"Initialized pipeline with engine: {engine.name}")
    
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
        Translate entire document.
        
        Args:
            input_path: Path to input document
            output_path: Path for output document
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for glossary
            progress_callback: Optional progress callback
            
        Returns:
            TranslationJob with results
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
        
        try:
            job.status = TranslationStatus.PENDING
            job.started_at = datetime.utcnow()
            
            if progress_callback:
                progress_callback.on_start(job)
            
            # Step 1: Parse document
            logger.info(f"Parsing document: {input_path}")
            job.status = TranslationStatus.PARSING
            
            document = self._parse_document(input_path)
            segments = self._get_translatable_segments(document)
            
            job.total_segments = len(segments)
            logger.info(f"Found {job.total_segments} translatable segments")
            
            # Step 2: Translate segments
            logger.info(f"Translating segments")
            job.status = TranslationStatus.TRANSLATING
            
            translated_segments = []
            for idx, segment in enumerate(segments):
                try:
                    result = self._translate_segment(
                        segment,
                        source_lang,
                        target_lang,
                        domain
                    )
                    
                    # Update segment with translation
                    translated_segment = TextSegment(
                        id=segment.id,
                        text=result.translated_text,
                        segment_type=segment.segment_type,
                        position=segment.position,
                        text_formatting=segment.text_formatting,
                        paragraph_formatting=segment.paragraph_formatting,
                        cell_formatting=segment.cell_formatting
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
                    job.add_error(f"Segment {segment.id}: {e}")
                    
                    # Keep original text on error
                    translated_segments.append(segment)
            
            # Step 3: Format output document
            logger.info(f"Formatting output document")
            job.status = TranslationStatus.FORMATTING
            
            translated_document = Document(
                file_path=output_path,
                file_type=document.file_type,
                segments=translated_segments,
                metadata=document.metadata,
                styles=document.styles,
                headers=document.headers,
                footers=document.footers
            )
            
            self._format_document(translated_document, output_path)
            
            # Step 4: Complete
            job.status = TranslationStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            if progress_callback:
                progress_callback.on_complete(job)
            
            logger.info(
                f"Translation complete: {job.translated_segments}/{job.total_segments} segments, "
                f"{job.cached_segments} from cache, {job.failed_segments} failed"
            )
            
            return job
            
        except Exception as e:
            logger.error(f"Translation pipeline failed: {e}")
            job.status = TranslationStatus.FAILED
            job.add_error(str(e))
            job.completed_at = datetime.utcnow()
            
            if progress_callback:
                progress_callback.on_error(job, e)
            
            raise TranslationPipelineError(f"Pipeline failed: {e}")
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        domain: str = "general"
    ) -> TranslationResult:
        """
        Translate plain text (no document).
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            domain: Domain for glossary
            
        Returns:
            TranslationResult
        """
        request = TranslationRequest(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain
        )
        
        # Check cache
        if self.cache:
            cached = self.cache.get(request)
            if cached:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached
        
        # Apply glossary preprocessing
        preprocessed_text = text
        if self.glossary:
            preprocessed_text = self.glossary.preprocess(
                text, domain, source_lang, target_lang
            )
        
        # Translate
        translated_text = self.engine.translate(
            preprocessed_text,
            source_lang,
            target_lang
        )
        
        # Apply glossary postprocessing
        glossary_result = None
        if self.glossary:
            glossary_result = self.glossary.postprocess(
                translated_text, domain, source_lang, target_lang
            )
            translated_text = glossary_result.text
        
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
            glossary_applied=glossary_result is not None if glossary_result else False,
            glossary_terms_used=glossary_result.terms_applied if glossary_result else []
        )
        
        # Cache result
        if self.cache:
            self.cache.set(request, result)
        
        return result
    
    def set_engine(self, engine: ITranslationEngine) -> None:
        """Change translation engine."""
        logger.info(f"Switching engine from {self.engine.name} to {engine.name}")
        self.engine = engine
    
    def get_supported_file_types(self) -> List[FileType]:
        """Get list of supported file types."""
        return list(self.parsers.keys())
    
    def get_health(self) -> dict:
        """Get pipeline health status."""
        health = {
            'engine': {
                'name': self.engine.name,
                'model': self.engine.model_name,
                'status': 'healthy' if self.engine.validate_config() else 'error'
            },
            'supported_file_types': [ft.value for ft in self.get_supported_file_types()]
        }
        
        if self.cache:
            health['cache'] = {
                'status': 'healthy',
                'stats': self.cache.get_stats()
            }
        
        if self.glossary:
            health['glossary'] = {
                'status': 'healthy',
                'stats': self.glossary.get_stats()
            }
        
        return health
    
    def _parse_document(self, input_path: Path) -> Document:
        """Parse document using appropriate parser."""
        file_type = FileType.from_extension(input_path.suffix)
        
        if file_type not in self.parsers:
            raise ValueError(f"Unsupported file type: {file_type.value}")
        
        parser = self.parsers[file_type]
        return parser.parse(input_path)
    
    def _get_translatable_segments(self, document: Document) -> List[TextSegment]:
        """Get segments ready for translation."""
        return [s for s in document.segments if s.text.strip()]
    
    def _translate_segment(
        self,
        segment: TextSegment,
        source_lang: str,
        target_lang: str,
        domain: str
    ) -> TranslationResult:
        """Translate single segment."""
        request = TranslationRequest(
            text=segment.text,
            source_lang=source_lang,
            target_lang=target_lang,
            domain=domain,
            segment_id=segment.id
        )
        
        # Check cache
        if self.cache:
            cached = self.cache.get(request)
            if cached:
                return cached
        
        # Apply glossary preprocessing
        preprocessed_text = segment.text
        if self.glossary:
            preprocessed_text = self.glossary.preprocess(
                segment.text, domain, source_lang, target_lang
            )
        
        # Translate
        translated_text = self.engine.translate(
            preprocessed_text,
            source_lang,
            target_lang,
            context=segment.context
        )
        
        # Apply glossary postprocessing
        glossary_result = None
        if self.glossary:
            glossary_result = self.glossary.postprocess(
                translated_text, domain, source_lang, target_lang
            )
            translated_text = glossary_result.text
        
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
            glossary_applied=glossary_result is not None if glossary_result else False,
            glossary_terms_used=glossary_result.terms_applied if glossary_result else []
        )
        
        # Cache result
        if self.cache:
            self.cache.set(request, result)
        
        return result
    
    def _format_document(self, document: Document, output_path: Path):
        """Format and save document."""
        file_type = document.file_type
        
        if file_type not in self.formatters:
            raise ValueError(f"Unsupported file type: {file_type.value}")
        
        formatter = self.formatters[file_type]
        formatter.format(document, output_path, preserve_formatting=True)


class TranslationPipelineError(Exception):
    """Exception raised when pipeline fails."""
    pass


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    from ..engines.openai_engine import OpenAIEngine
    from .interfaces import ConsoleProgressCallback
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY environment variable")
        exit(1)
    
    # Create pipeline
    engine = OpenAIEngine(api_key=api_key, model="gpt-4o-mini")
    pipeline = TranslationPipeline(engine=engine)
    
    # Translate document
    input_file = Path("test_document.docx")
    output_file = Path("test_document_ru.docx")
    
    if input_file.exists():
        callback = ConsoleProgressCallback()
        
        job = pipeline.translate_document(
            input_path=input_file,
            output_path=output_file,
            source_lang="en",
            target_lang="ru",
            progress_callback=callback
        )
        
        print(f"\n✓ Translation complete!")
        print(f"  Output: {job.output_file}")
        print(f"  Duration: {job.duration:.2f}s")
        print(f"  Segments: {job.translated_segments}/{job.total_segments}")
        print(f"  Cached: {job.cached_segments}")
    else:
        print(f"Input file not found: {input_file}")