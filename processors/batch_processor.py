"""
Batch Processor for translating multiple documents.
Supports parallel processing, retry logic, and progress tracking.
"""
import logging
from pathlib import Path
from typing import List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

from ..core.pipeline import TranslationPipeline
from ..core.models import TranslationJob, TranslationStatus
from ..core.interfaces import IProgressCallback


logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Single task in batch processing."""
    input_file: Path
    output_file: Path
    source_lang: str
    target_lang: str
    domain: str = "general"
    priority: int = 0
    
    # Results
    job: Optional[TranslationJob] = None
    success: bool = False
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class BatchResult:
    """Results of batch processing."""
    batch_id: str
    total_tasks: int
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    tasks: List[BatchTask] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[float]:
        """Get batch duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def success_rate(self) -> float:
        """Get success rate percentage."""
        if self.total_tasks == 0:
            return 0.0
        return (self.successful / self.total_tasks) * 100
    
    def get_failed_tasks(self) -> List[BatchTask]:
        """Get list of failed tasks."""
        return [t for t in self.tasks if not t.success and t.error]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'batch_id': self.batch_id,
            'total_tasks': self.total_tasks,
            'successful': self.successful,
            'failed': self.failed,
            'skipped': self.skipped,
            'success_rate': f"{self.success_rate:.1f}%",
            'duration': f"{self.duration:.2f}s" if self.duration else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class BatchProcessor:
    """
    Batch processor for translating multiple documents.
    Supports parallel processing and retry logic.
    """
    
    def __init__(
        self,
        pipeline: TranslationPipeline,
        max_workers: int = 3,
        max_retries: int = 2,
        skip_existing: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            pipeline: Translation pipeline
            max_workers: Maximum parallel workers
            max_retries: Maximum retry attempts per task
            skip_existing: Skip files that already exist
        """
        self.pipeline = pipeline
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.skip_existing = skip_existing
        
        # State
        self._current_batch: Optional[BatchResult] = None
        self._stop_flag = threading.Event()
        self._task_queue = queue.Queue()
        
        logger.info(
            f"Initialized batch processor: "
            f"max_workers={max_workers}, max_retries={max_retries}"
        )
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        source_lang: str,
        target_lang: str,
        pattern: str = "*.docx",
        domain: str = "general",
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process all matching files in directory.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            source_lang: Source language code
            target_lang: Target language code
            pattern: File pattern (e.g., "*.docx", "*.xlsx")
            domain: Domain for glossary
            progress_callback: Optional callback(current, total, task)
            
        Returns:
            BatchResult with statistics
        """
        # Find files
        files = list(input_dir.glob(pattern))
        if not files:
            logger.warning(f"No files matching '{pattern}' found in {input_dir}")
            return BatchResult(
                batch_id=f"batch_{datetime.now().timestamp()}",
                total_tasks=0
            )
        
        logger.info(f"Found {len(files)} files to process")
        
        # Create tasks
        tasks = []
        for input_file in files:
            # Generate output filename
            output_file = output_dir / f"{input_file.stem}_{target_lang}{input_file.suffix}"
            
            # Skip if exists
            if self.skip_existing and output_file.exists():
                logger.info(f"Skipping existing file: {output_file.name}")
                continue
            
            task = BatchTask(
                input_file=input_file,
                output_file=output_file,
                source_lang=source_lang,
                target_lang=target_lang,
                domain=domain
            )
            tasks.append(task)
        
        # Process tasks
        return self.process_tasks(tasks, progress_callback)
    
    def process_tasks(
        self,
        tasks: List[BatchTask],
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """
        Process list of tasks.
        
        Args:
            tasks: List of BatchTask objects
            progress_callback: Optional callback(current, total, task)
            
        Returns:
            BatchResult with statistics
        """
        # Create batch result
        batch_id = f"batch_{datetime.now().timestamp()}"
        result = BatchResult(
            batch_id=batch_id,
            total_tasks=len(tasks),
            tasks=tasks,
            started_at=datetime.now()
        )
        
        self._current_batch = result
        self._stop_flag.clear()
        
        logger.info(f"Starting batch {batch_id}: {len(tasks)} tasks")
        
        # Create output directory
        if tasks:
            tasks[0].output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process tasks in parallel
        completed = 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._process_task, task): task
                for task in tasks
            }
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                if self._stop_flag.is_set():
                    logger.warning("Batch processing stopped by user")
                    break
                
                task = future_to_task[future]
                
                try:
                    success = future.result()
                    if success:
                        result.successful += 1
                    else:
                        result.failed += 1
                except Exception as e:
                    logger.error(f"Task failed unexpectedly: {e}")
                    task.error = str(e)
                    result.failed += 1
                
                completed += 1
                
                # Progress callback
                if progress_callback:
                    progress_callback(completed, len(tasks), task)
        
        result.completed_at = datetime.now()
        self._current_batch = None
        
        logger.info(
            f"Batch {batch_id} complete: "
            f"{result.successful} successful, {result.failed} failed, "
            f"duration={result.duration:.2f}s"
        )
        
        return result
    
    def _process_task(self, task: BatchTask) -> bool:
        """
        Process single task with retry logic.
        
        Args:
            task: BatchTask to process
            
        Returns:
            True if successful
        """
        for attempt in range(self.max_retries + 1):
            task.attempts = attempt + 1
            
            try:
                logger.info(
                    f"Processing {task.input_file.name} "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                
                # Translate document
                job = self.pipeline.translate_document(
                    input_path=task.input_file,
                    output_path=task.output_file,
                    source_lang=task.source_lang,
                    target_lang=task.target_lang,
                    domain=task.domain
                )
                
                # Check if successful
                if job.status == TranslationStatus.COMPLETED:
                    task.job = job
                    task.success = True
                    logger.info(f"✓ Completed: {task.output_file.name}")
                    return True
                else:
                    task.error = f"Translation failed: {job.status.value}"
                    logger.warning(f"✗ Failed: {task.input_file.name}")
                    
            except Exception as e:
                task.error = str(e)
                logger.error(f"Error processing {task.input_file.name}: {e}")
                
                # Retry if not last attempt
                if attempt < self.max_retries:
                    logger.info(f"Retrying... ({attempt + 1}/{self.max_retries})")
                    continue
        
        # All attempts failed
        task.success = False
        return False
    
    def stop(self):
        """Stop batch processing gracefully."""
        logger.info("Stopping batch processing...")
        self._stop_flag.set()
    
    def get_current_batch(self) -> Optional[BatchResult]:
        """Get current batch result (while processing)."""
        return self._current_batch


class BatchProgressCallback(IProgressCallback):
    """Progress callback for batch processing."""
    
    def __init__(self, on_file_start=None, on_file_complete=None):
        self.on_file_start = on_file_start
        self.on_file_complete = on_file_complete
        self.current_file = None
    
    def on_start(self, job):
        if self.on_file_start:
            self.on_file_start(job.input_file)
        self.current_file = job.input_file
    
    def on_progress(self, job, current, total):
        pass  # Handled by batch progress
    
    def on_complete(self, job):
        if self.on_file_complete:
            self.on_file_complete(job)
    
    def on_error(self, job, error):
        pass  # Handled by batch processor
    
    def on_segment_translated(self, segment, result):
        pass


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    from ..engines.openai_engine import OpenAIEngine
    from ..core.pipeline import TranslationPipeline
    
    # Setup
    api_key = os.getenv("OPENAI_API_KEY")
    engine = OpenAIEngine(api_key=api_key, model="gpt-4o-mini")
    pipeline = TranslationPipeline(engine=engine)
    
    # Create batch processor
    processor = BatchProcessor(
        pipeline=pipeline,
        max_workers=3,
        max_retries=2
    )
    
    # Progress callback
    def progress_callback(current, total, task):
        print(f"Progress: {current}/{total} - {task.input_file.name}")
        if task.success:
            print(f"  ✓ Success: {task.output_file.name}")
        elif task.error:
            print(f"  ✗ Failed: {task.error}")
    
    # Process directory
    result = processor.process_directory(
        input_dir=Path("documents"),
        output_dir=Path("translated"),
        source_lang="en",
        target_lang="ru",
        pattern="*.docx",
        progress_callback=progress_callback
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Batch Processing Results")
    print(f"{'='*60}")
    print(f"Total: {result.total_tasks}")
    print(f"Successful: {result.successful}")
    print(f"Failed: {result.failed}")
    print(f"Success Rate: {result.success_rate:.1f}%")
    print(f"Duration: {result.duration:.2f}s")
    
    # Show failed tasks
    if result.failed > 0:
        print(f"\nFailed tasks:")
        for task in result.get_failed_tasks():
            print(f"  - {task.input_file.name}: {task.error}")
