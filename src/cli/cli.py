#!/usr/bin/env python3
"""
AI Document Translator - Command Line Interface
"""
import sys
import click
from pathlib import Path
import os
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import print as rprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.factory import TranslationSystemFactory
from src.core.models import TranslationJob, TextSegment, TranslationResult, SUPPORTED_LANGUAGES
from src.core.interfaces import IProgressCallback


console = Console()


class RichProgressCallback(IProgressCallback):
    """Progress callback using Rich library."""
    
    def __init__(self, progress: Progress, task_id):
        self.progress = progress
        self.task_id = task_id
    
    def on_start(self, job: TranslationJob) -> None:
        self.progress.update(self.task_id, total=job.total_segments)
    
    def on_progress(self, job: TranslationJob, current: int, total: int) -> None:
        self.progress.update(self.task_id, completed=current)
    
    def on_segment_translated(
        self,
        segment: TextSegment,
        result: TranslationResult
    ) -> None:
        pass  # Rich handles this automatically
    
    def on_complete(self, job: TranslationJob) -> None:
        pass
    
    def on_error(self, job: TranslationJob, error: Exception) -> None:
        console.print(f"[red]Error: {error}[/red]")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """AI Document Translator - Translate documents with AI while preserving formatting."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.argument('output_file', type=click.Path(path_type=Path))
@click.option('--source', '-s', required=True, help='Source language code (e.g., en)')
@click.option('--target', '-t', required=True, help='Target language code (e.g., ru)')
@click.option('--engine', '-e', default='openai', type=click.Choice(['openai', 'deepl']), help='Translation engine')
@click.option('--model', '-m', default='gpt-4o-mini', help='Model name for OpenAI')
@click.option('--domain', '-d', default='general', help='Domain for glossary')
@click.option('--api-key', envvar='OPENAI_API_KEY', help='API key (or set OPENAI_API_KEY/DEEPL_API_KEY env var)')
@click.option('--deepl-pro', is_flag=True, help='Use DeepL Pro API')
@click.option('--no-cache', is_flag=True, help='Disable cache')
@click.option('--no-glossary', is_flag=True, help='Disable glossary')
def translate(input_file, output_file, source, target, engine, model, domain, api_key, deepl_pro, no_cache, no_glossary):
    """
    Translate a document from one language to another.
    
    Supported formats: .docx, .xlsx
    
    Examples:
    
        # Translate DOCX (English to Russian with OpenAI)
        translator translate document.docx document_ru.docx -s en -t ru
        
        # Translate XLSX (English to German with DeepL)
        translator translate data.xlsx data_de.xlsx -s en -t de --engine deepl
        
        # Use DeepL Pro
        translator translate doc.docx doc_fr.docx -s en -t fr --engine deepl --deepl-pro
        
        # Specify domain for better glossary
        translator translate legal.docx legal_ru.docx -s en -t ru --domain legal
    """
    # Get API key from environment if not provided
    if not api_key:
        if engine == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')
        else:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        console.print(f"[red]Error: API key required. Set {engine.upper()}_API_KEY or use --api-key[/red]")
        sys.exit(1)
    
    # Display header
    console.print("\n[bold cyan]🌐 AI Document Translator[/bold cyan]")
    console.print(f"[dim]Input:  {input_file}[/dim]")
    console.print(f"[dim]Output: {output_file}[/dim]")
    console.print(f"[dim]Languages: {source.upper()} → {target.upper()}[/dim]")
    console.print(f"[dim]Engine: {engine} ({model if engine == 'openai' else 'DeepL'})[/dim]\n")
    
    try:
        # Build engine kwargs
        engine_kwargs = {}
        if engine == 'openai':
            engine_kwargs['model'] = model
        elif engine == 'deepl':
            engine_kwargs['pro'] = deepl_pro
        
        # Create pipeline using factory
        console.print("[yellow]Initializing pipeline...[/yellow]")
        pipeline = TranslationSystemFactory.create_pipeline(
            engine_name=engine,
            api_key=api_key,
            cache_enabled=not no_cache,
            glossary_enabled=not no_glossary,
            **engine_kwargs
        )
        
        console.print("[green]✓ Pipeline initialized[/green]\n")
        
        # Translate with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Translating...", total=100)
            callback = RichProgressCallback(progress, task)
            
            job = pipeline.translate_document(
                input_path=input_file,
                output_path=output_file,
                source_lang=source,
                target_lang=target,
                domain=domain,
                progress_callback=callback
            )
        
        # Display results
        console.print(f"\n[bold green]✓ Translation complete![/bold green]")
        console.print(f"[dim]Saved to: {output_file}[/dim]\n")
        
        # Statistics table
        table = Table(title="Translation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total segments", str(job.total_segments))
        table.add_row("Translated", str(job.translated_segments))
        table.add_row("From cache", str(job.cached_segments))
        table.add_row("Failed", str(job.failed_segments))
        table.add_row("Duration", f"{job.duration:.2f}s")
        
        # Add engine stats
        stats = pipeline.engine.get_usage_stats()
        if engine == 'openai':
            table.add_row("Tokens used", str(stats.get('total_tokens', 0)))
        elif engine == 'deepl':
            table.add_row("Characters used", str(stats.get('total_chars', 0)))
        table.add_row("Estimated cost", f"${stats.get('estimated_cost_usd', 0):.4f}")
        
        console.print(table)
        
        # Warnings
        if job.failed_segments > 0:
            console.print(f"\n[yellow]Warning: {job.failed_segments} segments failed to translate[/yellow]")
            for error in job.errors[:3]:  # Show first 3 errors
                console.print(f"[dim]  • {error}[/dim]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Translation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


@cli.command()
@click.argument('text')
@click.option('--source', '-s', required=True, help='Source language code')
@click.option('--target', '-t', required=True, help='Target language code')
@click.option('--engine', '-e', default='openai', type=click.Choice(['openai', 'deepl']), help='Translation engine')
@click.option('--model', '-m', default='gpt-4o-mini', help='Model name (OpenAI only)')
@click.option('--api-key', help='API key')
@click.option('--deepl-pro', is_flag=True, help='Use DeepL Pro')
def text(text, source, target, engine, model, api_key, deepl_pro):
    """
    Translate plain text (no document).
    
    Example:
        translator text "Hello, world!" -s en -t ru
        translator text "Bonjour" -s fr -t en --engine deepl
    """
    # Get API key
    if not api_key:
        if engine == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')
        else:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        console.print("[red]Error: API key required[/red]")
        sys.exit(1)
    
    try:
        # Build engine kwargs
        engine_kwargs = {}
        if engine == 'openai':
            engine_kwargs['model'] = model
        elif engine == 'deepl':
            engine_kwargs['pro'] = deepl_pro
        
        # Create pipeline
        pipeline = TranslationSystemFactory.create_pipeline(
            engine_name=engine,
            api_key=api_key,
            cache_enabled=False,
            glossary_enabled=False,
            **engine_kwargs
        )
        
        # Translate
        console.print(f"\n[cyan]Translating: {source.upper()} → {target.upper()}[/cyan]")
        result = pipeline.translate_text(text, source, target)
        
        # Display result
        console.print(f"\n[bold]Original:[/bold]")
        console.print(f"  {result.original_text}")
        console.print(f"\n[bold]Translation:[/bold]")
        console.print(f"  [green]{result.translated_text}[/green]")
        
        if result.cached:
            console.print(f"\n[dim]ℹ From cache[/dim]")
        
        if result.glossary_applied:
            console.print(f"[dim]ℹ Glossary terms: {', '.join(result.glossary_terms_used)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option('--engine', '-e', default='openai', type=click.Choice(['openai', 'deepl']), help='Translation engine')
@click.option('--api-key', help='API key')
def test(engine, api_key):
    """
    Test translation engine configuration.
    
    Example:
        translator test
        translator test --engine deepl
    """
    # Get API key
    if not api_key:
        if engine == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')
        else:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        console.print("[red]Error: API key required[/red]")
        sys.exit(1)
    
    console.print(f"\n[cyan]Testing {engine} configuration...[/cyan]\n")
    
    try:
        # Create engine
        translation_engine = TranslationSystemFactory.create_engine(
            engine_name=engine,
            api_key=api_key
        )
        
        # Test translation
        console.print("[yellow]Running test translation...[/yellow]")
        result = translation_engine.translate("Hello", "en", "es")
        
        if result:
            console.print(f"[green]✓ Test passed: 'Hello' → '{result}'[/green]")
            console.print(f"\n[bold]Engine Info:[/bold]")
            console.print(f"  Name: {translation_engine.name}")
            console.print(f"  Model: {translation_engine.model_name}")
            
            langs = translation_engine.get_supported_languages()
            console.print(f"  Supported languages: {len(langs)}")
            
            # Show usage if available
            stats = translation_engine.get_usage_stats()
            if engine == 'deepl' and 'character_limit' in stats:
                console.print(f"  Usage: {stats['character_count']}/{stats['character_limit']} chars")
            
            console.print(f"\n[green]✓ Configuration valid[/green]")
        else:
            console.print("[red]✗ Test failed[/red]")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]✗ Test failed: {e}[/red]")
        sys.exit(1)


@cli.command()
def languages():
    """
    List supported languages.
    
    Example:
        translator languages
    """
    console.print("\n[bold cyan]Supported Languages[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Code", style="cyan", width=8)
    table.add_column("Language", style="white")
    
    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        table.add_row(code, name)
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(SUPPORTED_LANGUAGES)} languages[/dim]\n")


@cli.command()
def formats():
    """
    List supported file formats.
    
    Example:
        translator formats
    """
    from src.core.factory import ParserFactory, FormatterFactory
    from src.core.models import FileType
    
    console.print("\n[bold cyan]Supported File Formats[/bold cyan]\n")
    
    parsers = set(ParserFactory.get_supported_types())
    formatters = set(FormatterFactory.get_supported_types())
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Extension", style="cyan", width=12)
    table.add_column("Format", style="white")
    table.add_column("Read", style="green")
    table.add_column("Write", style="green")
    
    formats_info = {
        'docx': 'Microsoft Word',
        'xlsx': 'Microsoft Excel',
        'doc': 'Word (Legacy)',
        'xls': 'Excel (Legacy)',
        'pdf': 'PDF Document',
        'txt': 'Plain Text'
    }
    
    for ext, name in formats_info.items():
        try:
            ft = FileType(ext)
            can_read = "✓" if ft in parsers else "✗"
            can_write = "✓" if ft in formatters else "✗"
        except ValueError:
            can_read = "✗"
            can_write = "✗"
        
        table.add_row(f".{ext}", name, can_read, can_write)
    
    console.print(table)
    console.print()


@cli.command()
@click.option('--engine', '-e', default='openai', type=click.Choice(['openai', 'deepl']), help='Translation engine')
@click.option('--api-key', help='API key')
def stats(engine, api_key):
    """
    Show usage statistics.
    
    Example:
        translator stats
        translator stats --engine deepl
    """
    # Get API key
    if not api_key:
        if engine == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')
        else:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        console.print("[red]Error: API key required[/red]")
        sys.exit(1)
    
    try:
        translation_engine = TranslationSystemFactory.create_engine(
            engine_name=engine,
            api_key=api_key
        )
        stats = translation_engine.get_usage_stats()
        
        console.print(f"\n[bold cyan]Usage Statistics - {engine.upper()}[/bold cyan]\n")
        
        table = Table(show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Engine", stats['engine'])
        table.add_row("Model", stats['model'])
        table.add_row("Total Requests", str(stats['total_requests']))
        
        if engine == 'openai':
            table.add_row("Total Tokens", str(stats.get('total_tokens', 0)))
        elif engine == 'deepl':
            table.add_row("Total Characters", str(stats.get('total_chars', 0)))
            if 'character_limit' in stats:
                table.add_row("Usage Limit", f"{stats['character_count']}/{stats['character_limit']}")
        
        table.add_row("Total Errors", str(stats['total_errors']))
        table.add_row("Estimated Cost", f"${stats.get('estimated_cost_usd', 0):.4f}")
        
        console.print(table)
        console.print()
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True, path_type=Path))
@click.argument('output_dir', type=click.Path(path_type=Path))
@click.option('--source', '-s', required=True, help='Source language code')
@click.option('--target', '-t', required=True, help='Target language code')
@click.option('--engine', '-e', default='openai', type=click.Choice(['openai', 'deepl']), help='Translation engine')
@click.option('--pattern', '-p', default='*.docx', help='File pattern to match (*.docx, *.xlsx)')
@click.option('--api-key', help='API key')
@click.option('--deepl-pro', is_flag=True, help='Use DeepL Pro')
@click.option('--model', '-m', default='gpt-4o-mini', help='Model name (OpenAI only)')
def batch(input_dir, output_dir, source, target, engine, pattern, api_key, deepl_pro, model):
    """
    Translate multiple documents in a directory.
    
    Example:
        translator batch ./documents ./translated -s en -t ru
        translator batch ./spreadsheets ./translated -s en -t de --pattern "*.xlsx" --engine deepl
    """
    # Get API key
    if not api_key:
        if engine == 'deepl':
            api_key = os.getenv('DEEPL_API_KEY')
        else:
            api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        console.print("[red]Error: API key required[/red]")
        sys.exit(1)
    
    # Find files
    files = list(input_dir.glob(pattern))
    if not files:
        console.print(f"[yellow]No files matching '{pattern}' found in {input_dir}[/yellow]")
        sys.exit(0)
    
    console.print(f"\n[bold cyan]Batch Translation[/bold cyan]")
    console.print(f"[dim]Found {len(files)} files[/dim]\n")
    
    try:
        # Build engine kwargs
        engine_kwargs = {}
        if engine == 'openai':
            engine_kwargs['model'] = model
        elif engine == 'deepl':
            engine_kwargs['pro'] = deepl_pro
        
        # Create pipeline
        pipeline = TranslationSystemFactory.create_pipeline(
            engine_name=engine,
            api_key=api_key,
            cache_enabled=True,  # Enable cache for batch
            glossary_enabled=False,
            **engine_kwargs
        )
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each file
        successful = 0
        failed = 0
        
        for file in files:
            output_file = output_dir / f"{file.stem}_{target}{file.suffix}"
            
            console.print(f"[cyan]Processing:[/cyan] {file.name}")
            
            try:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("", total=100)
                    callback = RichProgressCallback(progress, task)
                    
                    job = pipeline.translate_document(
                        input_path=file,
                        output_path=output_file,
                        source_lang=source,
                        target_lang=target,
                        progress_callback=callback
                    )
                
                console.print(f"[green]  ✓ Saved to: {output_file.name}[/green]\n")
                successful += 1
                
            except Exception as e:
                console.print(f"[red]  ✗ Failed: {e}[/red]\n")
                failed += 1
        
        # Summary
        console.print(f"\n[bold]Batch Summary:[/bold]")
        console.print(f"  Successful: [green]{successful}[/green]")
        console.print(f"  Failed: [red]{failed}[/red]")
        console.print(f"  Total: {len(files)}\n")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    cli()
