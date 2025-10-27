# AI Document Translator

Professional AI-powered document translation with perfect formatting preservation.

## Features

- 🌐 **Multiple Translation Engines**: OpenAI (GPT-4, GPT-3.5) and DeepL
- 📄 **Format Preservation**: Maintains all formatting in DOCX and XLSX files
- 🚀 **Smart Caching**: Reduces costs and speeds up retranslation
- 📚 **Glossary Management**: Enforce consistent terminology
- ⚡ **Batch Processing**: Translate multiple documents in parallel
- 💻 **Multiple Interfaces**: CLI, GUI, and Python API
- 🔒 **Production Ready**: Thread-safe, error handling, logging

## Supported Formats

- **Microsoft Word** (.docx)
- **Microsoft Excel** (.xlsx)
- Coming soon: PDF, TXT, HTML

## Supported Languages

30+ languages including: English, Spanish, French, German, Russian, Japanese, Chinese, Korean, Arabic, and more.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-document-translator.git
cd ai-document-translator

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

## Quick Start

### CLI Usage

```bash
# Translate a document
python -m src.cli.cli_interface translate document.docx output.docx -s en -t ru

# Batch translate a directory
python -m src.cli.cli_interface batch ./documents ./translated -s en -t ru

# Test configuration
python -m src.cli.cli_interface test

# List supported languages
python -m src.cli.cli_interface languages
```

### Python API Usage

```python
from pathlib import Path
from src.core.factory import TranslationSystemFactory

# Create pipeline
pipeline = TranslationSystemFactory.create_pipeline(
    engine_name="openai",
    api_key="your-api-key",
    model="gpt-4o-mini"
)

# Translate document
job = pipeline.translate_document(
    input_path=Path("document.docx"),
    output_path=Path("document_ru.docx"),
    source_lang="en",
    target_lang="ru"
)

print(f"Translated {job.translated_segments} segments")
```

### GUI Usage

```bash
# Launch macOS GUI
python -m src.gui.macos_gui
```

## Configuration

Create `config.yaml`:

```yaml
engine:
  name: openai
  model: gpt-4o-mini
  
cache:
  enabled: true
  max_age_days: 180

glossary:
  enabled: true
```

Or use environment variables:
```bash
export OPENAI_API_KEY="your-key"
export TRANSLATION_ENGINE="openai"
export CACHE_ENABLED="true"
```

## Advanced Features

### Glossary Management

```python
from src.glossary.glossary_manager import GlossaryManager, GlossaryTerm

# Add term
manager.upsert_term(
    source="Agreement",
    target="Соглашение",
    domain="legal",
    status="approved"
)

# Apply to text
result = manager.apply_to_text(text, domain="legal")
```

### Batch Processing

```python
from src.processors.batch_processor import BatchProcessor

processor = BatchProcessor(pipeline, max_workers=3)

result = processor.process_directory(
    input_dir=Path("documents"),
    output_dir=Path("translated"),
    source_lang="en",
    target_lang="ru"
)

print(f"Success rate: {result.success_rate:.1f}%")
```

## Project Structure

```
ai-document-translator/
├── src/
│   ├── cache/           # Translation caching
│   ├── cli/             # Command-line interface
│   ├── core/            # Core models and pipeline
│   ├── engines/         # Translation engines
│   ├── formatters/      # Document formatters
│   ├── glossary/        # Glossary management
│   ├── gui/             # GUI interfaces
│   ├── parsers/         # Document parsers
│   ├── processors/      # Batch processing
│   └── utils/           # Utilities
├── tests/               # Test suite
├── docs/                # Documentation
├── examples/            # Example files
└── config.yaml          # Configuration
```

## API Keys

### OpenAI
1. Sign up at https://platform.openai.com
2. Create an API key
3. Add to `.env`: `OPENAI_API_KEY=sk-...`

### DeepL
1. Sign up at https://www.deepl.com/pro-api
2. Get API key
3. Add to `.env`: `DEEPL_API_KEY=...`

## Cost Estimation

### OpenAI (GPT-4o-mini)
- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Typical document: $0.01 - $0.10

### DeepL
- Free: 500,000 characters/month
- Pro: $5.49 per 1M characters

## Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=src tests/

# Specific test
pytest tests/test_cache_manager.py
```

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Format code
black src/

# Lint
flake8 src/

# Type check
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/ai-document-translator/issues)
- Email: support@example.com

## Roadmap

- [ ] PDF support
- [ ] More translation engines (Google Translate, Azure)
- [ ] Web interface
- [ ] REST API
- [ ] Docker deployment
- [ ] Cloud storage integration

## Acknowledgments

Built with:
- OpenAI GPT models
- DeepL Translation API
- python-docx
- openpyxl
- Rich (CLI)
- PyQt6 (GUI)
