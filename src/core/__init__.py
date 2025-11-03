"""Core translation system components."""
from .factory import TranslationSystemFactory, ParserFactory, FormatterFactory, EngineFactory
from .models import (
    Document, TextSegment, TranslationJob, TranslationResult,
    TranslationRequest,  # ДОБАВЛЕНО: используется в cache
    FileType, SegmentType, TranslationStatus, SUPPORTED_LANGUAGES
)
from .pipeline import EnhancedTranslationPipeline
# ИСПРАВЛЕНО: создаем alias для обратной совместимости
TranslationPipeline = EnhancedTranslationPipeline

from .interfaces import ITranslationEngine, IDocumentParser, IDocumentFormatter

__all__ = [
    "TranslationSystemFactory", "ParserFactory", "FormatterFactory", "EngineFactory",
    "Document", "TextSegment", "TranslationJob", "TranslationResult",
    "TranslationRequest",  # ДОБАВЛЕНО
    "FileType", "SegmentType", "TranslationStatus", "SUPPORTED_LANGUAGES",
    "TranslationPipeline",  # Теперь это alias
    "EnhancedTranslationPipeline",  # ДОБАВЛЕНО: явный доступ к enhanced версии
    "ITranslationEngine", "IDocumentParser", "IDocumentFormatter",
]
