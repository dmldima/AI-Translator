"""AI Document Translator - Professional document translation with AI."""
__version__ = "1.0.0"
__author__ = "AI Document Translator Team"

from src.core.factory import TranslationSystemFactory
from src.core.models import TranslationJob, TranslationStatus, SUPPORTED_LANGUAGES
from src.core.pipeline import TranslationPipeline

__all__ = [
    "TranslationSystemFactory",
    "TranslationPipeline",
    "TranslationJob",
    "TranslationStatus",
    "SUPPORTED_LANGUAGES",
]
