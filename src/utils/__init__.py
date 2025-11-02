"""Utility functions."""
from .config_manager import ConfigManager, get_config_manager, get_config
from .logger import setup_logging, get_logger
from .validators import validate_document, validate_language_pair
from .text_normalizer import CachedTextNormalizer, normalize, norm_key  # NEW

__all__ = [
    "ConfigManager", "get_config_manager", "get_config",
    "setup_logging", "get_logger",
    "validate_document", "validate_language_pair",
    "CachedTextNormalizer", "normalize", "norm_key",  # NEW
]
