"""
Engine Registry - Factory for translation engines.
Provides unified interface for creating and managing engines.
"""
from typing import Dict, Type, List
import logging

from ..core.interfaces import ITranslationEngine
from .openai_engine import OpenAIEngine
from .deepl_engine import DeepLEngine


logger = logging.getLogger(__name__)


class EngineRegistry:
    """
    Registry for translation engines.
    Manages available engines and provides factory methods.
    """
    
    # Registry of available engines
    _engines: Dict[str, Type[ITranslationEngine]] = {}
    
    @classmethod
    def register(cls, name: str, engine_class: Type[ITranslationEngine]) -> None:
        """
        Register a translation engine.
        
        Args:
            name: Engine name (e.g., 'openai', 'deepl')
            engine_class: Engine class
        """
        cls._engines[name] = engine_class
        logger.info(f"Registered engine: {name}")
    
    @classmethod
    def get_engine(cls, name: str, **kwargs) -> ITranslationEngine:
        """
        Create engine instance.
        
        Args:
            name: Engine name
            **kwargs: Engine-specific parameters
            
        Returns:
            Engine instance
            
        Raises:
            ValueError: If engine not found
        """
        if name not in cls._engines:
            available = ', '.join(cls.list_engines())
            raise ValueError(
                f"Unknown engine: {name}. "
                f"Available engines: {available}"
            )
        
        engine_class = cls._engines[name]
        
        try:
            return engine_class(**kwargs)
        except Exception as e:
            logger.error(f"Failed to create engine {name}: {e}")
            raise
    
    @classmethod
    def list_engines(cls) -> List[str]:
        """
        Get list of registered engine names.
        
        Returns:
            List of engine names
        """
        return list(cls._engines.keys())
    
    @classmethod
    def get_engine_info(cls, name: str) -> Dict[str, str]:
        """
        Get information about an engine.
        
        Args:
            name: Engine name
            
        Returns:
            Dict with engine info
        """
        if name not in cls._engines:
            return {}
        
        engine_class = cls._engines[name]
        
        return {
            'name': name,
            'class': engine_class.__name__,
            'module': engine_class.__module__,
            'doc': engine_class.__doc__ or "No description"
        }
    
    @classmethod
    def is_engine_available(cls, name: str) -> bool:
        """
        Check if engine is available.
        
        Args:
            name: Engine name
            
        Returns:
            True if available
        """
        return name in cls._engines


# Register built-in engines
EngineRegistry.register('openai', OpenAIEngine)
EngineRegistry.register('deepl', DeepLEngine)


# ===== Factory Helper Functions =====

def create_openai_engine(
    api_key: str,
    model: str = "gpt-4o-mini",
    **kwargs
) -> OpenAIEngine:
    """
    Create OpenAI engine with common defaults.
    
    Args:
        api_key: OpenAI API key
        model: Model name
        **kwargs: Additional parameters
        
    Returns:
        OpenAIEngine instance
    """
    return EngineRegistry.get_engine(
        'openai',
        api_key=api_key,
        model=model,
        **kwargs
    )


def create_deepl_engine(
    api_key: str,
    pro: bool = False,
    **kwargs
) -> DeepLEngine:
    """
    Create DeepL engine with common defaults.
    
    Args:
        api_key: DeepL API key
        pro: Use Pro API
        **kwargs: Additional parameters
        
    Returns:
        DeepLEngine instance
    """
    return EngineRegistry.get_engine(
        'deepl',
        api_key=api_key,
        pro=pro,
        **kwargs
    )


def get_best_engine(api_keys: Dict[str, str], preferences: List[str] = None) -> ITranslationEngine:
    """
    Get best available engine based on API keys and preferences.
    
    Args:
        api_keys: Dict of engine_name -> api_key
        preferences: Ordered list of preferred engines
        
    Returns:
        Best available engine
        
    Raises:
        ValueError: If no engine available
    """
    if preferences is None:
        preferences = ['deepl', 'openai']  # Default preference order
    
    # Try preferences in order
    for engine_name in preferences:
        if engine_name in api_keys and api_keys[engine_name]:
            try:
                if engine_name == 'openai':
                    return create_openai_engine(api_keys[engine_name])
                elif engine_name == 'deepl':
                    return create_deepl_engine(api_keys[engine_name])
            except Exception as e:
                logger.warning(f"Failed to create {engine_name}: {e}")
                continue
    
    # Try any available engine
    for engine_name, api_key in api_keys.items():
        if api_key and EngineRegistry.is_engine_available(engine_name):
            try:
                return EngineRegistry.get_engine(engine_name, api_key=api_key)
            except Exception:
                continue
    
    raise ValueError("No translation engine available. Please provide API keys.")


# ===== Example Usage =====

if __name__ == "__main__":
    import os
    
    # List available engines
    print("Available engines:")
    for engine_name in EngineRegistry.list_engines():
        info = EngineRegistry.get_engine_info(engine_name)
        print(f"  - {engine_name}: {info['doc'].strip()}")
    
    # Create OpenAI engine
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        engine = create_openai_engine(openai_key)
        print(f"\n✓ Created {engine.name} engine (model: {engine.model_name})")
        
        # Test translation
        result = engine.translate("Hello", "en", "es")
        print(f"  Test: Hello → {result}")
    
    # Create DeepL engine
    deepl_key = os.getenv("DEEPL_API_KEY")
    if deepl_key:
        engine = create_deepl_engine(deepl_key)
        print(f"\n✓ Created {engine.name} engine")
        
        # Test translation
        result = engine.translate("Hello", "en", "de")
        print(f"  Test: Hello → {result}")
    
    # Get best engine
    api_keys = {
        'openai': openai_key,
        'deepl': deepl_key
    }
    
    try:
        best_engine = get_best_engine(api_keys, preferences=['deepl', 'openai'])
        print(f"\n✓ Best available engine: {best_engine.name}")
    except ValueError as e:
        print(f"\n✗ {e}")
