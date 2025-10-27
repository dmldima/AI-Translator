"""
Configuration manager for loading and saving settings from YAML/JSON files.
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
import os
from dotenv import load_dotenv


@dataclass
class EngineConfig:
    """Configuration for translation engine."""
    name: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout: int = 30
    max_retries: int = 3
    
    # DeepL specific
    pro: bool = False
    formality: str = "default"


@dataclass
class CacheConfig:
    """Configuration for cache system."""
    enabled: bool = True
    db_path: str = "data/cache.db"
    max_age_days: int = 180
    max_size_mb: int = 500
    log_level: str = "INFO"


@dataclass
class GlossaryConfig:
    """Configuration for glossary system."""
    enabled: bool = True
    db_path: str = "data/glossary.db"
    log_level: str = "INFO"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_workers: int = 3
    max_retries: int = 2
    skip_existing: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_dir: str = "logs"
    log_level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    max_bytes: int = 10_000_000
    backup_count: int = 5
    use_colors: bool = True
    use_json: bool = False


@dataclass
class UIConfig:
    """Configuration for UI."""
    theme: str = "auto"  # auto, light, dark
    language: str = "en"
    window_width: int = 1200
    window_height: int = 800
    remember_window_position: bool = True


@dataclass
class AppConfig:
    """Main application configuration."""
    # Engine settings
    engine: EngineConfig = field(default_factory=EngineConfig)
    
    # Feature settings
    cache: CacheConfig = field(default_factory=CacheConfig)
    glossary: GlossaryConfig = field(default_factory=GlossaryConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # General settings
    default_source_lang: str = "en"
    default_target_lang: str = "ru"
    default_domain: str = "general"
    preserve_formatting: bool = True
    
    # Paths
    output_dir: str = "output"
    temp_dir: str = "temp"


class ConfigManager:
    """
    Configuration manager for loading/saving application settings.
    Supports YAML and JSON formats, environment variables, and defaults.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize config manager.
        
        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config_path = config_path or Path("config.yaml")
        self.config: AppConfig = AppConfig()
        
        # Load environment variables
        load_dotenv()
        
        # Load config if exists
        if self.config_path.exists():
            self.load()
        else:
            # Create default config
            self.save()
    
    def load(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig instance
        """
        if not self.config_path.exists():
            return self.config
        
        # Determine format by extension
        if self.config_path.suffix in ['.yaml', '.yml']:
            data = self._load_yaml()
        elif self.config_path.suffix == '.json':
            data = self._load_json()
        else:
            raise ValueError(f"Unsupported config format: {self.config_path.suffix}")
        
        # Parse config
        self.config = self._parse_config(data)
        
        # Override with environment variables
        self._apply_env_vars()
        
        return self.config
    
    def save(self, config: Optional[AppConfig] = None):
        """
        Save configuration to file.
        
        Args:
            config: Config to save (uses current if None)
        """
        if config:
            self.config = config
        
        # Convert to dict
        data = self._config_to_dict(self.config)
        
        # Create parent directory
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if self.config_path.suffix in ['.yaml', '.yml']:
            self._save_yaml(data)
        elif self.config_path.suffix == '.json':
            self._save_json(data)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key.
        
        Args:
            key: Configuration key (e.g., 'engine.name', 'cache.enabled')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        parts = key.split('.')
        value = self.config
        
        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by dot notation key.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        parts = key.split('.')
        obj = self.config
        
        # Navigate to parent
        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(f"Invalid config key: {key}")
        
        # Set value
        setattr(obj, parts[-1], value)
    
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML config file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    
    def _load_json(self) -> Dict[str, Any]:
        """Load JSON config file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_yaml(self, data: Dict[str, Any]):
        """Save config as YAML."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    def _save_json(self, data: Dict[str, Any]):
        """Save config as JSON."""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _parse_config(self, data: Dict[str, Any]) -> AppConfig:
        """Parse dictionary to AppConfig."""
        config = AppConfig()
        
        # Parse engine
        if 'engine' in data:
            config.engine = EngineConfig(**data['engine'])
        
        # Parse cache
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        
        # Parse glossary
        if 'glossary' in data:
            config.glossary = GlossaryConfig(**data['glossary'])
        
        # Parse batch
        if 'batch' in data:
            config.batch = BatchConfig(**data['batch'])
        
        # Parse logging
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Parse UI
        if 'ui' in data:
            config.ui = UIConfig(**data['ui'])
        
        # Parse general settings
        for key in ['default_source_lang', 'default_target_lang', 'default_domain', 
                    'preserve_formatting', 'output_dir', 'temp_dir']:
            if key in data:
                setattr(config, key, data[key])
        
        return config
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'engine': asdict(config.engine),
            'cache': asdict(config.cache),
            'glossary': asdict(config.glossary),
            'batch': asdict(config.batch),
            'logging': asdict(config.logging),
            'ui': asdict(config.ui),
            'default_source_lang': config.default_source_lang,
            'default_target_lang': config.default_target_lang,
            'default_domain': config.default_domain,
            'preserve_formatting': config.preserve_formatting,
            'output_dir': config.output_dir,
            'temp_dir': config.temp_dir
        }
    
    def _apply_env_vars(self):
        """Override config with environment variables."""
        # API keys
        if os.getenv('OPENAI_API_KEY'):
            self.config.engine.api_key = os.getenv('OPENAI_API_KEY')
        if os.getenv('DEEPL_API_KEY'):
            self.config.engine.api_key = os.getenv('DEEPL_API_KEY')
        
        # Engine
        if os.getenv('TRANSLATION_ENGINE'):
            self.config.engine.name = os.getenv('TRANSLATION_ENGINE')
        
        # Cache
        if os.getenv('CACHE_ENABLED'):
            self.config.cache.enabled = os.getenv('CACHE_ENABLED').lower() == 'true'
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.config.logging.log_level = os.getenv('LOG_LEVEL')
    
    def export_template(self, output_path: Path):
        """
        Export configuration template with comments.
        
        Args:
            output_path: Path to save template
        """
        template = """# AI Document Translator Configuration

# Translation Engine Settings
engine:
  name: openai              # Engine: openai, deepl
  api_key: null             # API key (or use env var OPENAI_API_KEY/DEEPL_API_KEY)
  model: gpt-4o-mini        # Model for OpenAI
  temperature: 0.3          # Sampling temperature (0.0 = deterministic)
  max_tokens: 4000          # Maximum tokens in response
  timeout: 30               # Request timeout in seconds
  max_retries: 3            # Number of retry attempts
  pro: false                # Use DeepL Pro API
  formality: default        # DeepL formality: default, more, less

# Cache Settings
cache:
  enabled: true             # Enable caching
  db_path: data/cache.db    # SQLite database path
  max_age_days: 180         # Cache entry TTL
  max_size_mb: 500          # Maximum cache size
  log_level: INFO           # Log level for cache

# Glossary Settings
glossary:
  enabled: true             # Enable glossary
  db_path: data/glossary.db # SQLite database path
  log_level: INFO           # Log level for glossary

# Batch Processing Settings
batch:
  max_workers: 3            # Parallel workers
  max_retries: 2            # Retry failed tasks
  skip_existing: true       # Skip existing output files

# Logging Settings
logging:
  log_dir: logs             # Log directory
  log_level: INFO           # Overall log level
  console_level: INFO       # Console output level
  file_level: DEBUG         # File output level
  max_bytes: 10000000       # Max log file size (10MB)
  backup_count: 5           # Number of backup files
  use_colors: true          # Colored console output
  use_json: false           # JSON format for file logs

# UI Settings
ui:
  theme: auto               # Theme: auto, light, dark
  language: en              # UI language
  window_width: 1200        # Window width
  window_height: 800        # Window height
  remember_window_position: true

# General Settings
default_source_lang: en     # Default source language
default_target_lang: ru     # Default target language
default_domain: general     # Default glossary domain
preserve_formatting: true   # Preserve document formatting
output_dir: output          # Output directory
temp_dir: temp              # Temporary files directory
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(template)


# ===== Global Config Instance =====

_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[Path] = None) -> ConfigManager:
    """
    Get global config manager instance.
    
    Args:
        config_path: Path to config file
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> AppConfig:
    """Get current application configuration."""
    return get_config_manager().config


# ===== Example Usage =====

if __name__ == "__main__":
    # Create config manager
    config_mgr = ConfigManager(Path("config.yaml"))
    
    # Access config
    config = config_mgr.config
    print(f"Engine: {config.engine.name}")
    print(f"Cache enabled: {config.cache.enabled}")
    
    # Get value by key
    engine_name = config_mgr.get('engine.name')
    print(f"Engine name: {engine_name}")
    
    # Set value
    config_mgr.set('engine.model', 'gpt-4')
    
    # Save config
    config_mgr.save()
    print("✓ Config saved")
    
    # Export template
    config_mgr.export_template(Path("config.template.yaml"))
    print("✓ Template exported")
