"""
Secure credential storage using system keyring.
Replaces plaintext API key storage.
"""
import os
import logging
from typing import Optional
from pathlib import Path

try:
    import keyring
    from keyring.errors import KeyringError
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring not available - falling back to environment variables")


logger = logging.getLogger(__name__)


class SecureCredentialStore:
    """
    Secure credential storage using system keyring.
    Falls back to environment variables if keyring unavailable.
    """
    
    SERVICE_NAME = "ai-document-translator"
    
    def __init__(self):
        self.use_keyring = KEYRING_AVAILABLE
        if not self.use_keyring:
            logger.warning(
                "Using environment variables for credentials. "
                "Install 'keyring' package for secure storage: pip install keyring"
            )
    
    def set_credential(self, key: str, value: str) -> bool:
        """
        Store credential securely.
        
        Args:
            key: Credential key (e.g., 'openai_api_key')
            value: Credential value
            
        Returns:
            True if successful
        """
        if not value or not value.strip():
            raise ValueError("Credential value cannot be empty")
        
        # Validate API key format
        self._validate_api_key(key, value)
        
        if self.use_keyring:
            try:
                keyring.set_password(self.SERVICE_NAME, key, value)
                logger.info(f"Stored credential '{key}' in system keyring")
                return True
            except KeyringError as e:
                logger.error(f"Failed to store credential: {e}")
                return False
        else:
            # Fallback: warn user to set environment variable
            logger.warning(
                f"Store credential as environment variable: "
                f"export {key.upper()}='{value}'"
            )
            return False
    
    def get_credential(self, key: str) -> Optional[str]:
        """
        Retrieve credential.
        
        Args:
            key: Credential key
            
        Returns:
            Credential value or None
        """
        # Try keyring first
        if self.use_keyring:
            try:
                value = keyring.get_password(self.SERVICE_NAME, key)
                if value:
                    logger.debug(f"Retrieved credential '{key}' from keyring")
                    return value
            except KeyringError as e:
                logger.error(f"Failed to retrieve credential: {e}")
        
        # Fallback to environment variable
        env_key = key.upper()
        value = os.getenv(env_key)
        if value:
            logger.debug(f"Retrieved credential '{key}' from environment")
            return value
        
        logger.warning(f"Credential '{key}' not found")
        return None
    
    def delete_credential(self, key: str) -> bool:
        """
        Delete credential.
        
        Args:
            key: Credential key
            
        Returns:
            True if successful
        """
        if self.use_keyring:
            try:
                keyring.delete_password(self.SERVICE_NAME, key)
                logger.info(f"Deleted credential '{key}' from keyring")
                return True
            except KeyringError as e:
                logger.error(f"Failed to delete credential: {e}")
                return False
        
        logger.warning("Cannot delete environment variables programmatically")
        return False
    
    def list_credentials(self) -> list:
        """
        List all stored credential keys.
        
        Returns:
            List of credential keys
        """
        # Keyring doesn't provide list functionality
        # Return known credential types
        return [
            'openai_api_key',
            'deepl_api_key',
            'anthropic_api_key'
        ]
    
    def _validate_api_key(self, key: str, value: str):
        """Validate API key format."""
        if len(value) < 20:
            logger.warning(f"API key '{key}' seems too short: {len(value)} chars")
        
        # OpenAI validation
        if 'openai' in key.lower():
            if not value.startswith('sk-'):
                logger.warning("OpenAI API keys should start with 'sk-'")
        
        # DeepL validation
        if 'deepl' in key.lower():
            if '-' not in value:
                logger.warning("DeepL API keys typically contain hyphens")
    
    def migrate_from_plaintext(self, config_file: Path) -> int:
        """
        Migrate API keys from plaintext config to secure storage.
        
        Args:
            config_file: Path to config file containing plaintext keys
            
        Returns:
            Number of keys migrated
        """
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            return 0
        
        import yaml
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        migrated = 0
        
        # Migrate OpenAI key
        if config.get('engine', {}).get('api_key'):
            key = config['engine']['api_key']
            if self.set_credential('openai_api_key', key):
                migrated += 1
                logger.info("Migrated OpenAI API key")
        
        # Check for DeepL key
        if config.get('deepl_api_key'):
            if self.set_credential('deepl_api_key', config['deepl_api_key']):
                migrated += 1
                logger.info("Migrated DeepL API key")
        
        if migrated > 0:
            logger.info(
                f"Migrated {migrated} API keys to secure storage. "
                f"Remove them from {config_file} manually."
            )
        
        return migrated


# Global instance
_credential_store: Optional[SecureCredentialStore] = None


def get_credential_store() -> SecureCredentialStore:
    """Get global credential store instance."""
    global _credential_store
    if _credential_store is None:
        _credential_store = SecureCredentialStore()
    return _credential_store


def get_api_key(engine: str) -> Optional[str]:
    """
    Get API key for engine.
    
    Args:
        engine: Engine name ('openai', 'deepl', 'anthropic')
        
    Returns:
        API key or None
    """
    store = get_credential_store()
    key_name = f"{engine.lower()}_api_key"
    return store.get_credential(key_name)


def set_api_key(engine: str, api_key: str) -> bool:
    """
    Set API key for engine.
    
    Args:
        engine: Engine name
        api_key: API key value
        
    Returns:
        True if successful
    """
    store = get_credential_store()
    key_name = f"{engine.lower()}_api_key"
    return store.set_credential(key_name, api_key)


# CLI utility for managing credentials
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python secure_credentials.py set <engine> <api_key>")
        print("  python secure_credentials.py get <engine>")
        print("  python secure_credentials.py delete <engine>")
        print("  python secure_credentials.py migrate <config_file>")
        print("  python secure_credentials.py list")
        sys.exit(1)
    
    command = sys.argv[1]
    store = get_credential_store()
    
    if command == "set":
        if len(sys.argv) != 4:
            print("Usage: python secure_credentials.py set <engine> <api_key>")
            sys.exit(1)
        
        engine = sys.argv[2]
        api_key = sys.argv[3]
        
        if set_api_key(engine, api_key):
            print(f"✓ Stored {engine} API key securely")
        else:
            print(f"✗ Failed to store API key")
            sys.exit(1)
    
    elif command == "get":
        if len(sys.argv) != 3:
            print("Usage: python secure_credentials.py get <engine>")
            sys.exit(1)
        
        engine = sys.argv[2]
        api_key = get_api_key(engine)
        
        if api_key:
            # Show only first/last 4 chars for security
            masked = f"{api_key[:4]}...{api_key[-4:]}"
            print(f"{engine} API key: {masked}")
        else:
            print(f"✗ No API key found for {engine}")
            sys.exit(1)
    
    elif command == "delete":
        if len(sys.argv) != 3:
            print("Usage: python secure_credentials.py delete <engine>")
            sys.exit(1)
        
        engine = sys.argv[2]
        key_name = f"{engine.lower()}_api_key"
        
        if store.delete_credential(key_name):
            print(f"✓ Deleted {engine} API key")
        else:
            print(f"✗ Failed to delete API key")
            sys.exit(1)
    
    elif command == "migrate":
        if len(sys.argv) != 3:
            print("Usage: python secure_credentials.py migrate <config_file>")
            sys.exit(1)
        
        config_file = Path(sys.argv[2])
        migrated = store.migrate_from_plaintext(config_file)
        
        if migrated > 0:
            print(f"✓ Migrated {migrated} API keys")
            print(f"⚠ Remove plaintext keys from {config_file}")
        else:
            print("No keys to migrate")
    
    elif command == "list":
        keys = store.list_credentials()
        print("Supported credential keys:")
        for key in keys:
            value = store.get_credential(key)
            status = "✓ stored" if value else "✗ not set"
            print(f"  {key}: {status}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
