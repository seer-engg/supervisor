"""
Secrets store - loads specific secrets from .env file.
Simple dictionary-based store, similar to runtime_tool_store.
"""
from typing import Dict, Optional
from dotenv import dotenv_values
from pathlib import Path

class SecretsStore:
    """Simple store for secrets loaded from .env file."""
    
    def __init__(self):
        self._secrets: Dict[str, str] = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load specific secrets from .env file."""
        # Try current directory first
        env_file = Path(".env")
        
        if not env_file.exists():
            # Fallback: try parent directories
            current = Path(__file__).parent
            for _ in range(3):  # Check up to 3 levels up
                current = current.parent
                env_file = current / ".env"
                if env_file.exists():
                    break
            else:
                env_file = Path(".env")  # Default to current dir
        
        if env_file.exists():
            # Read only from .env file
            env_vars = dotenv_values(env_file)
            # Load only the secrets we care about
            secrets_to_load = [
                "ASANA_WORKSPACE_ID",
                "ASANA_PROJECT_ID",
            ]
            
            for key in secrets_to_load:
                value = env_vars.get(key)
                if value:
                    self._secrets[key] = value
    
    def get(self, key: str) -> Optional[str]:
        """Get a secret by key."""
        return self._secrets.get(key)
    
    def get_all(self) -> Dict[str, str]:
        """Get all secrets."""
        return self._secrets.copy()
    
    def format_for_prompt(self) -> str:
        """Format secrets for inclusion in prompts."""
        if not self._secrets:
            return ""
        
        lines = []
        for key, value in sorted(self._secrets.items()):
            lines.append(f"- {key}: {value}")
        
        return "\n".join(lines)

# Global instance
_secrets_store = SecretsStore()

