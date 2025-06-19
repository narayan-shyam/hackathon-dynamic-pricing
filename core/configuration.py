"""
Configuration Management with Environment-Specific Settings
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

from .exceptions import ConfigurationError
from .utilities import DataValidator
from .azure_integration import AzureKeyVaultManager

logger = logging.getLogger(__name__)


@dataclass
class ProjectConfig:
    """Project configuration with environment-specific settings"""
    project_name: str
    version: str
    environment: str
    azure_key_vault_url: Optional[str] = None
    azure_instrumentation_key: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    log_level: str = "INFO"
    max_workers: int = 4
    rate_limit_tokens: int = 100
    rate_limit_period: float = 60.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProjectConfig':
        return cls(**data)


class ConfigurationManager:
    """Environment-specific configuration management with Azure Key Vault integration"""
    
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.validator = DataValidator()
        self.key_vault_manager = None
    
    def setup_azure_integration(self, vault_url: str):
        """Initialize Azure Key Vault integration"""
        try:
            self.key_vault_manager = AzureKeyVaultManager(vault_url)
            logger.info("Azure Key Vault integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Azure Key Vault: {str(e)}")
            raise ConfigurationError(f"Azure Key Vault initialization failed: {str(e)}")
    
    def load_config(self, environment: str = "development") -> ProjectConfig:
        """Load environment-specific configuration with Key Vault secret integration"""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not self.validator.validate_file_path(str(config_file)):
            logger.warning(f"Config file not found: {config_file}. Using default configuration.")
            return self._get_default_config(environment)
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Merge with secrets from Key Vault if available
            if self.key_vault_manager:
                config_data = self._merge_vault_secrets(config_data)
            
            return ProjectConfig.from_dict(config_data)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise ConfigurationError(f"Configuration loading failed: {str(e)}")
    
    def _merge_vault_secrets(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge secrets from Azure Key Vault into configuration"""
        secret_mappings = {
            "azure_instrumentation_key": "azure-instrumentation-key",
            "mlflow_tracking_uri": "mlflow-tracking-uri"
        }
        
        for config_key, vault_key in secret_mappings.items():
            try:
                secret_value = self.key_vault_manager.get_secret(vault_key)
                if secret_value:
                    config_data[config_key] = secret_value
            except Exception as e:
                logger.warning(f"Failed to retrieve secret '{vault_key}': {str(e)}")
        
        return config_data
    
    def _get_default_config(self, environment: str) -> ProjectConfig:
        """Get default configuration for environment"""
        return ProjectConfig(
            project_name="ml-pricing-project",
            version="1.0.0",
            environment=environment,
            log_level="INFO" if environment == "production" else "DEBUG"
        )
