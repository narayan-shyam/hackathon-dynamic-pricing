"""
Custom Exception Classes for ML Project Setup
"""


class ProjectSetupError(Exception):
    """Base exception for project setup errors"""
    pass


class DirectoryCreationError(ProjectSetupError):
    """Raised when directory creation fails"""
    pass


class ConfigurationError(ProjectSetupError):
    """Raised when configuration setup fails"""
    pass


class DependencyInstallError(ProjectSetupError):
    """Raised when dependency installation fails"""
    pass


class AzureKeyVaultError(ProjectSetupError):
    """Raised when Azure Key Vault operations fail"""
    pass


class ValidationError(ProjectSetupError):
    """Raised when data validation fails"""
    pass


class ModelError(ProjectSetupError):
    """Raised when model operations fail"""
    pass
