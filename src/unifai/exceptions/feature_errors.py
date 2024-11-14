from typing import Optional
from ._base import UnifAIError

class UnsupportedFeatureError(UnifAIError):
    """Raised when a feature is not supported by a provider or model"""

class ProviderUnsupportedFeatureError(UnsupportedFeatureError):
    """Raised when a feature is not supported by a provider"""

class ModelUnsupportedFeatureError(UnsupportedFeatureError):
    """Raised when a feature is not supported by a model but is supported by the provider"""
