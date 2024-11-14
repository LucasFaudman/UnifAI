from ._base import UnifAIError
from .api_errors import APIError, BadRequestError
from .usage_errors import TokenLimitExceededError

class EmbeddingError(UnifAIError):
    """Base class for all embedding errors"""

class EmbeddingAPIError(APIError, EmbeddingError):
    """Raised when an embedding API call fails"""

class EmbeddingDimensionsError(BadRequestError, EmbeddingAPIError):
    """Raised when the requested dimensions are invalid. Either larger than the model's output dimensions or less than 1."""

class EmbeddingTokenLimitExceededError(TokenLimitExceededError, EmbeddingError):
    """Raised when the token limit is exceeded. (The size of the input document(s) exceeds the model's token limit.)"""