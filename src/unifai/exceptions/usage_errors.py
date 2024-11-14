from .api_errors import APIError

class TokenLimitExceededError(APIError):
    """Raised when the token limit is exceeded"""

class ContentFilterError(APIError):
    """Raised when input is rejected by the content filter"""

