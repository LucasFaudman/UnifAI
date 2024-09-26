from typing import Optional
from ._base import UnifAIError

class ContentFilterError(UnifAIError):
    """Raised when input is rejected by the content filter"""

    def __init__(self, 
                 message: str, 
                 details: Optional[dict] = None,
                 original_exception: Optional[Exception] = None
                 ):
        super().__init__(message, original_exception)
        self.details = details or {}