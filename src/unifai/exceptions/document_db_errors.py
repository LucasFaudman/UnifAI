from ._base import UnifAIError
from .api_errors import APIError, NotFoundError

class DocumentDBError(UnifAIError):
    """Base class for all DocumentDB errors"""

class DocumentDBAPIError(APIError, DocumentDBError):
    """Base class for all DocumentDB API errors"""

class DocumentNotFoundError(NotFoundError, DocumentDBAPIError):
    """Raised when the specified document does not exist."""

class DocumentReadError(DocumentDBAPIError):
    """Raised when the document get operation fails."""

class DocumentWriteError(DocumentDBAPIError):
    """Raised when the document set operation fails."""

class DocumentDeleteError(DocumentDBAPIError):
    """Raised when the document delete operation fails."""






