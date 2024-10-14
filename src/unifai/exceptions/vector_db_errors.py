from ._base import UnifAIError
from .api_errors import APIError
from .embedding_errors import EmbeddingDimensionsError

class VectorDBError(UnifAIError):
    """Base class for all VectorDB errors"""


class VectorDBAPIError(APIError, VectorDBError):
    """Base class for all VectorDB API errors"""


class IndexNotFoundError(VectorDBAPIError):
    """Raised when the specified index does not exist. Use get_or_create_index instead of get_index to avoid this error."""


class IndexAlreadyExistsError(VectorDBAPIError):
    """Raised when trying to create an index with the same name as an existing index."""


class InvalidQueryError(VectorDBAPIError):
    """Raised when the query is invalid."""


class DimensionsMismatchError(EmbeddingDimensionsError, VectorDBAPIError):
    """Raised when the dimensions of the input embeddings(s) do not match the dimensions of the index."""


class DocumentDBError(VectorDBError):
    """Base class for all DocumentDB errors"""


class DocumentDBAPIError(VectorDBAPIError, DocumentDBError):
    """Base class for all DocumentDB API errors"""


class DocumentNotFoundError(DocumentDBAPIError):
    """Raised when the specified document does not exist."""


class DocumentReadError(DocumentDBAPIError):
    """Raised when the document get operation fails."""


class DocumentWriteError(DocumentDBAPIError):
    """Raised when the document set operation fails."""


class DocumentDeleteError(DocumentDBAPIError):
    """Raised when the document get operation fails."""







