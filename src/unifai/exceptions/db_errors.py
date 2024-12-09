from ._base import UnifAIError
from .api_errors import APIError, NotFoundError
from .embedding_errors import EmbeddingDimensionsError

class DBError(UnifAIError):
    """Base class for all VectorDB errors"""

class DBAPIError(APIError, DBError):
    """Base class for all VectorDB API errors"""

class CollectionNotFoundError(DBAPIError, NotFoundError):
    """Raised when the specified index does not exist. Use get_or_create_index instead of get_index to avoid this error."""

class CollectionAlreadyExistsError(DBAPIError):
    """Raised when trying to create an index with the same name as an existing index."""

class DocumentNotFoundError(DBAPIError, NotFoundError):
    """Raised when the specified document does not exist."""

class DocumentAlreadyExistsError(DBAPIError):
    """Raised when trying to add a document with the same ID as an existing document."""

class InvalidQueryError(DBAPIError):
    """Raised when the query is invalid."""

class DimensionsMismatchError(EmbeddingDimensionsError, DBAPIError):
    """Raised when the dimensions of the input embeddings(s) do not match the dimensions of the index."""