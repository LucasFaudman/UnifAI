from ._base import UnifAIError
from .api_errors import (
    APIError,
    UnknownAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    RequestTooLargeError,
    InternalServerError,
    ServerOverloadedError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    TeapotError,
    STATUS_CODE_TO_EXCEPTION_MAP,
)
from .document_db_errors import (
    DocumentDBError,
    DocumentDBAPIError,
    DocumentReadError,
    DocumentWriteError,
    DocumentDeleteError,
    DocumentNotFoundError,    
)
from .embedding_errors import (
    EmbeddingError,
    EmbeddingAPIError,
    EmbeddingDimensionsError,
    EmbeddingTokenLimitExceededError
)
from .feature_errors import (
    UnsupportedFeatureError,
    ProviderUnsupportedFeatureError,
    ModelUnsupportedFeatureError,
)
from .output_parser_errors import (
    OutputParserError,
)
from .tool_errors import (
    ToolError,
    ToolValidationError,
    ToolNotFoundError,
    ToolCallError,
    ToolCallArgumentValidationError,
    ToolCallableNotFoundError,
    ToolCallExecutionError,
    ToolChoiceError,
    ToolChoiceErrorRetriesExceeded,
)
from .usage_errors import (
    ContentFilterError,
    TokenLimitExceededError,
)
from .vector_db_errors import (
    VectorDBError,
    VectorDBAPIError,
    IndexNotFoundError,
    IndexAlreadyExistsError,
    InvalidQueryError,
    DimensionsMismatchError,
)


__all__ = [
    "UnifAIError",
    "APIError",
    "UnknownAPIError",
    "APIConnectionError",
    "APITimeoutError",
    "APIResponseValidationError",
    "APIStatusError",
    "AuthenticationError",
    "BadRequestError",
    "ConflictError",
    "RequestTooLargeError",
    "InternalServerError",
    "ServerOverloadedError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "UnprocessableEntityError",
    "TeapotError",
    "STATUS_CODE_TO_EXCEPTION_MAP",

    "EmbeddingError",
    "EmbeddingAPIError",
    "EmbeddingDimensionsError",
    "EmbeddingTokenLimitExceededError",

    "UnsupportedFeatureError",
    "ProviderUnsupportedFeatureError",
    "ModelUnsupportedFeatureError",

    "ToolError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolCallError",
    "ToolCallArgumentValidationError",
    "ToolCallableNotFoundError",
    "ToolCallExecutionError",
    "ToolChoiceError",
    "ToolChoiceErrorRetriesExceeded",

    "ContentFilterError",
    "TokenLimitExceededError",

    "VectorDBError",
    "VectorDBAPIError",
    "IndexNotFoundError",
    "IndexAlreadyExistsError",
    "InvalidQueryError",
    "DimensionsMismatchError",

    "DocumentDBError",
    "DocumentDBAPIError",
    "DocumentReadError",
    "DocumentWriteError",
    "DocumentDeleteError",
    "DocumentNotFoundError",

    "OutputParserError",
    
]
