from ._base import (
    UnifAIError, 
    UnknownUnifAIError
)
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
from .db_errors import (
    DBError,
    DBAPIError,
    CollectionNotFoundError,
    CollectionAlreadyExistsError,
    DocumentNotFoundError,
    DocumentAlreadyExistsError,
    InvalidQueryError,
    DimensionsMismatchError,
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
from .tokenizer_errors import (
    TokenizerError,
    TokenizerDisallowedSpecialTokenError,
    TokenizerVocabError,
)
from .usage_errors import (
    ContentFilterError,
    TokenLimitExceededError,
)
__all__ = [
    "UnifAIError",
    "UnknownUnifAIError",

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

    "OutputParserError",

    "ToolError",
    "ToolValidationError",
    "ToolNotFoundError",
    "ToolCallError",
    "ToolCallArgumentValidationError",
    "ToolCallableNotFoundError",
    "ToolCallExecutionError",
    "ToolChoiceError",
    "ToolChoiceErrorRetriesExceeded",

    "TokenizerError",
    "TokenizerDisallowedSpecialTokenError",
    "TokenizerVocabError",

    "ContentFilterError",
    "TokenLimitExceededError",

    "DBError",
    "DBAPIError",
    "CollectionNotFoundError",
    "CollectionAlreadyExistsError",
    "InvalidQueryError",
    "DimensionsMismatchError",    
]
