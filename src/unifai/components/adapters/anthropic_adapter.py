from anthropic import Anthropic
from anthropic import (
    AnthropicError,
    APIError as AnthropicAPIError,
    APIConnectionError as AnthropicAPIConnectionError,
    APITimeoutError as AnthropicAPITimeoutError,
    APIResponseValidationError as AnthropicAPIResponseValidationError,
    APIStatusError as AnthropicAPIStatusError,
    AuthenticationError as AnthropicAuthenticationError,
    BadRequestError as AnthropicBadRequestError,
    ConflictError as AnthropicConflictError,
    InternalServerError as AnthropicInternalServerError,
    NotFoundError as AnthropicNotFoundError,
    PermissionDeniedError as AnthropicPermissionDeniedError,
    RateLimitError as AnthropicRateLimitError,
    UnprocessableEntityError as AnthropicUnprocessableEntityError,
)

from ...exceptions import (
    UnifAIError,
    UnknownAPIError,
    APIResponseValidationError,
    STATUS_CODE_TO_EXCEPTION_MAP,
)
from .._base_components._base_adapter import UnifAIAdapter

class AnthropicAdapter(UnifAIAdapter):
    provider = "anthropic"
    client: Anthropic

    def import_client(self):
        from anthropic import Anthropic
        return Anthropic


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: AnthropicAPIError) -> UnifAIError:
        if isinstance(exception, AnthropicAPIResponseValidationError):
            return APIResponseValidationError(
                message=exception.message,
                status_code=exception.status_code, # Status code could be anything
                original_exception=exception,
            )
        
        if isinstance(exception, AnthropicAPIError):
            message = exception.message
            if isinstance(exception, AnthropicAPITimeoutError):
                status_code = 504
            elif isinstance(exception, AnthropicAPIConnectionError):
                status_code = 502
            else:
                status_code = getattr(exception, "status_code", -1)
        else:
            message = str(exception)
            status_code = 401 if "api_key" in message else -1
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message,
            status_code=status_code,
            original_exception=exception
        )
    

    # List Models
    def list_models(self) -> list[str]:
        claude_models = [
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
        ]
        return claude_models