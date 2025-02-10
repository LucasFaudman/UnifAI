from typing import Optional
from openai import OpenAI
from openai import (
    OpenAIError,
    APIError as OpenAIAPIError,
    APIConnectionError as OpenAIAPIConnectionError,
    APITimeoutError as OpenAIAPITimeoutError,
    APIResponseValidationError as OpenAIAPIResponseValidationError,
    APIStatusError as OpenAIAPIStatusError,
    AuthenticationError as OpenAIAuthenticationError,
    BadRequestError as OpenAIBadRequestError,
    ConflictError as OpenAIConflictError,
    InternalServerError as OpenAIInternalServerError,
    NotFoundError as OpenAINotFoundError,
    PermissionDeniedError as OpenAIPermissionDeniedError,
    RateLimitError as OpenAIRateLimitError,
    UnprocessableEntityError as OpenAIUnprocessableEntityError,
)


from unifai.exceptions import (
    UnifAIError,
    APIError,
    UnknownAPIError,
    APIConnectionError,
    APITimeoutError,
    APIResponseValidationError,
    APIStatusError,
    AuthenticationError,
    BadRequestError,
    ConflictError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
    STATUS_CODE_TO_EXCEPTION_MAP,
)


from .._base_components._base_adapter import UnifAIAdapter

class TempBaseURL:
    """
    Temporarily change the base URL of the client to the provided base URL and then reset it after exiting the context

    Nvidia API requires different base URLs for different models unlike OpenAI which uses the same base URL for all models and endpoints

    Args:
        client (OpenAI): The OpenAI client
        base_url (Optional[str]): The new base URL to use
        default_base_url (str): The default base URL to reset to after exiting the context 
    """

    def __init__(self, 
                 client: OpenAI, 
                 base_url: Optional[str], 
                 default_base_url: str
                 ):
        self.client = client
        self.base_url = base_url
        self.default_base_url = default_base_url

    def __enter__(self):
        if self.base_url:
            self.client.base_url = self.base_url

    def __exit__(self, exc_type, exc_value, traceback):
        if self.base_url:
            self.client.base_url = self.default_base_url

class OpenAIAdapter(UnifAIAdapter):
    provider = "openai"
    client: OpenAI
    default_base_url: Optional[str] = None
    default_headers: Optional[dict[str, str]] = None

    def import_client(self):
        from openai import OpenAI
        return OpenAI
    
    
    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def _convert_exception(self, exception: OpenAIAPIError) -> UnifAIError:
        if isinstance(exception, OpenAIAPIResponseValidationError):
            return APIResponseValidationError(
                message=exception.message,
                status_code=exception.status_code, # Status code could be anything
                error_code=exception.code,
                original_exception=exception,
            )
        
        message = getattr(exception, "message", str(exception))
        error_code = getattr(exception, "code", None)
        if isinstance(exception, OpenAIAPITimeoutError):
            status_code = 504            
        elif isinstance(exception, OpenAIAPIConnectionError):                
            status_code = 502
        elif isinstance(exception, OpenAIAPIStatusError):
            status_code = getattr(exception, "status_code", -1)
        else:
            status_code = 401 if "api_key" in message else getattr(exception, "status_code", -1)
        #TODO model does not support tool calls, images, etc feature errors

        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message, 
            status_code=status_code,
            error_code=error_code, 
            original_exception=exception
        )
        
    def init_client(self, **init_kwargs):
        if self.default_headers and "default_headers" not in init_kwargs:
            init_kwargs["default_headers"] = self.default_headers

        if self.default_base_url and "base_url" not in init_kwargs:
            # Add the Nvidia base URL if not provided since the default is OpenAI
            self._client = super().init_client(**init_kwargs)
            self._client.base_url = self.default_base_url
        else:
            self._client = super().init_client(**init_kwargs)
        return self._client

    # List Models
    def _list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list()]

