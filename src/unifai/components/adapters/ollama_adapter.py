from typing import Optional, Union, Sequence, Any, Literal, Mapping,  Iterator, Iterable, Generator, Collection

from ollama import Client as OllamaClient
from ollama._types import (
    RequestError as OllamaRequestError,
    ResponseError as OllamaResponseError,
)
from httpx import NetworkError, TimeoutException, HTTPError, RemoteProtocolError, TransportError

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
    ProviderUnsupportedFeatureError,
)


from .._base_components._base_adapter import UnifAIAdapter

class OllamaAdapter(UnifAIAdapter):
    provider = "ollama"
    client: OllamaClient


    def import_client(self):
        from ollama import Client as OllamaClient
        from ollama._types import Tool, ToolFunction, Parameters, Property, Message
        return OllamaClient


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def _convert_exception(self, exception: OllamaRequestError|OllamaResponseError|NetworkError|TimeoutError) -> UnifAIError:
        if isinstance(exception, OllamaRequestError):            
            message = exception.error
            status_code = 400
        elif isinstance(exception, OllamaResponseError):
            message = exception.error
            status_code = exception.status_code
        elif isinstance(exception, TimeoutException):
            message = exception.args[0]
            status_code = 504            
        elif isinstance(exception, TransportError):
            message = exception.args[0]
            status_code = 502 if '[Errno 65]' not in message else 504 # 502 is Bad Gateway, 504 is Gateway Timeout
        else:
            status_code = -1
            message = str(exception)
        
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=message, 
            status_code=status_code,
            original_exception=exception
        )

    # List Models
    def _list_models(self) -> list[str]:
        return [model_name for model_dict in self.client.list()["models"] if (model_name := model_dict.get("name"))]
    