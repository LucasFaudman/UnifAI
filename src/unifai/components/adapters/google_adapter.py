from typing import Any

from google.api_core.exceptions import (
    GoogleAPICallError,
)
from google.generativeai.types import (
    GenerateContentResponse,
    ContentType,
    BlockedPromptException,
    StopCandidateException,
    IncompleteIterationError,
    BrokenResponseError,
) 

from ...exceptions import (
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
    ModelUnsupportedFeatureError
)

from .._base_components._base_adapter import UnifAIAdapter

class GoogleAdapter(UnifAIAdapter):
    provider = "google"

    def import_client(self):
        import google.generativeai as genai
        return genai

    def init_client(self, **init_kwargs):
        if init_kwargs:
            self.init_kwargs.update(init_kwargs)        
        self._client = self.import_client()
        self._client.configure(**self.init_kwargs)
        return self._client

    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def _convert_exception(self, exception: Exception) -> UnifAIError:      
        if isinstance(exception, GoogleAPICallError):
            message = exception.message
            status_code = exception.code

            if status_code == 400:
                if "API key" in message:                    
                    status_code = 401 # BadRequestError to AuthenticationError
                elif "unexpected model" in message:
                    status_code = 404 # BadRequestError to NotFoundError
            elif status_code == 403 and "authentication" in message:
                status_code = 401 # PermissionDeniedError to AuthenticationError                                      
        else:
            message = str(exception)
            status_code = None

        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError
        return unifai_exception_type(message=message, status_code=status_code, original_exception=exception)

    def format_model_name(self, model: str) -> str:
        if model.startswith("models/"):
            return model
        return f"models/{model}"
    
    # List Models
    def _list_models(self) -> list[str]:
        return [model.name[7:] for model in self.client.list_models()]    