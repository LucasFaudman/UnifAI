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

class OpenAIAdapter(UnifAIAdapter):
    provider = "openai"
    client: OpenAI
 

    def import_client(self):
        from openai import OpenAI
        return OpenAI
    
    
    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: OpenAIAPIError) -> UnifAIError:
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
        

    # List Models
    def list_models(self) -> list[str]:
        return [model.id for model in self.client.models.list()]

