from typing import Optional
from ._base import UnifAIError


class APIError(UnifAIError):
    """Raised when an API call fails"""
    default_status_code = -1
    default_error_code = "api_error"

    def __init__(self, 
                 message: str, 
                 status_code: Optional[int] = None,
                 error_code: Optional[str] = None,
                 original_exception: Optional[Exception] = None
                 ):
        self.status_code = status_code or self.default_status_code
        self.error_code = error_code or self.default_error_code
        super().__init__(message, original_exception)

    # def __init__(self, 
    #              message: str, 
    #              status_code: Optional[int] = None,
    #              error_code: Optional[str] = None,
    #              original_exception: Optional[Exception] = None
    #              ):
    #     self.message = message
    #     self.status_code = status_code or self.default_status_code
    #     self.error_code = error_code or self.default_error_code
    #     super().__init__(original_exception)


class UnknownAPIError(APIError):
    """
    Raised when an unknown error occurs during an API call
    Cause: Unknown error occurred during API call.
    Solution: Contact the provider with the error message and response body for further assistance.
    """
    default_status_code = -1
    default_error_code = "unknown_error"        


class APIConnectionError(APIError):
    """
    Raised when there is an issue connecting to the API
    Cause: Issue connecting to AI provider API.
    Solution: Check your network settings, proxy configuration, SSL certificates, or firewall rules.
    """
    default_status_code = 502
    default_error_code = "connection_error"
    

class APITimeoutError(APIConnectionError):
    """
    Raised when a request to the API times out
    Cause: Request timed out.
    Solution: Retry your request after a brief wait and contact the AI provider if the issue persists.
    """
    default_status_code = 504
    default_error_code = "timeout_error"


class APIResponseValidationError(APIError):
    """
    Raised when the API response is invalid for the expected schema
    Cause: Data returned by API invalid for expected schema.
    Solution: Contact the provider with the error message and response body for further assistance.
    """
    default_status_code = 500
    default_error_code = "response_validation_error"   
    

class APIStatusError(APIError):
    """Raised when an API response has a status code of 4xx or 5xx."""


class AuthenticationError(APIStatusError):
    """
    Raised when an API key or token is invalid, expired, or revoked
    Cause: Your API key or token was invalid, expired, or revoked.
    Solution: Check your API key or token and make sure it is correct and active. You may need to generate a new one from your account dashboard.
    """
    default_status_code = 401
    default_error_code = "authentication_error"


class BadRequestError(APIStatusError):
    """
    Raised when a request is malformed or missing required parameters
    Cause: Your request was malformed or missing some required parameters, such as a token or an input.
    Solution: The error message should advise you on the specific error made. Check the documentation for the specific API method you are calling and make sure you are sending valid and complete parameters. You may also need to check the encoding, format, or size of your request data.
    """
    default_status_code = 400
    default_error_code = "bad_request_error"


class ConflictError(APIStatusError):
    """
    Raised when a resource was updated by another request
    Cause: The resource was updated by another request.
    Solution: Try to update the resource again and ensure no other requests are trying to update it.
    """
    default_status_code = 409
    default_error_code = "conflict_error"


class RequestTooLargeError(APIStatusError):
    """
    Raised when the request size exceeds the limit
    Cause: The request size exceeds the limit.
    Solution: Reduce the size of your request and try again.
    """
    default_status_code = 413
    default_error_code = "request_too_large_error"


class InternalServerError(APIStatusError):
    """
    Raised when there is an issue on the AI provider's side
    Cause: Issue on AI provider's side.
    Solution: Retry your request after a brief wait and contact the AI provider if the issue persists.
    """
    default_status_code = 500
    default_error_code = "internal_server_error"


class ServerOverloadedError(APIStatusError):
    """
    Raised when the server is overloaded and cannot process the request
    Cause: Server is overloaded and cannot process the request.
    Solution: Retry your request after a brief wait and contact the AI provider if the issue persists.
    """
    default_status_code = 503
    default_error_code = "server_overloaded_error"


class NotFoundError(APIStatusError):
    """
    Raised when a requested resource does not exist
    Cause: Requested resource does not exist.
    Solution: Ensure you are the correct resource identifier.
    """
    default_status_code = 404
    default_error_code = "not_found_error"


class PermissionDeniedError(APIStatusError):
    """
    Raised when access to a resource is denied
    Cause: You don't have access to the requested resource.
    Solution: Ensure you are using the correct API key, organization ID, and resource ID.
    """
    default_status_code = 403
    default_error_code = "permission_denied_error"


class RateLimitError(APIStatusError):
    """
    Raised when the rate limit has been exceeded
    Cause: You have hit your assigned rate limit.
    Solution: Pace your requests. Read more in our Rate limit guide.
    """
    default_status_code = 429
    default_error_code = "rate_limit_error"


class UnprocessableEntityError(APIStatusError):
    """
    Raised when a request cannot be processed despite having the correct format
    Cause: Unable to process the request despite the format being correct.
    Solution: Please try the request again.
    """
    default_status_code = 422
    default_error_code = "unprocessable_entity_error"


class TeapotError(APIStatusError):
    """
    Raised when the server is a teapot
    Cause: The server is a teapot.
    Solution: Drink some tea and try again later.
    """
    default_status_code = 418
    default_error_code = "teapot_error"


STATUS_CODE_TO_EXCEPTION_MAP: dict[int, type[APIError]] = {
    -1: UnknownAPIError,
    400: BadRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: RequestTooLargeError,
    418: TeapotError,
    422: UnprocessableEntityError,
    429: RateLimitError,
    500: InternalServerError,
    502: APIConnectionError,
    503: ServerOverloadedError,
    504: APITimeoutError,  
    505: APIConnectionError,    
}