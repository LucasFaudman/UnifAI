from __future__ import annotations

from typing import TYPE_CHECKING, Callable
from json import loads as json_loads, JSONDecodeError

if TYPE_CHECKING:
    from pinecone.grpc import PineconeGRPC
from pinecone.exceptions import PineconeException, PineconeApiException

from ...exceptions import UnifAIError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError
from ...components._base_component import UnifAIComponent
from ._base_adapter import UnifAIAdapter

class PineconeExceptionConverter(UnifAIComponent):
    def convert_exception(self, exception: PineconeException) -> UnifAIError:
        if not isinstance(exception, PineconeApiException):
            return UnifAIError(
                message=str(exception),
                original_exception=exception
            )
        status_code=exception.status
        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError
                    
        error_code = None
        if body := getattr(exception, "body", None):
            message = body
            try:
                decoded_body = json_loads(body)
                error = decoded_body["error"]
                message = error.get("message") or body
                error_code = error.get("code")
            except (JSONDecodeError, KeyError, AttributeError):
                pass
        else:
            message = str(exception)
        
        return unifai_exception_type(
            message=message,
            error_code=error_code,
            status_code=status_code,
            original_exception=exception,
        )   


class PineconeAdapter(UnifAIAdapter, PineconeExceptionConverter):
    client: PineconeGRPC
    default_embedding_provider = "pinecone"

    def import_client(self) -> Callable:
        from pinecone.grpc import PineconeGRPC
        return PineconeGRPC

