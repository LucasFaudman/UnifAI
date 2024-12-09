from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from cohere import ClientV2
from cohere.core import ApiError as CohereAPIError

from ...exceptions import UnifAIError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError
from .._base_components._base_adapter import UnifAIAdapter

T = TypeVar("T")

class CohereAdapter(UnifAIAdapter):
    provider = "cohere"
    client: ClientV2

    def import_client(self):
        from cohere import ClientV2
        return ClientV2
    
    
    def init_client(self, **client_kwargs) -> ClientV2:
        self.client_kwargs.update(client_kwargs)
        if not (api_key := self.client_kwargs.get("api_key")):
            raise ValueError("Cohere API key is required")
        self._client = self.import_client()(api_key) # Cohere ClientV2 does require an API key as a positional argument
        return self._client


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: CohereAPIError) -> UnifAIError:
        message = exception.body
        status_code = exception.status_code
        if status_code is not None:
                unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
                unifai_exception_type = UnknownAPIError
        return unifai_exception_type(
            message=message,
            status_code=status_code,
            original_exception=exception
        )        
