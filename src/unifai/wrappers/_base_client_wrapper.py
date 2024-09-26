from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError

yieldT = TypeVar("yieldT")
returnT = TypeVar("returnT")

class UnifAIExceptionConverter:
    # Convert Exceptions from Client Exception Types to UnifAI Exceptions for easier handling
    def convert_exception(self, exception: Exception) -> UnifAIError:
        raise NotImplementedError("This method must be implemented by the subclass")


    def run_func_convert_exceptions(self, func: Callable[..., returnT], *args, **kwargs) -> returnT:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            if isinstance(e, UnifAIError):
                raise e
            raise self.convert_exception(e) from e


    def run_func_convert_exceptions_generator(self, func: Callable[..., Generator[yieldT, None, returnT]], *args, **kwargs) ->  Generator[yieldT, None, returnT]:
        try:
            rval = yield from func(self, *args, **kwargs)
            return rval
        except Exception as e:
            if isinstance(e, UnifAIError):
                raise e
            raise self.convert_exception(e) from e 


def convert_exceptions(func: Callable[..., returnT]) -> Callable[..., returnT]:
    def wrapper(instance: UnifAIExceptionConverter, *args, **kwargs) -> returnT:
        return instance.run_func_convert_exceptions(func, *args, **kwargs)
    return wrapper


def convert_exceptions_generator(func: Callable[..., Generator[yieldT, None, returnT]]) -> Callable[..., Generator[yieldT, None, returnT]]:
    def wrapper(instance: UnifAIExceptionConverter, *args, **kwargs) -> Generator[yieldT, None, returnT]:
        return instance.run_func_convert_exceptions_generator(func, *args, **kwargs)
    return wrapper     
       
                
class BaseClientWrapper(UnifAIExceptionConverter):
    provider = "base"
    
    def import_client(self) -> Callable:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def init_client(self, **client_kwargs) -> Any:
        if client_kwargs:
            self.client_kwargs.update(client_kwargs)
        self._client = self.import_client()(**self.client_kwargs)
        return self._client

    def __init__(self, **client_kwargs):
        self._client = None
        self.client_kwargs = client_kwargs

    @property
    @convert_exceptions
    def client(self) -> Type:
        if self._client is None:
            return self.init_client(**self.client_kwargs)
            # TODO: ClientInitError
            # return self.run_func_convert_exceptions(self.init_client, **self.client_kwargs)
        return self._client      

