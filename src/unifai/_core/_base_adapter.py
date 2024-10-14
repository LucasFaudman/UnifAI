from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator

from ..types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from ..exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ._base_component import UnifAIComponent


class UnifAIAdapter(UnifAIComponent):
    provider = "base"
    
    def import_client(self) -> Callable:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def init_client(self, **client_kwargs) -> Any:
        if client_kwargs:
            self.client_kwargs.update(client_kwargs)

        # TODO: ClientInitError
        self._client = self.import_client()(**self.client_kwargs)
        return self._client    

    def __init__(self, **client_kwargs):
        self._client = None
        self.client_kwargs = client_kwargs

    @property
    def client(self) -> Type:
        if self._client is None:
            return self.init_client(**self.client_kwargs)
        return self._client      

