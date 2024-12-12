from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Callable, Iterator, Iterable, Generator, Generic
from importlib import import_module

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ._base_component import UnifAIComponent, ConfigT

from ...utils import combine_dicts
from ...configs._base_configs import ComponentConfig

class UnifAIAdapter(UnifAIComponent[ConfigT]):

    def _setup(self) -> None:
        self._client = None

    def import_client(self) -> Callable:
        raise NotImplementedError("This method must be implemented by the subclass")    
            
    def init_client(self, **init_kwargs) -> Any:
        if init_kwargs:
            self.init_kwargs.update(init_kwargs)
        # TODO: ClientInitError
        self._client = self.import_client()(**self.init_kwargs)
        return self._client    

    @property
    def client(self) -> Type:
        if self._client is None:
            return self.init_client()
        return self._client