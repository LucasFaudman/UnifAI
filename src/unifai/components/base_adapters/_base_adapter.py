from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator
from importlib import import_module

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError
from .._base_component import UnifAIComponent


class UnifAIAdapter(UnifAIComponent):
    provider = "base"

    def __init__(self, **client_kwargs):
        self._client = None
        self.client_kwargs = client_kwargs    
    
    def import_client(self) -> Callable:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def init_client(self, **client_kwargs) -> Any:
        if client_kwargs:
            self.client_kwargs.update(client_kwargs)

        # TODO: ClientInitError
        self._client = self.import_client()(**self.client_kwargs)
        return self._client    

    @property
    def client(self) -> Type:
        if self._client is None:
            return self.init_client(**self.client_kwargs)
        return self._client      

    def lazy_import(self, module_name: str) -> Any:
        module_name, *submodules = module_name.split(".")
        if not (module := globals().get(module_name)):
            # TODO - ClientImportError
            module = import_module(module_name)
            globals()[module_name] = module
                    
        for submodule in submodules:
            module = getattr(module, submodule)        
        return module    