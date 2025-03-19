from typing import TYPE_CHECKING, Any, Literal, Optional, Type, cast, ParamSpec, TypeVar, overload
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName, InputP
    from ..configs.document_loader_config import DocumentLoaderConfig, FileIODocumentLoaderConfig, SourceT, LoadedSourceT
    from ..components.document_loaders.text_file_loader import TextFileDocumentLoader

from ..utils import copy_init_from
from ..components._base_components._base_document_loader import DocumentLoader, FileIODocumentLoader, DocumentLoaderConfigT
from ..configs.document_loader_config import DocumentLoaderConfig
from ._base_client import BaseClient

class UnifAIDocumentLoaderClient(BaseClient):

    # @overload
    # def _get_document_loader(
    #         self,
    #         config_or_name: Literal["text_file_loader"] | tuple[Literal["text_file_loader"], "ComponentName"],
    #         **init_kwargs
    # ) -> "TextFileDocumentLoader":
    #     ...

    # @overload
    # def _get_document_loader(
    #         self,
    #         config_or_name: "FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT]",
    #         **init_kwargs
    # ) -> "FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
    #     ...

    # @overload
    # def _get_document_loader(
    #         self,
    #         config_or_name: "DocumentLoaderConfig[InputP]",
    #         **init_kwargs
    # ) -> "DocumentLoader[InputP]":
    #     ...

    # @overload
    # def _get_document_loader(
    #         self,
    #         config_or_name: "ProviderName | tuple[ProviderName, ComponentName]" = "default",
    #         **init_kwargs
    # ) -> "DocumentLoader":
    #     ...

    def _get_document_loader(
            self, 
            config_or_name: "DocumentLoaderConfig[InputP] | FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT] | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "DocumentLoader[InputP] | FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
        return self._get_component("document_loader", config_or_name, init_kwargs)
    

    @overload
    def document_loader_from_config(
            self, 
            config: "FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT]",
            **init_kwargs
            ) -> "FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
        ...    
    @overload
    def document_loader_from_config(
            self, 
            config: "DocumentLoaderConfig[InputP]",
            **init_kwargs
            ) -> "DocumentLoader[InputP]":
        ...        
    def document_loader_from_config(
            self, 
            config: "DocumentLoaderConfig[InputP] | FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT]",
            **init_kwargs
            ) -> "DocumentLoader[InputP] | FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
        return self._get_document_loader(config, **init_kwargs)
    
    @overload
    def document_loader_from_name(
            self,
            provider: Literal["text_file_loader"],
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "TextFileDocumentLoader":
        ...
    @overload
    def document_loader_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "DocumentLoader":
        ...
    def document_loader_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ):
        return self._get_document_loader((provider, name), **init_kwargs)
    

    @copy_init_from(DocumentLoaderConfig.__init__)
    def document_loader(self, *args, **kwargs):
        config = DocumentLoaderConfig(*args, **kwargs)
        return self._get_document_loader(config)

    
# l = UnifAIDocumentLoaderClient().document_loader(provider="text_file_loader")