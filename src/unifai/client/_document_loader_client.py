from typing import TYPE_CHECKING, Any, Literal, Optional, Type, cast, ParamSpec, TypeVar, overload
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName, InputP
    from ..configs.document_loader_config import DocumentLoaderConfig, FileIODocumentLoaderConfig, SourceT, LoadedSourceT
    from ..components.document_loaders.text_file_loader import TextFileDocumentLoader
    from ..components._base_components._base_document_loader import DocumentLoader, FileIODocumentLoader, DocumentLoaderConfigT


from ._base_client import BaseClient

class UnifAIDocumentLoaderClient(BaseClient):

    @overload
    def document_loader(
            self,
            provider_config_or_name: Literal["text_file_loader"] | tuple[Literal["text_file_loader"], "ComponentName"],
            **init_kwargs
    ) -> "TextFileDocumentLoader":
        ...

    @overload
    def document_loader(
            self,
            provider_config_or_name: "FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT]",
            **init_kwargs
    ) -> "FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
        ...

    @overload
    def document_loader(
            self,
            provider_config_or_name: "DocumentLoaderConfig[InputP]",
            **init_kwargs
    ) -> "DocumentLoader[InputP]":
        ...

    @overload
    def document_loader(
            self,
            provider_config_or_name: "ProviderName | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
    ) -> "DocumentLoader":
        ...

    def document_loader(
            self, 
            provider_config_or_name: "ProviderName | DocumentLoaderConfig[InputP] | FileIODocumentLoaderConfig[InputP, SourceT, LoadedSourceT] | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "DocumentLoader[InputP] | FileIODocumentLoader[InputP, SourceT, LoadedSourceT]":
        return self._get_component("document_loader", provider_config_or_name, init_kwargs)