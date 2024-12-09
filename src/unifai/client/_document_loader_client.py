from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_document_loader import DocumentLoader
    from ..types.annotations import ComponentName, ProviderName
    from ..configs.document_loader_config import DocumentLoaderConfig

from ._base_client import BaseClient

class UnifAIDocumentLoaderClient(BaseClient):
    
    def get_document_loader(
            self, 
            provider_config_or_name: "ProviderName | DocumentLoaderConfig | tuple[ProviderName, ComponentName]" = "default",     
            **init_kwargs
            ) -> "DocumentLoader":
        return self._get_component("document_loader", provider_config_or_name, init_kwargs)