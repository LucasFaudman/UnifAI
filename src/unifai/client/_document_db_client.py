from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_document_db import DocumentDB
    from ..types.annotations import ComponentName, ProviderName
    from ..configs.document_db_config import DocumentDBConfig

from ._base_client import BaseClient

class UnifAIDocumentDBClient(BaseClient):
    
    def document_db(
            self, 
            provider_config_or_name: "ProviderName | DocumentDBConfig | tuple[ProviderName, ComponentName]" = "default",        
            **init_kwargs
            ) -> "DocumentDB":
        return self._get_component("document_db", provider_config_or_name, init_kwargs)
