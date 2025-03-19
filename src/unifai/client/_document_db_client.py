from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_document_db import DocumentDB
    from ..types.annotations import ComponentName, ProviderName

from ..utils import copy_init_from
from ..configs.document_db_config import DocumentDBConfig
from ._base_client import BaseClient

class UnifAIDocumentDBClient(BaseClient):

    def _get_document_db(
            self, 
            config_or_name: "DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default", 
            **init_kwargs
            ) -> "DocumentDB":
        return self._get_component("document_db", config_or_name, init_kwargs)

    def document_db_from_config(
            self, 
            config: "DocumentDBConfig", 
            **init_kwargs
            ):
        return self._get_document_db(config, **init_kwargs)
    
    def document_db_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ):
        return self._get_document_db((provider, name), **init_kwargs)
    
    @copy_init_from(DocumentDBConfig.__init__)
    def document_db(self, *args, **kwargs):
        config = DocumentDBConfig(*args, **kwargs)
        return self._get_document_db(config)

