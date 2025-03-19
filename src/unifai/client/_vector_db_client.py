from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..components._base_components._base_vector_db_collection import VectorDBCollection
    from ..components._base_components._base_vector_db import VectorDB
    from ..components._base_components._base_document_db import DocumentDB

from ..utils import copy_init_from
from ..configs.vector_db_config import VectorDBConfig
from ._embedder_client import UnifAIEmbedClient
from ._document_db_client import UnifAIDocumentDBClient

class UnifAIVectorDBClient(UnifAIEmbedClient, UnifAIDocumentDBClient):

    def _get_vector_db(
            self, 
            config_or_name: "VectorDBConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",          
            **init_kwargs
            ) -> "VectorDB":
        return self._get_component("vector_db", config_or_name, init_kwargs)

    def vector_db_from_config(
            self, 
            config: "VectorDBConfig", 
            **init_kwargs
            ) -> "VectorDB":
        return self._get_vector_db(config, **init_kwargs)
    
    def vector_db_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "VectorDB":
        return self._get_vector_db((provider, name), **init_kwargs)
    
    @copy_init_from(VectorDBConfig.__init__)
    def vector_db(self, *args, **kwargs) -> "VectorDB":
        config = VectorDBConfig(*args, **kwargs)
        return self._get_vector_db(config)

    
