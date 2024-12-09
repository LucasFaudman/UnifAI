from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..components._base_components._base_vector_db_collection import VectorDBCollection
    from ..components._base_components._base_vector_db import VectorDB
    from ..components._base_components._base_document_db import DocumentDB
    from ..configs.vector_db_config import VectorDBConfig

from ._embed_client import UnifAIEmbedClient
from ._document_db_client import UnifAIDocumentDBClient

class UnifAIVectorDBClient(UnifAIEmbedClient, UnifAIDocumentDBClient):
    def get_vector_db(
            self, 
            provider_config_or_name: "ProviderName | VectorDBConfig | tuple[ProviderName, ComponentName]" = "default",          
            **client_kwargs
            ) -> "VectorDB":
        return self._get_component("vector_db", provider_config_or_name, client_kwargs)

    
