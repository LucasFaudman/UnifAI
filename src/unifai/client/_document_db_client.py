from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.document_dbs._base_document_db import DocumentDB

from ._base_client import BaseClient

class UnifAIDocumentDBClient(BaseClient):
    
    def get_document_db(self, provider: Optional[str] = None, **client_kwargs) -> "DocumentDB":
        provider = provider or self.config.default_providers["document_db"]
        return self.get_component(provider, "document_db", **client_kwargs)
