from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.document_chunkers._base_document_chunker import DocumentChunker

from ._base_client import BaseClient

class UnifAIDocumentChunkerClient(BaseClient):
    
    def get_document_chunker(self, provider: Optional[str] = None, **client_kwargs) -> "DocumentChunker":
        return self._get_component(provider, "document_chunker", **client_kwargs)