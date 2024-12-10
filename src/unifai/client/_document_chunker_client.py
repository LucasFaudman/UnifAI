from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_document_chunker import DocumentChunker
    from ..types.annotations import ComponentName, ProviderName
    from ..configs.document_chunker_config import DocumentChunkerConfig
    

from ._tokenizer_client import UnifAITokenizerClient

class UnifAIDocumentChunkerClient(UnifAITokenizerClient):
    
    def document_chunker(
            self, 
            provider_config_or_name: "ProviderName | DocumentChunkerConfig | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "DocumentChunker":
        return self._get_component("document_chunker", provider_config_or_name, init_kwargs)