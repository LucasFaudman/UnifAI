from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_document_chunker import DocumentChunker
    from ..types.annotations import ComponentName, ProviderName
from ..configs.document_chunker_config import DocumentChunkerConfig
    
from ..utils import copy_init_from
from ._tokenizer_client import UnifAITokenizerClient

class UnifAIDocumentChunkerClient(UnifAITokenizerClient):

    def _get_document_chunker(
            self, 
            config_or_name: "DocumentChunkerConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "DocumentChunker":
        return self._get_component("document_chunker", config_or_name, init_kwargs)

    def document_chunker_from_config(
            self, 
            config: "DocumentChunkerConfig",
            **init_kwargs
            ):
        return self._get_document_chunker(config, **init_kwargs)
    
    def document_chunker_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ):            
        return self._get_document_chunker((provider, name), **init_kwargs)
    
    @copy_init_from(DocumentChunkerConfig.__init__)
    def document_chunker(self, *args, **kwargs):
        config = DocumentChunkerConfig(*args, **kwargs)
        return self._get_document_chunker(config)
    

    
# UnifAIDocumentChunkerClient.document_chunker