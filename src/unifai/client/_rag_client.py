from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..configs.unifai_config import UnifAIConfig
    from ..components.ragpipes import RAGPipe, RAGPrompter
    from pathlib import Path

from ..type_conversions import standardize_config

from ._base_client import BaseClient
from ._vector_db_client import UnifAIVectorDBClient
from ._reranker_client import UnifAIRerankClient
from ._document_chunker_client import UnifAIDocumentChunkerClient
from ._document_loader_client import UnifAIDocumentLoaderClient

from ..configs.rag_config import RAGConfig, RAGPrompterConfig, InputP

class UnifAIRAGClient(UnifAIVectorDBClient, UnifAIRerankClient, UnifAIDocumentChunkerClient, UnifAIDocumentLoaderClient):
    
    def ragpipe(
            self, 
            provider_config_or_name: "ProviderName | RAGConfig[InputP] | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "RAGPipe[InputP]":
        return self._get_component("ragpipe", provider_config_or_name, init_kwargs)
    

    def rag_prompter(
            self, 
            provider_config_or_name: "ProviderName | RAGPrompterConfig[InputP] | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "RAGPrompter[InputP]":
        return self._get_component("rag_prompter", provider_config_or_name, init_kwargs)
            
    
    def configure(
        self,
        config: Optional["UnifAIConfig|dict[str, Any]|str|Path"] = None,
        api_keys: Optional["dict[ProviderName, str]"] = None,
        **kwargs
    ) -> None:
        BaseClient.configure(self, config, api_keys, **kwargs)
        self._init_rag_configs()

    def _init_rag_configs(self) -> None:
        self._rag_configs: dict[str, RAGConfig] = {}
        if self.config.rag_configs:
            self.register_rag_configs(*self.config.rag_configs)

    def register_rag_configs(self, *rag_configs: RAGConfig|dict) -> None:
        for _rag_config in rag_configs:
            _rag_config = standardize_config(_rag_config, RAGConfig)
            self._rag_configs[_rag_config.name] = _rag_config

    def get_rag_config(self, name: str) -> RAGConfig:
        if (rag_config := self._rag_configs.get(name)) is None:
            raise KeyError(f"RAG config '{name}' not found in self.rag_configs")
        return rag_config
        
    def _cleanup_rag_configs(self) -> None:
        self._rag_configs.clear()

    def cleanup(self) -> None:
        BaseClient.cleanup(self)
        self._cleanup_rag_configs()


      