from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..types.annotations import ComponentName, ProviderName
    from ..configs.rag_config import RAGConfig
    from ..components.ragpipe import RAGPipe

from ..type_conversions import standardize_configs, standardize_config
from ._base_client import BaseClient
from ._vector_db_client import UnifAIVectorDBClient
from ._rerank_client import UnifAIRerankClient
from ._document_chunker_client import UnifAIDocumentChunkerClient
from ._document_loader_client import UnifAIDocumentLoaderClient

from ..configs.rag_config import RAGConfig

class UnifAIRAGClient(UnifAIVectorDBClient, UnifAIRerankClient, UnifAIDocumentChunkerClient, UnifAIDocumentLoaderClient):
    
    def ragpipe(
            self, 
            provider_config_or_name: "ProviderName | RAGConfig | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "RAGPipe":
        return self._get_component("ragpipe", provider_config_or_name, init_kwargs)
    
    # def __init__(
    #     self,
    #     config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
    #     provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
    #     api_keys: Optional[dict[str, str]] = None,
    #     default_providers: Optional[dict[str, str]] = None, 
    #     rag_configs: Optional[dict[str, RAGPromptConfig|dict[str, Any]]] = None,
    #     **kwargs
    # ):
    #     BaseClient.__init__(
    #         self,
    #         config=config_obj_dict_or_path,
    #         provider_configs=provider_configs,
    #         api_keys=api_keys,
    #         default_providers=default_providers,
    #         **kwargs
    #     )
    #     self._init_rag_configs(rag_configs)

    # def _init_rag_configs(self, rag_configs: Optional[dict[str, RAGPromptConfig|dict[str, Any]]] = None) -> None:
    #     self._rag_configs: dict[str, RAGPromptConfig] = {}
    #     if rag_configs:
    #         self.register_rag_configs(rag_configs)

    # def _cleanup_rag_configs(self) -> None:
    #     self._rag_configs.clear()

    # def cleanup(self) -> None:
    #     BaseClient.cleanup(self)
    #     self._cleanup_rag_configs()

    # def register_rag_config(self, name: str, rag_config: RAGPromptConfig|dict) -> None:
    #     rag_config = standardize_config(rag_config, RAGPromptConfig)
    #     self._rag_configs[name] = rag_config

    # def register_rag_configs(self, rag_configs: dict[str, RAGPromptConfig|dict[str, Any]]) -> None:
    #     for name, rag_config in rag_configs.items():
    #         self.register_rag_config(name, rag_config)

    # def get_rag_config(self, name: str) -> RAGPromptConfig:
    #     if (rag_config := self._rag_configs.get(name)) is None:
    #         raise KeyError(f"RAG config '{name}' not found in self.rag_configs")
    #     return rag_config
        