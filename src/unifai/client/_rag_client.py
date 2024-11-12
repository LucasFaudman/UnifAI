from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.retrievers._base_vector_db_index import Retriever, VectorDBIndex
    from ..components.retrievers._base_vector_db_client import VectorDBClient
    from ..components.rerankers._base_reranker import Reranker, VectorDBQueryResult

from ..type_conversions import standardize_configs, standardize_config
from ._base_client import BaseClient, Config, ProviderConfig, Path
from ._vector_db_client import UnifAIVectorDBClient
from ._rerank_client import UnifAIRerankClient

from .rag_engine import RAGEngine, RAGConfig

class UnifAIRAGClient(UnifAIVectorDBClient, UnifAIRerankClient):
    def __init__(
        self,
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_providers: Optional[dict[str, str]] = None, 
        rag_configs: Optional[dict[str, RAGConfig|dict[str, Any]]] = None,
        **kwargs
    ):
        BaseClient.__init__(
            self,
            config_obj_dict_or_path=config_obj_dict_or_path,
            provider_configs=provider_configs,
            api_keys=api_keys,
            default_providers=default_providers,
            **kwargs
        )
        self._init_rag_configs(rag_configs)

    def _init_rag_configs(self, rag_configs: Optional[dict[str, RAGConfig|dict[str, Any]]] = None) -> None:
        self._rag_configs: dict[str, RAGConfig] = {}
        if rag_configs:
            self.register_rag_configs(rag_configs)

    def _cleanup_rag_configs(self) -> None:
        self._rag_configs.clear()

    def cleanup(self) -> None:
        BaseClient.cleanup(self)
        self._cleanup_rag_configs()

    def register_rag_config(self, name: str, rag_config: RAGConfig|dict) -> None:
        rag_config = standardize_config(rag_config, RAGConfig)
        self._rag_configs[name] = rag_config

    def register_rag_configs(self, rag_configs: dict[str, RAGConfig|dict[str, Any]]) -> None:
        for name, rag_config in rag_configs.items():
            self.register_rag_config(name, rag_config)

    def get_rag_config(self, name: str) -> RAGConfig:
        if (rag_config := self._rag_configs.get(name)) is None:
            raise KeyError(f"RAG config '{name}' not found in self.rag_configs")
        return rag_config

    def get_rag_engine(
            self,
            config_obj_or_name: Optional[RAGConfig | str] = None,
            retriever_instance: Optional["Retriever|VectorDBIndex"] = None,
            reranker_instance: Optional["Reranker"] = None,
            **kwargs
    ) -> RAGEngine:
        if isinstance(config_obj_or_name, str):
            rag_config = self.get_rag_config(config_obj_or_name)            
        elif isinstance(config_obj_or_name, RAGConfig):
            rag_config = config_obj_or_name.model_copy(update=kwargs, deep=True) if kwargs else config_obj_or_name
        elif config_obj_or_name is None:
            rag_config = RAGConfig(**kwargs)
        else:
            raise ValueError(
                f"Invalid rag_config: {config_obj_or_name}. Must be a RAGConfig object or a string (name of a registered RAGConfig)")

        if retriever_instance is None:  
            retriever_instance = self.get_or_create_index(
                name=rag_config.index_name,
                vector_db_provider=rag_config.vector_db_provider,
                document_db_provider=rag_config.document_db_provider,
                embedding_provider=rag_config.embedding_provider,
                embedding_model=rag_config.embedding_model,
                dimensions=rag_config.embedding_dimensions,
                distance_metric=rag_config.embedding_distance_metric,
            )

        if reranker_instance is None and rag_config.rerank_provider:
            reranker_instance = self.get_reranker(rag_config.rerank_provider)
        
        return RAGEngine(
            config=rag_config,
            retriever=retriever_instance,
            reranker=reranker_instance
        )  
    
