from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self

from ..types.annotations import ComponentName, ModelName, ProviderName, CollectionName, EmbeddingTaskTypeInput
from ._base_configs import BaseDBCollectionConfig, BaseDBConfig
from .document_db_config import DocumentDBConfig, DocumentDBCollectionConfig
from .embedder_config import EmbedderConfig

class VectorDBCollectionConfig(BaseDBCollectionConfig):
    component_type = "vector_db_collection"
    
    name: CollectionName = "default_collection"
    dimensions: Optional[int] = None
    distance_metric: Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"] = "cosine"

    embedder: EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName] = "default"
    embedding_model: Optional[ModelName] = None
    embed_document_task_type: EmbeddingTaskTypeInput = "retrieval_document"
    embed_query_task_type: EmbeddingTaskTypeInput = "retrieval_query"
    
    document_db_collection: Optional[DocumentDBCollectionConfig | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal[
        "add",
        "update",
        "upsert",
        "delete",
        "get", 
        "query",
        "query_many",
        "embed",     
        ], dict[str, Any]]] = None
    

class VectorDBConfig(BaseDBConfig):
    component_type = "vector_db"

    default_collection: VectorDBCollectionConfig = VectorDBCollectionConfig(provider="default", name="default_collection")
    document_db: Optional[DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal[
        "create_collection", 
        "get_collection",
        "list_collections",
        "count_collections",
        "delete_collection",
        "add",
        "update",
        "upsert",
        "delete",
        "get",
        "query",
        "query_many",        
        ], dict[str, Any]]] = None

VectorDBCollectionConfig(provider="default")
VectorDBConfig(provider="default")