from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self, Generic, TypeAlias
from abc import abstractmethod

from ._base_adapter import UnifAIAdapter
from ._base_db import BaseDB, WrappedT
from ._base_vector_db_collection import VectorDBCollection

from ._base_document_db import DocumentDB, DocumentDBCollection

from ._base_embedder import Embedder

from ...types import ResponseInfo, Embedding, Embeddings, EmbeddingTaskTypeInput, Usage, GetResult, QueryResult, Document, Documents, RankedDocument, RankedDocuments
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError, NotFoundError, CollectionNotFoundError

from ...types.annotations import ComponentName, ModelName, ProviderName, CollectionName
from ...configs import VectorDBConfig, VectorDBCollectionConfig, EmbedderConfig, DocumentDBConfig, DocumentDBCollectionConfig
from ...utils import check_filter, check_metadata_filters, limit_offset_slice, update_kwargs_with_locals
T = TypeVar("T")

# DBConfigT = TypeVar("DBConfigT", bound=VectorDBConfig)
CollectionT = TypeVar("CollectionT", bound=VectorDBCollection)

class VectorDB(BaseDB[VectorDBConfig, VectorDBCollectionConfig, CollectionT, WrappedT], Generic[CollectionT, WrappedT]):
    component_type = "vector_db"
    provider = "base"
    config_class = VectorDBConfig
    collection_class: Type[CollectionT]
    
    can_get_components = True

    def _setup(self):
        super()._setup()
        self._document_db = self._get_document_db(self.config.document_db) if self.config.document_db else None
        
    @abstractmethod
    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]:
        pass
    
    @abstractmethod
    def _create_wrapped_collection(self, config: VectorDBCollectionConfig) -> WrappedT:
        pass

    @abstractmethod
    def _get_wrapped_collection(self, config: VectorDBCollectionConfig) -> WrappedT:
        pass   

    # Concrete methods
    def _get_embedder(self, embedder: EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]) -> Embedder:
        return self._get_component("embedder", embedder)
    
    def _get_document_db(self, document_db: DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]) -> DocumentDB:
        return self._get_component("document_db", document_db)

    def _init_embedder(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
        ) -> tuple[VectorDBCollectionConfig, Embedder]:

        # Raise an error if the specified distance metric is not supported before doing any more work
        config.distance_metric = self._validate_distance_metric(config.distance_metric)        
        
        # Use or initialize the embedder
        embedder = embedder if isinstance(embedder, Embedder) else self._get_embedder(embedder or config.embedder)
        embedding_model = config.embedding_model or embedder.default_model
        
        if config.dimensions is None:
            # Use the dimensions of the embedding model if not specified
            config.dimensions = embedder.get_model_dimensions(embedding_model)
        else:
            # Raise an error if the specified dimensions are too large for the model
            config.dimensions = embedder.validate_dimensions(embedding_model, config.dimensions, reduce_dimensions=False)

        return config, embedder
    
    def _init_document_db_collection(
            self,
            config: VectorDBCollectionConfig,
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
        ) -> tuple[VectorDBCollectionConfig, Optional[DocumentDBCollection]]:
        
        document_db_collection = document_db_collection or config.document_db_collection
        if not document_db_collection:
            _default_document_db_collection = None
            if self._document_db:
                # If no collection is specified but the DB has a default document DB, 
                # get or create a collection from it with the same name as the vector DB collection
                _default_document_db_collection = self._document_db.get_or_create_collection(config.name)
            return config, _default_document_db_collection
    
        if isinstance(document_db_collection, DocumentDBCollection):
            # If a DocumentDBCollection instance is passed, use it as is
            return config, document_db_collection
        if isinstance(document_db_collection, DocumentDBCollectionConfig):
            document_db = self._document_db or self._get_document_db(document_db_collection.provider)
            # If the document DB collection config has a different name, get or create a collection with that name
            return config, document_db.get_or_create_collection_from_config(document_db_collection)
        
        if isinstance(document_db_collection, DocumentDB):
            # Arg passed is a DocumentDB instance not DocumentDBCollection 
            # so get or create a collection with the same name as the VectorDBCollection from the DocumentDB
            document_db = document_db_collection 
        elif isinstance(document_db_collection, DocumentDBConfig):
            # Same as above but with a DocumentDBConfig
            document_db = self._get_document_db(document_db_collection)
        else:
            # Same as above but with a provider name or tuple or (provider name, component name)
            document_db = self._get_document_db(document_db_collection)
        
        # DocumentDBCollection is not specified so use VectorDBCollection name
        return config, document_db.get_or_create_collection(config.name)

    def _create_collection_from_config(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            ) -> CollectionT:
        
        config, embedder = self._init_embedder(config, embedder)
        config, document_db_collection = self._init_document_db_collection(config, document_db_collection)
        wrapped = self._create_wrapped_collection(config)
        return self.init_collection_from_wrapped(config, wrapped, embedder=embedder, document_db_collection=document_db_collection)
    
    def _get_collection_from_config(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            ) -> CollectionT:
        if collection := self.collections.get(config.name):
            return collection        
        config, embedder = self._init_embedder(config, embedder)
        config, document_db_collection = self._init_document_db_collection(config, document_db_collection)
        wrapped = self._get_wrapped_collection(config)
        return self.init_collection_from_wrapped(config, wrapped, embedder=embedder, document_db_collection=document_db_collection)   

    # Public methods
    # Collection Creation/Getting methods
    def create_collection_from_config(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            ) -> CollectionT:        
        return self._run_func(self._create_collection_from_config, config, embedder, document_db_collection)

    def get_collection_from_config(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            ) -> CollectionT:
        return self._run_func(self._get_collection_from_config, config, embedder, document_db_collection)

    def get_or_create_collection_from_config(
            self, 
            config: VectorDBCollectionConfig,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            ) -> CollectionT:
        try:
            return self.get_collection_from_config(config, embedder, document_db_collection)
        except (CollectionNotFoundError, NotFoundError, BadRequestError):
            return self.create_collection_from_config(config, embedder, document_db_collection)       
    
    def create_collection(
            self, 
            name: CollectionName = "default_collection",
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            embedding_model: Optional[str] = None,            
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,   
            **collection_kwargs
        ) -> CollectionT:    
        
        config = self._kwargs_to_collection_config(
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            **collection_kwargs
        )        
        return self.create_collection_from_config(config, embedder, document_db_collection)        
        
    def get_collection(
            self, 
            name: CollectionName = "default_collection",
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            embedding_model: Optional[str] = None,            
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,   
            **collection_kwargs
        ) -> CollectionT:
        if name and (collection := self.collections.get(name)):
            return collection            
        config = self._kwargs_to_collection_config(
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            **collection_kwargs
        )        
        return self.get_collection_from_config(config, embedder, document_db_collection) 

    def get_or_create_collection(
            self, 
            name: CollectionName = "default_collection",
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            embedding_model: Optional[str] = None,            
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embedder: Optional[Embedder | EmbedderConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,            
            document_db_collection: Optional[DocumentDBCollection | DocumentDBCollectionConfig | DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]] = None,   
            **collection_kwargs
        ) -> CollectionT:
        if name and (collection := self.collections.get(name)):
            return collection        
        config = self._kwargs_to_collection_config(
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            **collection_kwargs
        )
        return self.get_or_create_collection_from_config(config, embedder, document_db_collection)
    
    # Collection Modifier Methods
    def add(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).add(ids, metadatas, texts, embeddings, **kwargs)
    
    def delete(
            self, 
            collection: CollectionName | CollectionT,
            ids: list[str],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).delete(ids, where, where_document, **kwargs)

    def update(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).update(ids, metadatas, texts, embeddings, **kwargs)
    
    def upsert(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).upsert(ids, metadatas, texts, embeddings, **kwargs)
    
    def get(
            self,
            collection: CollectionName | CollectionT,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
            ) -> GetResult:
        return self._resolve_collection(collection).get(ids, where, where_document, include, limit, offset, **kwargs)

    def query(self,
              collection: CollectionName | CollectionT,
              query_input: str|Embedding,      
              top_k: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "distances"],
              **kwargs
              ) -> QueryResult:
        return self._resolve_collection(collection).query(query_input, top_k, where, where_document, include, **kwargs)    
    
    def query_many(self,
              collection: CollectionName | CollectionT,
              query_inputs: list[str] | list[Embedding] | Embeddings,        
              top_k: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "distances"],
              **kwargs
              ) -> list[QueryResult]:
        return self._resolve_collection(collection).query_many(query_inputs, top_k, where, where_document, include, **kwargs)        
    
    # Document Methods    
    def get_documents(
            self, 
            collection: CollectionName | CollectionT,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> list[Document]:
        return self._resolve_collection(collection).get_documents(ids, where, where_document, include, limit, offset, **kwargs)
    
    def query_documents(
            self, 
            collection: CollectionName | CollectionT,
            query_input: str|Embedding,       
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "distances"],
            **kwargs
    ) -> list[RankedDocument]:
        return self._resolve_collection(collection).query_documents(query_input, top_k, where, where_document, include, **kwargs)