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
from ...utils import check_filter, check_metadata_filters, limit_offset_slice, update_kwargs_with_locals, clean_locals

CollectionT = TypeVar("CollectionT", bound=VectorDBCollection)

class VectorDB(BaseDB[VectorDBConfig, VectorDBCollectionConfig, CollectionT, WrappedT], Generic[CollectionT, WrappedT]):
    component_type = "vector_db"
    provider = "base"
    config_class = VectorDBConfig
    collection_class: Type[CollectionT]
    
    can_get_components = True

    def _setup(self):
        super()._setup()
        self._document_db = None
        
    @abstractmethod
    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]:
        pass
    
    def _set_document_db(self, document_db: Optional["DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]"]):
        if document_db is None or isinstance(document_db, DocumentDB):
            self._document_db = document_db
        else:
            self._document_db = self._get_component("document_db", document_db)

    @property
    def document_db(self) -> Optional[DocumentDB]:
        if not self._document_db and self.config.document_db:
            self._set_document_db(self.config.document_db)
        return self._document_db
    
    @document_db.setter
    def document_db(self, document_db: Optional["DocumentDB | DocumentDBConfig | ProviderName | tuple[ProviderName, ComponentName]"]):
        self._set_document_db(document_db)
    
    def _init_document_db_collection(
            self,
            config: VectorDBCollectionConfig,
        ) -> VectorDBCollectionConfig:
        
        document_db_collection = config.document_db_collection
        if not document_db_collection:
            if self.document_db:
                # If no collection is specified but the DB has a default document DB, 
                # get or create a collection from it with the same name as the vector DB collection
                config.document_db_collection = self.document_db.get_or_create_collection(config.name)
            return config
    
        if isinstance(document_db_collection, DocumentDBCollection):
            # If a DocumentDBCollection instance is passed, use it as is
            return config
        
        if isinstance(document_db_collection, DocumentDBCollectionConfig):
            if self.document_db and self.document_db.provider == document_db_collection.provider:
                document_db = self.document_db
            else:
                document_db = self._get_component("document_db", document_db_collection.provider)
            # If the document DB collection config has a different name, get or create a collection with that name
            config.document_db_collection = document_db.get_or_create_collection_from_config(document_db_collection)
            return config
        
        if isinstance(document_db_collection, DocumentDB):
            # Arg passed is a DocumentDB instance not DocumentDBCollection 
            # so get or create a collection with the same name as the VectorDBCollection from the DocumentDB
            document_db = document_db_collection
        elif isinstance(document_db_collection, DocumentDBConfig):
            # Same as above but with a DocumentDBConfig
            document_db = self._get_component("document_db", document_db_collection)
        else:
            # Same as above but with a provider name or tuple or (provider name, component name)
            document_db = self._get_component("document_db", document_db_collection)
        
        # DocumentDBCollection is not specified so use VectorDBCollection name
        config.document_db_collection = document_db.get_or_create_collection(config.name)
        return config
    
    def _init_embedder(
            self, 
            config: VectorDBCollectionConfig,
        ) -> VectorDBCollectionConfig:

        # Raise an error if the specified distance metric is not supported before doing any more work
        config.distance_metric = self._validate_distance_metric(config.distance_metric)        
        
        # Use or initialize the embedder
        embedder = config.embedder
        if not isinstance(embedder, Embedder):
            embedder = self._get_component("embedder", embedder)

        embedding_model = config.embedding_model or embedder.default_model
        
        if config.dimensions is None:
            # Use the dimensions of the embedding model if not specified
            config.dimensions = embedder.get_model_dimensions(embedding_model)
        else:
            # Raise an error if the specified dimensions are too large for the model
            config.dimensions = embedder.validate_dimensions(embedding_model, config.dimensions, reduce_dimensions=False)
        
        config.embedder = embedder
        return config    
    
    def _init_collection_config_components(
            self,
            config: VectorDBCollectionConfig,
            **init_kwargs
    ) -> VectorDBCollectionConfig:
        config = self._init_document_db_collection(config)
        config = self._init_embedder(config)
        return config    

    # Public methods
    # Collection Creation/Getting methods    
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
            **init_kwargs
        ) -> CollectionT:    
        config = self._kwargs_to_collection_config(**clean_locals(locals()), **init_kwargs)
        return self.create_collection_from_config(config, **init_kwargs)
             
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
            **init_kwargs
        ) -> CollectionT:
        if name in self.collections:
            return self.collections[name]
        config = self._kwargs_to_collection_config(**clean_locals(locals()), **init_kwargs)
        return self.get_collection_from_config(config, **init_kwargs)

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
            **init_kwargs
        ) -> CollectionT:
        if name in self.collections:
            return self.collections[name]      
        config = self._kwargs_to_collection_config(**clean_locals(locals()), **init_kwargs)
        return self.get_or_create_collection_from_config(config, **init_kwargs)
    
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
    ) -> Documents:
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
    ) -> RankedDocuments:
        return self._resolve_collection(collection).query_documents(query_input, top_k, where, where_document, include, **kwargs)