from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ..components.retreivers._base_vector_db_client import VectorDBIndex, VectorDBClient, DocumentDB

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, EmbeddingProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, BadRequestError
from unifai.components._base_component import UnifAIComponent, convert_exceptions

# import chromadb
from chromadb import Client as ChromaDefaultClient, PersistentClient as ChromaPersistentClient
from chromadb.api import ClientAPI as ChromaClientAPI
from chromadb.api.models.Collection import Collection as ChromaCollection
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings as ChromaSettings
from chromadb.api.types import (
    Documents,
    EmbeddingFunction,
    Embeddings,
    GetResult as ChromaGetResult,
    QueryResult as ChromaQueryResult
)
from chromadb.errors import ChromaError

from itertools import zip_longest

 
class ChromaExceptionConverter(UnifAIComponent):
    def convert_exception(self, exception: ChromaError) -> UnifAIError:
        status_code=exception.code()
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message(), 
            status_code=status_code,
            original_exception=exception
        )   


class ChromaIndex(VectorDBIndex, ChromaExceptionConverter):
    provider = "chroma"
    wrapped: ChromaCollection
    
    @convert_exceptions
    def count(self, **kwargs) -> int:
        return self.wrapped.count()
    
    @convert_exceptions
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[EmbeddingProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",
               **kwargs
               ) -> Self:
        
        if new_name is not None:
            self.name = new_name
        if new_metadata is not None:
            if metadata_update_mode == "replace":
                self.metadata = new_metadata
            else:
                self.metadata.update(new_metadata)
                
        self.wrapped.modify(name=self.name, metadata=self.metadata, **kwargs)
        return self

            
    @convert_exceptions
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> Self:
        
        self.wrapped.add(
            ids=ids, 
            metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings,
            **kwargs
        )
        return self

    @convert_exceptions
    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
          
        self.wrapped.update(
            ids=ids, 
            metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings,
            **kwargs
        )
        return self
    
    @convert_exceptions
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
        
        self.wrapped.upsert(
            ids=ids, 
            metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings,
            **kwargs
        )
        return self
    
    @convert_exceptions
    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        return self.wrapped.delete(ids=ids, where=where, where_document=where_document, **kwargs)


    @convert_exceptions
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        
        get_result = self.wrapped.get(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=include,
            **kwargs
        )
        return VectorDBGetResult(
            ids=get_result["ids"],
            metadatas=get_result["metadatas"],
            documents=get_result["documents"],
            embeddings=get_result["embeddings"],
            included=["ids", *get_result["included"]] 
        )
    

    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:
        if query_text is not None:
            query_texts = [query_text]
            query_embeddings = None
        elif query_embedding is not None:
            query_texts = None
            query_embeddings = [query_embedding]
        else:
            raise ValueError("Either query_text or query_embedding must be provided")
        return self.query_many(query_texts, query_embeddings, n_results, where, where_document, include, **kwargs)[0]

    
    @convert_exceptions
    def query_many(self,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]:
        
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        query_result = self.wrapped.query(
            query_embeddings=query_embeddings, 
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document, 
            include=include,
            **kwargs
        )

        included = query_result["included"]
        empty_tuple = ()
        return [
            VectorDBQueryResult(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings,
                distances=distances,
                included=["ids", *included]
            ) for ids, metadatas, documents, embeddings, distances in zip_longest(
                query_result["ids"],
                query_result["metadatas"] or empty_tuple,
                query_result["documents"] or empty_tuple,
                query_result["embeddings"] or empty_tuple,
                query_result["distances"] or empty_tuple,
                fillvalue=None
            )
        ]    
    


    def list_ids(self, **kwargs) -> list[str]:
        return self.get(include=[], **kwargs).ids


    def delete_all(self, **kwargs) -> None:
        self.delete(ids=self.list_ids(), **kwargs)    


class ChromaClient(VectorDBClient, ChromaExceptionConverter):
    client: ChromaClientAPI
    default_embedding_provider = "ollama"

    def import_client(self) -> Callable:
        from chromadb import Client
        return Client

        
    def init_client(self, **client_kwargs) -> ChromaClientAPI:
        self.client_kwargs.update(client_kwargs)
        # tentant = self.client_kwargs.get("tenant", DEFAULT_TENANT)
        # database = self.client_kwargs.get("database", DEFAULT_DATABASE)
        path = self.client_kwargs.pop("path", None)
        settings = self.client_kwargs.get("settings", None)

        extra_kwargs = {k: v for k, v in self.client_kwargs.items() if k not in ["tenant", "database", "settings"]}

        if settings is None:
            settings = ChromaSettings(**extra_kwargs)
        elif isinstance(settings, dict):
            settings = ChromaSettings(**settings, **extra_kwargs)
        elif not isinstance(settings, ChromaSettings):
            raise ValueError("Settings must be a dictionary or a chromadb.config.Settings object")

        for k in extra_kwargs:
            setattr(settings, k, self.client_kwargs.pop(k))

        if path is not None:
            if settings.persist_directory:
                raise ValueError("Path and persist_directory cannot both be set. path is shorthand for persist_directory={path} and is_persistent=True")
            settings.persist_directory = path if isinstance(path, str) else str(path)
            settings.is_persistent = True
        elif settings.persist_directory and not settings.is_persistent:
            settings.is_persistent = True           

        self.client_kwargs["settings"] = settings
        self._client = self.import_client()(**self.client_kwargs)
        return self._client
    
                                   
    @convert_exceptions                           
    def create_index(self, 
                     name: str,
                     embedding_provider: Optional[EmbeddingProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     document_db: Optional[DocumentDB] = None,
                     metadata: Optional[dict] = None,                     
                     **kwargs
                     ) -> ChromaIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric
        document_db = document_db or self.default_document_db

        if metadata is None:
            metadata = {}
        if "_unifai_embedding_config" not in metadata:
            metadata["_unifai_embedding_config"] = ",".join((
                str(embedding_provider),
                str(embedding_model),
                str(dimensions),
                str(distance_metric)
            ))

        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions
        )

        index_kwargs = {**self.default_index_kwargs, **kwargs}                
        collection = self.client.create_collection(
            name=name, 
            metadata=metadata,
            embedding_function=embedding_function,
            **index_kwargs
        )
        index = ChromaIndex(
            wrapped=collection,
            name=name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            metadata=metadata,
            **index_kwargs
        )
        self.indexes[name] = index
        return index
    
    @convert_exceptions
    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[EmbeddingProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  document_db: Optional[DocumentDB] = None,
                  **kwargs                                    
                  ) -> ChromaIndex:
        if index := self.indexes.get(name):
            return index
        
        index_kwargs = {**self.default_index_kwargs, **kwargs}

        if not (embedding_provider or embedding_model or dimensions or distance_metric):
            # get by name, extract metadata, and use that to create the index
            collection = self.client.get_collection(name=name, **index_kwargs)
            if not (embedding_config := collection.metadata.get("_unifai_embedding_config")):
                raise ValueError(f"Index {name} does not have an embedding config and kwargs are not provided")            
            embedding_provider, embedding_model, dimensions, distance_metric = (
                config_val if config_val != "None" else None for config_val in embedding_config.split(",")
            )
            dimensions = int(dimensions) if dimensions else None


        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric 
        document_db = document_db or self.default_document_db               

        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions
        )

        collection = self.client.get_collection(
            name=name,
            embedding_function=embedding_function,
            **index_kwargs
        )
        
        index = ChromaIndex(
            wrapped=collection,
            name=name,
            metadata=collection.metadata,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            **index_kwargs
        )
        self.indexes[name] = index
        return index        


    @convert_exceptions
    def count_indexes(self) -> int:
        return self.client.count_collections()


    @convert_exceptions
    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop
                     ) -> list[str]:
    
        return [collection.name for collection in self.client.list_collections(limit=limit, offset=offset)]
    
    @convert_exceptions
    def delete_index(self, name: str, **kwargs) -> None:
        self.indexes.pop(name, None)
        return self.client.delete_collection(name=name, **kwargs)
