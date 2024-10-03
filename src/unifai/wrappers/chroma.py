from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_vector_db_client import VectorDBIndex, VectorDBClient

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, BadRequestError
from unifai.wrappers._base_client_wrapper import UnifAIExceptionConverter, convert_exceptions

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

 
class ChromaExceptionConverter(UnifAIExceptionConverter):
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

    def __init__(self,
                 wrapped: ChromaCollection,
                 name: str,
                 metadata: Optional[dict] = None,
                 embedding_provider: Optional[LLMProvider] = None,
                 embedding_model: Optional[str] = None,
                 dimensions: Optional[int] = None,
                 distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                 **kwargs
                 ):
        
        self.wrapped = wrapped
        self.name = name
        self.metadata = metadata or {}
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.kwargs = kwargs

    
    @convert_exceptions
    def count(self) -> int:
        return self.wrapped.count()
    
    @convert_exceptions
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[LLMProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",

               ) -> Self:
        
        if new_name is not None:
            self.name = new_name
        if new_metadata is not None:
            if metadata_update_mode == "replace":
                self.metadata = new_metadata
            else:
                self.metadata.update(new_metadata)
                
        self.wrapped.modify(name=self.name, metadata=self.metadata)
        return self

            
    @convert_exceptions
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            ) -> Self:
        
        self.wrapped.add(
            ids=ids, 
            metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings
        )
        return self

    @convert_exceptions
    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
          
        self.wrapped.update(
            ids=ids, metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings
        )
        return self
    
    @convert_exceptions
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
        
        self.wrapped.upsert(
            ids=ids, metadatas=metadatas, 
            documents=documents, 
            embeddings=embeddings
        )
        return self
    
    @convert_exceptions
    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               ) -> None:
        return self.wrapped.delete(ids=ids, where=where, where_document=where_document)


    @convert_exceptions
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"]
            ) -> VectorDBGetResult:
        
        get_result = self.wrapped.get(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=include
        )
        return VectorDBGetResult(
            ids=get_result["ids"],
            metadatas=get_result["metadatas"],
            documents=get_result["documents"],
            embeddings=get_result["embeddings"],
            included=get_result["included"] 
        )
    

    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> VectorDBQueryResult:
        if query_text is not None:
            query_texts = [query_text]
            query_embeddings = None
        elif query_embedding is not None:
            query_texts = None
            query_embeddings = [query_embedding]
        else:
            raise ValueError("Either query_text or query_embedding must be provided")
        return self.query_many(query_texts, query_embeddings, n_results, where, where_document, include)[0]

    
    @convert_exceptions
    def query_many(self,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> list[VectorDBQueryResult]:
        
        if query_texts is None and query_embeddings is None:
            raise ValueError("Either query_texts or query_embeddings must be provided")

        query_result = self.wrapped.query(
            query_embeddings=query_embeddings, 
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document, 
            include=include
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
                included=included
            ) for ids, metadatas, documents, embeddings, distances in zip_longest(
                query_result["ids"],
                query_result["metadatas"] or empty_tuple,
                query_result["documents"] or empty_tuple,
                query_result["embeddings"] or empty_tuple,
                query_result["distances"] or empty_tuple,
                fillvalue=None
            )
        ]    


class UnifAIChromaEmbeddingFunction(EmbeddingFunction[list[str]]):
    def __init__(
            self,
            parent,
            embedding_provider: Optional[str] = None,
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            response_infos: Optional[list[ResponseInfo]] = None,
    ):
        self.parent = parent
        self.embedding_provider = embedding_provider or parent.default_provider
        self.model = model
        self.max_dimensions = max_dimensions
        self.response_infos = response_infos or []

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents."""
        print(f"Embedding {len(input)} documents")
        embed_result = self.parent.embed(
            input=input,
            model=self.model,
            provider=self.embedding_provider,
            max_dimensions=self.max_dimensions
        )
        self.response_infos.append(embed_result.response_info)
        return embed_result.list()
        

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
    
    
    def get_embedding_function(self,
                               embedding_provider: Optional[str] = None,
                               embedding_model: Optional[str] = None,
                               dimensions: Optional[int] = None,
                               response_infos: Optional[list[ResponseInfo]] = None,
                                 ) -> UnifAIChromaEmbeddingFunction:
        return UnifAIChromaEmbeddingFunction(
            parent=self.parent,
            embedding_provider=embedding_provider or self.default_embedding_provider,
            model=embedding_model or self.default_embedding_model,
            max_dimensions=dimensions or self.default_dimensions,
            response_infos=response_infos
        )
                               
    @convert_exceptions                           
    def create_index(self, 
                     name: str,
                     metadata: Optional[dict] = None,
                     embedding_provider: Optional[LLMProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     **kwargs
                     ) -> ChromaIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric

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
            metadata=metadata,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            **index_kwargs
        )
        self.indexes[name] = index
        return index
    
    @convert_exceptions
    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[LLMProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
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
    def delete_index(self, name: str) -> None:
        self.indexes.pop(name, None)
        return self.client.delete_collection(name=name)
