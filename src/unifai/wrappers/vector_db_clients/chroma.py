from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_vector_db_client import BaseVectorDBIndex, BaseVectorDBClient

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, AIProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, BadRequestError
from unifai.wrappers._base_client_wrapper import UnifAIExceptionConverter, convert_exceptions
from pydantic import BaseModel

# import chromadb
from chromadb import Client as ChromaDefaultClient
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

class ChromaExceptionConverter(UnifAIExceptionConverter):
    def convert_exception(self, exception: ChromaError) -> UnifAIError:
        status_code=exception.code()
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message(), 
            status_code=status_code,
            original_exception=exception
        )    

class ChromaIndex(BaseVectorDBIndex, ChromaExceptionConverter):

    def __init__(self,
                 wrapped: ChromaCollection,
                 name: str,
                 metadata: Optional[dict] = None,
                 embedding_provider: Optional[AIProvider] = None,
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

    
    def convert_exception(self, exception: ChromaError) -> UnifAIError:
        status_code=exception.code()
        unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        return unifai_exception_type(
            message=exception.message(), 
            status_code=status_code,
            original_exception=exception
        )

    @convert_exceptions
    def count(self) -> int:
        return self.wrapped.count()
    
    @convert_exceptions
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[AIProvider] = None,
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

    def convert_get_result(self, client_get_result: ChromaGetResult) -> VectorDBGetResult:
        return VectorDBGetResult(
            ids=client_get_result["ids"],
            metadatas=client_get_result["metadatas"],
            documents=client_get_result["documents"],
            embeddings=client_get_result["embeddings"],
            included=client_get_result["included"] 
        )
    
    def convert_query_result(self, client_query_result: Any) -> VectorDBQueryResult:
        raise NotImplementedError("This method must be implemented by the subclass")  
        
    @convert_exceptions
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            ) -> Self:
        
        self.wrapped.add(
            ids=ids, metadatas=metadatas, 
            documents=documents, 
            embeddings=self.prep_embeddings(embeddings)
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
            embeddings=self.prep_embeddings(embeddings)
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
            embeddings=self.prep_embeddings(embeddings)
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
        
        client_get_result = self.wrapped.get(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=include
        )
        return self.convert_get_result(client_get_result)

    @convert_exceptions
    def query(self,
              query_embeddings: Optional[list[Embedding]] = None,
              query_texts: Optional[list[str]] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> VectorDBQueryResult:
        
        client_query_result = self.wrapped.query(
            query_embeddings=self.prep_embeddings(query_embeddings) if query_embeddings else None, 
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document, 
            include=include
        )
        return self.convert_query_result(client_query_result)   


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
        embded_result = self.parent.embed(
            input=input,
            model=self.model,
            provider=self.embedding_provider,
            max_dimensions=self.max_dimensions
        )
        self.response_infos.append(embded_result.response_info)
        return embded_result.list()
        

class ChromaClient(BaseVectorDBClient, ChromaExceptionConverter):
    client: ChromaClientAPI

    def import_client(self) -> Callable:
        from chromadb import Client
        return Client
        
    def init_client(self, **client_kwargs) -> ChromaClientAPI:
        self.client_kwargs.update(client_kwargs)
        tentant = self.client_kwargs.pop("tenant", DEFAULT_TENANT)
        database = self.client_kwargs.pop("database", DEFAULT_DATABASE)
        
        # extra kwargs will be passed to the Settings constructor
        extra_kwargs = {k: v for k, v in self.client_kwargs.items() if k not in ["settings", "tenant", "database"]}
        self.client_kwargs.clear()

        if self.client_kwargs.get("settings") is None:
            self.client_kwargs["settings"] = ChromaSettings(**extra_kwargs)
        if not isinstance((settings := self.client_kwargs["settings"]), ChromaSettings):
            self.client_kwargs["settings"] = ChromaSettings(**settings, **extra_kwargs)            
        if self.client_kwargs["settings"].persist_directory and self.client_kwargs["settings"].is_persistent is None:
            self.client_kwargs["settings"].is_persistent = True

        self.client_kwargs["tenant"] = tentant
        self.client_kwargs["database"] = database
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
                     embedding_provider: Optional[AIProvider] = None,
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
                  embedding_provider: Optional[AIProvider] = None,
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
    def get_or_create_index(self, 
                            name: str,
                            metadata: Optional[dict] = None,
                            embedding_provider: Optional[AIProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                            **kwargs
                            ) -> ChromaIndex:
        try:
            return self.get_index(
                name=name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                dimensions=dimensions,
                distance_metric=distance_metric,
                **kwargs
            )
        except BadRequestError:
            return self.create_index(
                name=name,
                metadata=metadata,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                dimensions=dimensions,
                distance_metric=distance_metric,
                **kwargs
            )
    
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
    def delete_index(self, name: str):
        del self.indexes[name]
        return self.client.delete_collection(name=name)
