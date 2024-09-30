from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from .._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, AIProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from pydantic import BaseModel

T = TypeVar("T")

class VectorDBIndex(UnifAIExceptionConverter):

    def __init__(self,
                 wrapped: Any,
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


    def count(self) -> int:
        return self.wrapped.count()
    
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[AIProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",

               ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")


    def prep_embeddings(self, embeddings: list[Embedding]) -> list[list[float]]|None:
        # if embeddings:
        #     return [embedding.vector for embedding in embeddings]
        return embeddings

    def prep_sequence(self, sequence: Sequence[T]) -> list[T]:
        if not isinstance(sequence, list):
            return list(sequence)
        return sequence
            
    
    def convert_get_result(self, client_get_result: Any) -> VectorDBGetResult:
        raise NotImplementedError("This method must be implemented by the subclass")    
    
    def convert_query_result(self, client_query_result: Any) -> list[VectorDBQueryResult]:
        raise NotImplementedError("This method must be implemented by the subclass")  
        

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
            embeddings=self.prep_embeddings(embeddings)
        )
        return self

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
    
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
        
        self.wrapped.upsert(
            ids=ids, 
            metadatas=metadatas, 
            documents=documents, 
            embeddings=self.prep_embeddings(embeddings)
        )
        return self
    

    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               ) -> None:
        return self.wrapped.delete(ids=ids, where=where, where_document=where_document)

    
    
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ("metadatas", "documents")
            ) -> VectorDBGetResult:
        
        client_get_result = self.wrapped.get(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=self.prep_sequence(include)
        )
        return self.convert_get_result(client_get_result)


    def query(self,              
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "distances"]] = ("metadatas", "documents", "distances"),
              ) -> list[VectorDBQueryResult]:
        
        client_query_result = self.wrapped.query(
            query_embeddings=self.prep_embeddings(query_embeddings) if query_embeddings else None, 
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document, 
            include=self.prep_sequence(include)
        )
        return self.convert_query_result(client_query_result) 
    

    def get_all_ids(self) -> list[str]:
        return self.get(include=[]).ids
    
    def delete_all(self) -> None:
        self.delete(ids=self.get_all_ids())
    

class VectorDBClient(BaseClientWrapper):
    provider = "base_vector_db"

    def __init__(self, 
                 parent,
                 default_embedding_provider: Optional[AIProvider] = None,
                 default_embedding_model: Optional[str] = None,
                 default_dimensions: Optional[int] = None,
                 default_distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                 default_index_kwargs: Optional[dict] = None,
                 **client_kwargs
                 ):
        super().__init__(**client_kwargs)

        self.parent = parent
        self.indexes = {}

        self.default_embedding_provider = default_embedding_provider
        self.default_embedding_model = default_embedding_model
        self.default_dimensions = default_dimensions
        self.default_distance_metric = default_distance_metric
        self.default_index_kwargs = default_index_kwargs or {}


    def create_index(self, 
                     name: str,
                     metadata: Optional[dict] = None,
                     embedding_provider: Optional[AIProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     **kwargs
                     ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def get_index(self, name: str) -> VectorDBIndex:
        if index := self.indexes.get(name):
            return index
        raise UnifAIError(f"Index {name} not found")


    def get_or_create_index(self, 
                            name: str,
                            metadata: Optional[dict] = None,
                            embedding_provider: Optional[AIProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                            **kwargs
                            ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop
                     ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_index(self, name: str) -> dict:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_all_indexes(self) -> None:
        for name in self.list_indexes():
            self.delete_index(name)


    def count_indexes(self) -> int:
        return self.client.count_collections()


    def count(self, name: str) -> int:
        return self.get_index(name).count()
    

    def modify_index(self, 
                     name: str, 
                     new_name: Optional[str]=None,
                     new_metadata: Optional[dict]=None,
                     ) -> VectorDBIndex:
        return self.get_index(name).modify(new_name, new_metadata)   


    def add(self,
            name: str,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            ) -> VectorDBIndex:
        return self.get_index(name).add(ids, metadatas, documents, embeddings)
    

    def update(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               ) -> VectorDBIndex:
        return self.get_index(name).update(ids, metadatas, documents, embeddings)
    

    def upsert(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               ) -> VectorDBIndex:
          return self.get_index(name).upsert(ids, metadatas, documents, embeddings)
    

    def delete(self, 
               name: str,
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               ) -> None:
        return self.get_index(name).delete(ids, where, where_document)

    def get(self,
            name: str,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ("metadatas", "documents")
            ) -> VectorDBGetResult:
        return self.get_index(name).get(ids, where, limit, offset, where_document, include)


    def query(self,
              name: str,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ("metadatas", "documents", "distances"),
              ) -> VectorDBQueryResult:
        return self.get_index(name).query(query_texts, query_embeddings, n_results, where, where_document, include)    