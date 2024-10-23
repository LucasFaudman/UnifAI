from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import Embedding, EmbeddingProvider, VectorDBGetResult, VectorDBQueryResult
from ..document_dbs._base_document_db import DocumentDB
from ._base_retriever import Retriever

class VectorDBIndex(Retriever):
    provider = "base_vector_db"

    def __init__(self,
                 wrapped: Any,
                 name: str,
                 embedding_function: Optional[Callable] = None,
                 embedding_provider: Optional[EmbeddingProvider] = None,
                 embedding_model: Optional[str] = None,
                 dimensions: Optional[int] = None,
                 distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                 document_db: Optional[DocumentDB] = None,
                 metadata: Optional[dict] = None,
                 **kwargs
                 ):
        
        self.wrapped = wrapped
        self.name = name
        self.embedding_function = embedding_function
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.document_db = document_db
        self.metadata = metadata or {}
        self.kwargs = kwargs


    def check_filter(self, filter: dict|str, value: Any) -> bool:
        if isinstance(filter, str):
            return value == filter
        filter_operator, filter_value = next(iter(filter.items()))
        if filter_operator == "$eq":
            return value == filter_value
        if filter_operator == "$ne":
            return value != filter_value
        if filter_operator == "$gt":
            return value > filter_value
        if filter_operator == "$gte":
            return value >= filter_value
        if filter_operator == "$lt":
            return value < filter_value
        if filter_operator == "$lte":
            return value <= filter_value
        if filter_operator == "$in":
            return value in filter_value
        if filter_operator == "$nin":
            return value not in filter_value
        if filter_operator == "$exists":
            return bool(value) == filter_value
        if filter_operator == "$contains":
            return filter_value in value
        if filter_operator == "$not_contains":
            return filter_value not in value
        raise ValueError(f"Invalid filter {filter}")


    def check_metadata_filters(self, where: dict, metadata: dict) -> bool:
        for key, filter in where.items():
            if key == "$and":
                for sub_filter in filter:
                    if not self.check_metadata_filters(sub_filter, metadata):
                        return False
                continue
            
            if key == "$or":
                _any = False
                for sub_filter in filter:
                    if self.check_metadata_filters(sub_filter, metadata):
                        _any = True
                        break
                if not _any:                    
                    return False
                continue
            
            value = metadata.get(key)
            if not self.check_filter(filter, value):
                return False
            
        return True


    def count(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")
    

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
        
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")


    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
          
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        raise NotImplementedError("This method must be implemented by the subclass")


    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:
        
        raise NotImplementedError("This method must be implemented by the subclass")

    def query_many(self,              
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]:
        
        raise NotImplementedError("This method must be implemented by the subclass")    


    def list_ids(self, **kwargs) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_all(self, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")