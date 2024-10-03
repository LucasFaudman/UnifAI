from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from pydantic import BaseModel

T = TypeVar("T")

class VectorDBIndex(UnifAIExceptionConverter):
    provider = "base_vector_db"

    def __init__(self,
                 wrapped: Any,
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


    def count(self) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[LLMProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",

               ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")


    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
          
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings"]] = ["metadatas", "documents"],
            ) -> VectorDBGetResult:
        raise NotImplementedError("This method must be implemented by the subclass")


    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> VectorDBQueryResult:
        
        raise NotImplementedError("This method must be implemented by the subclass")

    def query_many(self,              
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> list[VectorDBQueryResult]:
        
        raise NotImplementedError("This method must be implemented by the subclass")    

    def get_all_ids(self) -> list[str]:
        return self.get(include=[]).ids
    

    def delete_all(self) -> None:
        self.delete(ids=self.get_all_ids())