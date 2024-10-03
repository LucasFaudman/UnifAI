from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper
from ._base_vector_db_index import VectorDBIndex

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError


T = TypeVar("T")

class VectorDBClient(BaseClientWrapper):
    provider = "base_vector_db"

    def __init__(self, 
                 parent,
                 default_embedding_provider: Optional[LLMProvider] = None,
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
                     embedding_provider: Optional[LLMProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     **kwargs
                     ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[LLMProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  **kwargs  
                  ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")


    def get_or_create_index(self, 
                            name: str,
                            metadata: Optional[dict] = None,
                            embedding_provider: Optional[LLMProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                            **kwargs
                            ) -> VectorDBIndex:
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
    

    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop
                     ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_index(self, name: str) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_indexes(self, names: Collection[str]) -> None:
        for name in names:
            self.delete_index(name)


    def delete_all_indexes(self) -> None:
        self.delete_indexes(self.list_indexes())


    def count_indexes(self) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")


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
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"]
            ) -> VectorDBGetResult:
        return self.get_index(name).get(ids, where, limit, offset, where_document, include)


    def query(self,
              name: str,
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,         
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> VectorDBQueryResult:
        return self.get_index(name).query(query_text, query_embedding, n_results, where, where_document, include)    
    

    def query_many(self,
              name: str,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> list[VectorDBQueryResult]:
        return self.get_index(name).query_many(query_texts, query_embeddings, n_results, where, where_document, include)        