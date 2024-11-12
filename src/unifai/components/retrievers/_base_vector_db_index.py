from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import Embedding, Embeddings, EmbeddingProvider, EmbeddingTaskTypeInput, VectorDBGetResult, VectorDBQueryResult
from ..document_dbs._base_document_db import DocumentDB
from ._base_retriever import Retriever



class VectorDBIndex(Retriever):
    provider = "base_vector_db"

    default_embed_kwargs = {
        "input_too_large": "raise_error",
        "dimensions_too_large": "raise_error",
        "task_type_not_supported": "use_closest_supported"
    }    

    def __init__(self,
                 _wrapped: Any,
                 _embed: Callable[..., Embeddings],           
                 name: str,                                 
                 dimensions: Optional[int] = None,
                 distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                 document_db: Optional[DocumentDB] = None,                 

                 embedding_provider: Optional[EmbeddingProvider] = None,
                 embedding_model: Optional[str] = None,
                 embed_document_task_type: EmbeddingTaskTypeInput = "retrieval_document",
                 embed_query_task_type: EmbeddingTaskTypeInput = "retrieval_query",
                 embed_kwargs: dict = default_embed_kwargs,
                #  embed_input_too_large: Literal["truncate_end", "truncate_start", "raise_error"] = "raise_error",
                #  embed_dimensions_too_large: Literal["reduce_dimensions", "raise_error"] = "raise_error",
                #  embed_task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported",                 
                 
                 **index_kwargs
                 ):
        
        self._wrapped = _wrapped        
        self._embed = _embed
        
        self.name = name
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.index_kwargs = index_kwargs        
        self.document_db = document_db
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model

        self.embed_document_task_type = embed_document_task_type
        self.embed_query_task_type = embed_query_task_type
        self.embed_kwargs = embed_kwargs
        self.embed_response_infos = []

        # self.embed_input_too_large = embed_input_too_large
        # self.embed_dimensions_too_large = embed_dimensions_too_large
        # self.embed_task_type_not_supported = embed_task_type_not_supported


    def embed(self, *args, **kwargs) -> Embeddings:
        embed_kwargs = {
            **self.embed_kwargs, 
            "model": self.embedding_model,
            "dimensions": self.dimensions,
            **kwargs
        }
        embeddings = self._embed(*args, **embed_kwargs)
        self.embed_response_infos.append(embeddings.response_info)
        return embeddings

    # def embed_documents(self, documents: list[str], **kwargs) -> Embeddings:
    #     return self.embed(documents, task_type=self.embed_document_task_type, **kwargs)
    
    # def embed_queries(self, queries: list[str], **kwargs) -> Embeddings:
    #     return self.embed(queries, task_type=self.embed_query_task_type, **kwargs)


    def _prepare_embeddings(
            self, 
            embed_as: Literal["documents", "queries"],
            texts_or_embeddings: list[str] | list[Embedding] | Embeddings, 
            **kwargs
        ) -> list[Embedding]:
        
        if isinstance(texts_or_embeddings, Embeddings):
            return texts_or_embeddings.list()        
        if not isinstance(texts_or_embeddings, list):
            raise ValueError(f"Invalid input type {type(texts_or_embeddings)}")        
        if not texts_or_embeddings:
            raise ValueError("No texts or embeddings provided")
        if isinstance(texts_or_embeddings[0], str):
            task_type = self.embed_document_task_type if embed_as == "documents" else self.embed_query_task_type    
            return self.embed(texts_or_embeddings, task_type=task_type, **kwargs).list()
        
        return texts_or_embeddings # List of embeddings


    def _check_filter(self, filter: dict|str, value: Any) -> bool:
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


    def _check_metadata_filters(self, where: dict, metadata: dict) -> bool:
        for key, filter in where.items():
            if key == "$and":
                for sub_filter in filter:
                    if not self._check_metadata_filters(sub_filter, metadata):
                        return False
                continue
            
            if key == "$or":
                _any = False
                for sub_filter in filter:
                    if self._check_metadata_filters(sub_filter, metadata):
                        _any = True
                        break
                if not _any:                    
                    return False
                continue
            
            value = metadata.get(key)
            if not self._check_filter(filter, value):
                return False
            
        return True


    def count(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def modify(
            self, 
            new_name: Optional[str]=None, 
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,               
            **kwargs
               ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            **kwargs
            ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")


    def update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            **kwargs
                ) -> Self:
          
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            **kwargs
                ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete(
            self, 
            ids: list[str],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def get(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        raise NotImplementedError("This method must be implemented by the subclass")

    def query(
            self,              
            text_or_embedding: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
            **kwargs
              ) -> VectorDBQueryResult:
        
        raise NotImplementedError("This method must be implemented by the subclass")

    def query_many(
            self,              
            texts_or_embeddings: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
            **kwargs
              ) -> list[VectorDBQueryResult]:
        
        raise NotImplementedError("This method must be implemented by the subclass")  


    # def query(self,              
    #           query_text: Optional[str] = None,
    #           query_embedding: Optional[Embedding] = None,
    #           top_k: int = 10,
    #           where: Optional[dict] = None,
    #           where_document: Optional[dict] = None,
    #           include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
    #           **kwargs
    #           ) -> VectorDBQueryResult:
        
    #     raise NotImplementedError("This method must be implemented by the subclass")

    # def query_many(self,              
    #           query_texts: Optional[list[str]] = None,
    #           query_embeddings: Optional[list[Embedding]] = None,
    #           top_k: int = 10,
    #           where: Optional[dict] = None,
    #           where_document: Optional[dict] = None,
    #           include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
    #           **kwargs
    #           ) -> list[VectorDBQueryResult]:
        
    #     raise NotImplementedError("This method must be implemented by the subclass")    


    def list_ids(self, **kwargs) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_all(self, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")