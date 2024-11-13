from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ..base_adapters._base_adapter import UnifAIAdapter
from ._base_vector_db_index import Retriever, VectorDBIndex, DocumentDB
from ..embedders._base_embedder import Embedder

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, EmbeddingProvider, EmbeddingTaskTypeInput, Usage, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError, NotFoundError, IndexNotFoundError


T = TypeVar("T")
IndexType = TypeVar("IndexType", bound=VectorDBIndex)

class VectorDBClient(UnifAIAdapter, Retriever):
    provider = "base_vector_db"
    index_type = VectorDBIndex

    def __init__(self, 
                 _embed: Callable[..., Embeddings],
                 default_dimensions: int = 768,
                 default_distance_metric: Literal["cosine", "dotproduct", "euclidean", "ip", "l2"] = "cosine",                 
                 default_document_db: Optional[DocumentDB] = None,
                 default_index_kwargs: Optional[dict] = None,                 
                 default_embedding_provider: EmbeddingProvider = "openai",
                 default_embedding_model: Optional[str] = None,
                 default_embed_document_task_type: EmbeddingTaskTypeInput = "retrieval_document",
                 default_embed_query_task_type: EmbeddingTaskTypeInput = "retrieval_query",
                 default_embed_kwargs: dict = {
                    "input_too_large": "raise_error",
                    "dimensions_too_large": "raise_error",
                    "task_type_not_supported": "use_closest_supported"
                 },                 
                 **client_kwargs
                 ):
        super().__init__(**client_kwargs)

        self.indexes = {}
        self.embed_response_infos = []

        self._embed = _embed
        self.default_dimensions = default_dimensions
        self.default_distance_metric = self._validate_distance_metric(default_distance_metric)
        self.default_document_db = default_document_db
        self.default_index_kwargs = default_index_kwargs or {}
        self.default_embedding_provider = default_embedding_provider
        self.default_embedding_model = default_embedding_model
        self.default_embed_document_task_type = default_embed_document_task_type
        self.default_embed_query_task_type = default_embed_query_task_type
        self.default_embed_kwargs = default_embed_kwargs


    def embed(self, *args, **kwargs) -> Embeddings:
        embeddings = self._embed(*args, **kwargs)
        self.embed_response_infos.append(embeddings.response_info)
        return embeddings

    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]:
        raise NotImplementedError("This method must be implemented by the subclass")

    def _init_index(
            self, 
            _wrapped: Any,
            _index_type: Type[IndexType],
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
    ) -> IndexType:
        index_kwargs.update(
                name=name,
                dimensions=dimensions or self.default_dimensions,
                distance_metric=distance_metric or self.default_distance_metric,
                document_db=document_db or self.default_document_db,                     
                embedding_provider=embedding_provider or self.default_embedding_provider,
                embedding_model=embedding_model or self.default_embedding_model,
                embed_document_task_type=embed_document_task_type or self.default_embed_document_task_type,
                embed_query_task_type=embed_query_task_type or self.default_embed_query_task_type,
                embed_kwargs=embed_kwargs or self.default_embed_kwargs,       
        )        

        index = _index_type(
            _wrapped=_wrapped,
            _embed=self.embed,
            **index_kwargs
        )
        self.indexes[name] = index
        return index       

    def _create_wrapped_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def _get_wrapped_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")


    def create_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> VectorDBIndex:
        
        distance_metric = self._validate_distance_metric(distance_metric) if distance_metric else self.default_distance_metric
        index_kwargs = {**self.default_index_kwargs, **index_kwargs}            
        _wrapped = self._create_wrapped_index(
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            embed_kwargs=embed_kwargs,
            **index_kwargs
        )
        return self._init_index(
            _wrapped=_wrapped,
            _index_type=self.index_type,
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            embed_kwargs=embed_kwargs,
            **index_kwargs
        )

    def get_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                  ) -> VectorDBIndex:
        
        if index := self.indexes.get(name):
            return index        
        
        distance_metric = self._validate_distance_metric(distance_metric) if distance_metric else self.default_distance_metric
        index_kwargs = {**self.default_index_kwargs, **index_kwargs}            
        _wrapped = self._get_wrapped_index(
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            embed_kwargs=embed_kwargs,
            **index_kwargs
        )
        return self._init_index(
            _wrapped=_wrapped,
            _index_type=self.index_type,
            name=name,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embed_document_task_type=embed_document_task_type,
            embed_query_task_type=embed_query_task_type,
            embed_kwargs=embed_kwargs,
            **index_kwargs
        )


    def get_or_create_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                            ) -> VectorDBIndex:
        try:
            return self.get_index(
                name=name,
                dimensions=dimensions,
                distance_metric=distance_metric,                
                document_db=document_db,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embed_document_task_type=embed_document_task_type,
                embed_query_task_type=embed_query_task_type,
                embed_kwargs=embed_kwargs,
                **index_kwargs
            )
        except (IndexNotFoundError, NotFoundError, BadRequestError):
            return self.create_index(
                name=name,
                dimensions=dimensions,
                distance_metric=distance_metric,                
                document_db=document_db,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embed_document_task_type=embed_document_task_type,
                embed_query_task_type=embed_query_task_type,
                embed_kwargs=embed_kwargs,
                **index_kwargs
            )        
    

    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop,
                     **kwargs
                     ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_index(self, name: str, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_indexes(self, names: Collection[str], **kwargs) -> None:
        for name in names:
            self.delete_index(name, **kwargs)


    def delete_all_indexes(self, **kwargs) -> None:
        self.delete_indexes(self.list_indexes(), **kwargs)


    def count_indexes(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")


    def count(self, name: str, **kwargs) -> int:
        return self.get_index(name).count(**kwargs)
    

    def modify_index(self, 
                     name: str, 
                     new_name: Optional[str]=None,
                     **kwargs
                     ) -> VectorDBIndex:
        return self.get_index(name).modify(new_name, **kwargs)   


    def add(self,
            name: str,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> VectorDBIndex:
        return self.get_index(name).add(ids, metadatas, documents, embeddings, **kwargs)
    

    def update(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               **kwargs
               ) -> VectorDBIndex:
        return self.get_index(name).update(ids, metadatas, documents, embeddings, **kwargs)
    

    def upsert(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               **kwargs
               ) -> VectorDBIndex:
          return self.get_index(name).upsert(ids, metadatas, documents, embeddings, **kwargs)
    

    def delete(self, 
               name: str,
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        return self.get_index(name).delete(ids, where, where_document, **kwargs)


    def get(self,
            name: str,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        return self.get_index(name).get(ids, where, limit, offset, where_document, include, **kwargs)


    def query(self,
              name: str,
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,         
              top_k: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:
        return self.get_index(name).query(query_text, query_embedding, top_k, where, where_document, include, **kwargs)    
    

    def query_many(self,
              name: str,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              top_k: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]:
        return self.get_index(name).query_many(query_texts, query_embeddings, top_k, where, where_document, include, **kwargs)        