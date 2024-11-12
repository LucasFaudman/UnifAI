from typing import Optional, Literal

from ...types import EmbeddingProvider, EmbeddingTaskTypeInput, Embeddings
from .._base_component import convert_exceptions
from ..document_dbs._base_document_db import DocumentDB
from ..base_adapters.chroma_base import ChromaAdapter
from ._base_vector_db_client import VectorDBClient
from .chroma_index import ChromaIndex


class ChromaClient(ChromaAdapter, VectorDBClient):

    def _init_index(
            self, 
            _wrapped,            
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
    ) -> ChromaIndex:

        index = ChromaIndex(
            _wrapped=_wrapped,
            _embed=self.embed,
            **self._apply_defaults(
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
        )
        self.indexes[name] = index
        return index    
                                       
    @convert_exceptions                           
    def create_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> ChromaIndex:

        index_kwargs = {**self.default_index_kwargs, **index_kwargs}        
        collection = self.client.create_collection(
            name=name, 
            embedding_function=None,
            **index_kwargs
        )
        return self._init_index(
            _wrapped=collection,
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

    
    @convert_exceptions
    def get_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs                          
                  ) -> ChromaIndex:
        if index := self.indexes.get(name):
            return index
        
        index_kwargs = {**self.default_index_kwargs, **index_kwargs}       
        collection = self.client.get_collection(
            name=name,
            embedding_function=None,
            **index_kwargs
        )        
        return self._init_index(
            _wrapped=collection,
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
