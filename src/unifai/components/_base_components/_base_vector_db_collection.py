from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self, Generic
from abc import abstractmethod

from itertools import zip_longest

from ...types import Document, Documents, Embedding, Embeddings, EmbeddingTaskTypeInput, GetResult, QueryResult, RankedDocument
from ._base_embedder import Embedder

from ...configs.vector_db_config import VectorDBCollectionConfig
from ...type_conversions.documents import iterables_to_documents, documents_to_lists
from ._base_document_db import DocumentDBCollection
from ._base_db import BaseDBCollection, WrappedT

class VectorDBCollection(BaseDBCollection[VectorDBCollectionConfig, WrappedT], Generic[WrappedT]):
    component_type = "vector_db_collection"
    provider = "base"
    config_class = VectorDBCollectionConfig

    _document_attrs = ("ids", "metadatas", "texts", "embeddings")
    _is_abstract = True
    _abstract_methods = ("_add", "_update", "_upsert", "_delete", "_get", "_query")

    def _setup(self) -> None:
        super()._setup()
        self.embedder: Embedder = self.init_kwargs.pop("embedder", None)
        self.document_db_collection: Optional[DocumentDBCollection] = self.init_kwargs.pop("document_db_collection", None)

        config = self.config
        self.dimensions = config.dimensions or self.embedder.default_dimensions
        self.distance_metric = config.distance_metric
        self.embedding_model = config.embedding_model or self.embedder.default_model
        self.embed_document_task_type = config.embed_document_task_type
        self.embed_query_task_type = config.embed_query_task_type
        if config.extra_kwargs and (embed_kwargs := config.extra_kwargs.get("embed")):
            self.embed_kwargs = embed_kwargs
        else:
            self.embed_kwargs = {}        
        self.embed_response_infos = []    

    # Embedding methods
    @property
    def embedding_provider(self) -> str:
        return self.embedder.provider

    def embed(self, *args, **kwargs) -> Embeddings:
        embed_kwargs = {
            **self.embed_kwargs, 
            "model": self.embedding_model,
            "dimensions": self.dimensions,
            **kwargs
        }
        print(f"Embedding {len(embed_kwargs.get('input', args[0]))} documents")
        embeddings = self.embedder.embed(*args, **embed_kwargs)
        self.embed_response_infos.append(embeddings.response_info)
        return embeddings

    def _prepare_embeddings(
            self, 
            embed_as: Literal["documents", "queries"],
            inputs: list[str] | list[Embedding] | Embeddings, 
            **kwargs
        ) -> list[Embedding]:
        if not inputs:
            raise ValueError("Must provide either documents or embeddings")        
        if isinstance(inputs, Embeddings):
            return inputs.list()        
        if not isinstance(inputs, list):
            raise ValueError(f"Invalid input type {type(inputs)}")
        if isinstance((item := inputs[0]), str):
            task_type = self.embed_document_task_type if embed_as == "documents" else self.embed_query_task_type    
            return self.embed(inputs, task_type=task_type, **kwargs).list()
        if isinstance(item, list) and isinstance(item[0], (int, float)):
            return inputs
        
        raise ValueError(f"Invalid input type {type(inputs)}")

    # Abstract methods (One of _<method> or _<method>_documents must be implemented by the subclass)
    # see AbstractBaseComponent.__init_subclass__ for more details
    
    # Abstract methods from BaseDBCollection:
    # def _count(self, **kwargs) -> int:
    # def _list_ids(self, **kwargs) -> list[str]:

    def _add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self._add_documents(iterables_to_documents(ids, metadatas, texts, embeddings), update_document_db, **kwargs)
        
    def _add_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self._add(*documents_to_lists(documents, self._document_attrs), update_document_db, **kwargs)

    def _update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._update_documents(iterables_to_documents(ids, metadatas, texts, embeddings), update_document_db, **kwargs)
    
    def _update_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,            
            **kwargs
                ) -> Self:
        return self._update(*documents_to_lists(documents, self._document_attrs), update_document_db, **kwargs)  
                 
    def _upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._upsert_documents(iterables_to_documents(ids, metadatas, texts, embeddings), update_document_db, **kwargs)
    
    def _upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._upsert(*documents_to_lists(documents, self._document_attrs), update_document_db, **kwargs)
  
    def _delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
               ) -> Self:
        return self._delete_documents(iterables_to_documents(ids, attrs=("id",)), where, where_document, update_document_db, **kwargs)

    def _delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:    
        ids = [doc.id for doc in documents] if documents else None
        return self._delete(ids, where, where_document, update_document_db, **kwargs)
    
    def _get(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
            ) -> GetResult:
        return GetResult.from_documents(self._get_documents(ids, where, where_document, include, limit, offset, **kwargs))
    
    def _get_documents(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> list[Document]:
        return self._get(ids, where, where_document, include, limit, offset, **kwargs).to_documents()
                        
    def _query(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> QueryResult:    
        # If not implemented try to use _query_many first otherwise use _query_documents
        if self._query_many is not VectorDBCollection._query_many:
            return self._query_many([query_input], top_k, where, where_document, include, **kwargs)[0]        
        return QueryResult.from_documents(self._query_documents(query_input, top_k, where, where_document, include, **kwargs))                    

    def _query_documents(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[RankedDocument]:        
        return self._query(query_input, top_k, where, where_document, include, **kwargs).to_documents()

    def _query_many(
            self,              
            query_inputs: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[QueryResult]:
        query_embeddings = self._prepare_embeddings("queries", query_inputs)
        return [self._query(query_embedding, top_k, where, where_document, include, **kwargs) for query_embedding in query_embeddings]


    # Public methods
    def add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self._run_func(self._add, ids, metadatas, texts, embeddings, update_document_db, **kwargs)
        
    def add_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self._run_func(self._add_documents, documents, update_document_db, **kwargs)

    def update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._run_func(self._update, ids, metadatas, texts, embeddings, update_document_db, **kwargs)
    
    def update_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,            
            **kwargs
                ) -> Self:
        return self._run_func(self._update_documents, documents, update_document_db, **kwargs)
                 
    def upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._run_func(self._upsert, ids, metadatas, texts, embeddings, update_document_db, **kwargs)
    
    def upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._run_func(self._upsert_documents, documents, update_document_db, **kwargs)
    
    def delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
               ) -> Self:
        return self._run_func(self._delete, ids, where, where_document, update_document_db, **kwargs)

    def delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:    
        return self._run_func(self._delete_documents, documents, where, where_document, update_document_db, **kwargs)
    
    def get(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
            ) -> GetResult:
        return self._run_func(self._get, ids, where, where_document, include, limit, offset, **kwargs)
    
    def get_documents(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> list[Document]:
        return self._run_func(self._get_documents, ids, where, where_document, include, limit, offset, **kwargs)

    def query(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> QueryResult:        
        return self._run_func(self._query, query_input, top_k, where, where_document, include, **kwargs)
    
    def query_documents(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[RankedDocument]:        
        return self._run_func(self._query_documents, query_input, top_k, where, where_document, include, **kwargs)

    def query_many(
            self,              
            query_inputs: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[QueryResult]:        
        return self._run_func(self._query_many, query_inputs, top_k, where, where_document, include, **kwargs)