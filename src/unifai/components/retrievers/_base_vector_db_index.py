from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from itertools import zip_longest

from ...types import Document, Documents, Embedding, Embeddings, EmbeddingProvider, EmbeddingTaskTypeInput, VectorDBGetResult, VectorDBQueryResult
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
                 distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
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
        print(f"Embedding {len(embed_kwargs.get('input', args[0]))} documents")
        embeddings = self._embed(*args, **embed_kwargs)
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

    def _documents_to_lists(self, documents: Iterable[Document] | Documents) -> tuple[list[str], Optional[list[dict]], Optional[list[str]], list[Embedding]]:
        if not documents:
            raise ValueError("No documents provided")
        if not isinstance(documents, list) or not isinstance((doc0 := documents[0]), Document):
            raise ValueError(f"Invalid documents type: {type(documents)}. Must be list of Document or a Documents object")
        
        ids = []
        if include_metadatas := doc0.metadata is not None:
            metadatas = []
        if include_documents := doc0.text is not None:
            texts = []
        if include_embeddings := doc0.embedding is not None:
            embeddings = []
        for document in documents:
            ids.append(document.id)
            if include_metadatas:
                metadatas.append(document.metadata)
            if include_documents:
                texts.append(document.text)
            if include_embeddings:
                embeddings.append(document.embedding)

        if len(ids) != (len_documents := len(documents)):
            raise ValueError("All documents must have an id")
        if include_metadatas and len(metadatas) != len_documents:
            raise ValueError("All documents must have metadata")
        if include_documents and len(texts) != len_documents:
            raise ValueError("All documents must have text")
        if include_embeddings and len(embeddings) != len_documents:
            raise ValueError("All documents must have an embedding")
        return ids, metadatas, texts, embeddings
    
    def _iterables_to_documents(
            self,
            ids: list[str],
            metadatas: Optional[Iterable[dict]] = None,
            documents: Optional[Iterable[str]] = None,
            embeddings: Optional[Iterable[Embedding]] = None,   
    ) -> Iterable[Document]:
        for id, metadata, text, embedding in zip_longest(ids, metadatas or (), documents or (), embeddings or ()):
            yield Document(id=id, metadata=metadata, text=text, embedding=embedding)    

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
    

    # def modify(
    #         self, 
    #         new_name: Optional[str]=None, 
    #         embedding_provider: Optional[EmbeddingProvider] = None,
    #         embedding_model: Optional[str] = None,
    #         dimensions: Optional[int] = None,
    #         distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,               
    #         **kwargs
    #            ) -> Self:
        
    #     raise NotImplementedError("This method must be implemented by the subclass")



    def update_document_db(
            self, 
            ids: Optional[list[str]],
            documents: Optional[Iterable[str] | Iterable[Document] | Documents]
            ) -> None:
        if documents is not None and self.document_db:
            self.document_db.set_documents(ids, documents)

    def delete_from_document_db(self, ids: list[str]) -> None:
        if self.document_db:
            self.document_db.delete_documents(ids)


    def add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self.add_documents(self._iterables_to_documents(ids, metadatas, documents, embeddings), update_document_db, **kwargs)
        
    def add_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        return self.add(*self._documents_to_lists(documents), update_document_db, **kwargs)

    def update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self.update_documents(self._iterables_to_documents(ids, metadatas, documents, embeddings), update_document_db, **kwargs)
    
    def update_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,            
            **kwargs
                ) -> Self:

        return self.update(*self._documents_to_lists(documents), update_document_db, **kwargs)  
                 
    def upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        
        return self.upsert_documents(self._iterables_to_documents(ids, metadatas, documents, embeddings), update_document_db, **kwargs)
    
    def upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self.upsert(*self._documents_to_lists(documents), update_document_db, **kwargs)
  
    def delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
               ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

    def delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> None:    
        ids = [doc.id for doc in documents] if documents else None
        return self.delete(ids, where, where_document, update_document_db, **kwargs)
    
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
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
            **kwargs
              ) -> VectorDBQueryResult:
        
        raise NotImplementedError("This method must be implemented by the subclass")

    def query_many(
            self,              
            query_inputs: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
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