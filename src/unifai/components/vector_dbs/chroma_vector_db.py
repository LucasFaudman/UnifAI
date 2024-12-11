from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Literal, Self, Iterable

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection as ChromaCollection, Include as ChromaInclude
from ...utils.iter_utils import zippable
from ...exceptions import ProviderUnsupportedFeatureError
from ...types import Embedding, Embeddings, GetResult, QueryResult, CollectionName
from ...configs import VectorDBCollectionConfig
from .._base_components._base_component import convert_exceptions
from .._base_components._base_vector_db_collection import VectorDBCollection
from .._base_components._base_vector_db import VectorDB
from ..adapters.chroma_adapter import ChromaExceptionConverter, ChromaAdapter

from itertools import zip_longest

class ChromaVectorDBCollection(ChromaExceptionConverter, VectorDBCollection["ChromaCollection"]):
    provider = "chroma"

    def _count(self, **kwargs) -> int:
        return self.wrapped.count()
    
    def _list_ids(self, **kwargs) -> list[str]:
        return self.get(include=[], **kwargs).ids
    
    def _update_dbs(
        self,
        wrapped_func_name: str,
        ids: list[str],
        metadatas: Optional[list[dict]] = None,
        texts: Optional[list[str]] = None,
        embeddings: Optional[list[Embedding]|Embeddings] = None,
        update_document_db: bool = True,
        **kwargs
        ) -> Self:
        
        if texts and not embeddings:
            embeddings = self._prepare_embeddings("documents", texts)
        
        getattr(self.wrapped, wrapped_func_name)(
            ids=ids, 
            metadatas=metadatas, 
            documents=texts, 
            embeddings=embeddings.list() if isinstance(embeddings, Embeddings) else embeddings,
            **kwargs
        )
        if update_document_db and self.document_db_collection:        
            self.document_db_collection.upsert(ids, metadatas, texts)            
        return self
                
    def _add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:        
        if not texts and not embeddings:
            raise ValueError("Either texts or embeddings must be provided")
        return self._update_dbs("add", ids, metadatas, texts, embeddings, update_document_db, **kwargs)
        
    def _update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._update_dbs("update", ids, metadatas, texts, embeddings, update_document_db, **kwargs)

    def _upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        return self._update_dbs("upsert", ids, metadatas, texts, embeddings, update_document_db, **kwargs)
    

    def _delete(self, 
               ids: Optional[list[str]] = None,
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               update_document_db: bool = True,
               **kwargs
               ) -> Self:
        self.wrapped.delete(ids=ids, where=where, where_document=where_document, **kwargs)
        if update_document_db and self.document_db_collection:
            self.document_db_collection.delete(ids=ids, where=where, where_document=where_document)
        return self


    def _to_chroma_include(self, include: list) -> ChromaInclude:
        return [key if key != "texts" else "documents" for key in include] # type: ignore

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
        result = self.wrapped.get(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=self._to_chroma_include(include),
            **kwargs
        )
        return GetResult(
            ids=result["ids"],
            metadatas=result["metadatas"],
            texts=result["documents"],
            embeddings=result["embeddings"],
            included=["ids", *include]
        )
        
    def _query_many(
            self,              
            query_inputs: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> list[QueryResult]:     
               
        query_result = self.wrapped.query(
            query_embeddings=self._prepare_embeddings("queries", query_inputs), 
            n_results=top_k,
            where=where, 
            where_document=where_document, 
            include=self._to_chroma_include(include),
            **kwargs
        )

        included = ["ids", *include]
        _empty = ()
        return [
            QueryResult(
                query=query,
                ids=ids,
                metadatas=metadatas,
                texts=documents,
                embeddings=embeddings,
                distances=distances,
                included=included,
            ) for query, ids, metadatas, documents, embeddings, distances in zip_longest(
                query_inputs,
                query_result["ids"],
                query_result["metadatas"] or _empty,
                query_result["documents"] or _empty,
                query_result["embeddings"] or _empty,
                query_result["distances"] or _empty,
            )
        ]



class ChromaVectorDB(ChromaAdapter, VectorDB[ChromaVectorDBCollection, "ChromaCollection"]):
    provider = "chroma"
    collection_class = ChromaVectorDBCollection

    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> Literal["cosine", "ip", "l2"]:
        if distance_metric in ("cosine", None):
            return "cosine"
        if distance_metric in ("dotproduct", "ip"):
            return "ip"
        if distance_metric == "l2":
            return "l2"
        if distance_metric == "euclidean":
            raise ProviderUnsupportedFeatureError(
                "Euclidean distance is not supported by Chroma. Use 'l2' instead. "
                "Note: l2 is squared euclidean distance which is the most similar to euclidean but still slightly different. "
                "'l2': Squared L2 distance: ∑(Ai−Bi)² vs 'euclidean': Euclidean distance: sqrt(∑(Ai−Bi)²)"
                )
        raise ValueError(f"Invalid distance_metric: {distance_metric}")
    
    def _list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
    ) -> list[str]:
        return [collection.name for collection in self.client.list_collections(limit=limit, offset=offset)]
    
    def _count_collections(self) -> int:
        return self.client.count_collections()
    
    def _delete_collection(self, name: CollectionName) -> None:
        self.collections.pop(name, None)
        return self.client.delete_collection(name=name)    

    def _create_wrapped_collection(self, config: VectorDBCollectionConfig) -> "ChromaCollection":
        # Note2self: distance_metric not need to be re-validated since done when initializing the embedder
        # Distance metric is a metadata key for Chroma
        if metadata := config.init_kwargs.get("metadata"):
            metadata["hnsw:space"] = config.distance_metric 
        else:
            metadata = {"hnsw:space": config.distance_metric}
        return self.client.create_collection(
            name=config.name, 
            metadata=metadata,
            embedding_function=None,
            configuration=config.init_kwargs.get("configuration"),
            data_loader=config.init_kwargs.get("data_loader"),
            get_or_create=bool(config.init_kwargs.get("get_or_create")),
        )

    def _get_wrapped_collection(self, config: VectorDBCollectionConfig) -> "ChromaCollection":
        return self.client.get_collection(
            name=config.name, 
            embedding_function=None,
            id=config.init_kwargs.get("id"),
            data_loader=config.init_kwargs.get("data_loader"),
        )
    
