from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Literal, Self
from itertools import zip_longest, chain

if TYPE_CHECKING:
    from pinecone.grpc import GRPCIndex

from ...exceptions import ProviderUnsupportedFeatureError
from ...types import Embedding, Embeddings, GetResult, QueryResult, CollectionName
from ...configs import VectorDBCollectionConfig
from ...utils import check_filter, check_metadata_filters, limit_offset_slice
from .._base_components._base_component import convert_exceptions
from .._base_components._base_vector_db_collection import VectorDBCollection
from .._base_components._base_vector_db import VectorDB
from ..adapters.pinecone_adapter import PineconeExceptionConverter, PineconeAdapter


from pinecone import ServerlessSpec, PodSpec

class PineconeVectorDBCollection(PineconeExceptionConverter, VectorDBCollection["GRPCIndex"]):
    default_namespace = ""
    
    def _add_default_namespace(self, kwargs: dict) -> dict:
        if "namespace" not in kwargs:
            kwargs["namespace"] = self.default_namespace
        return kwargs

    @convert_exceptions
    def count(self, **kwargs) -> int:
        return self.wrapped.describe_index_stats(**kwargs).total_vector_count

    @convert_exceptions
    def add(
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
        self.upsert(ids, metadatas, texts, embeddings, update_document_db, **kwargs)
        return self

    @convert_exceptions    
    def update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:
        if texts and not embeddings:
            embeddings = self._prepare_embeddings("documents", texts)     

        for id, metadata, embedding in zip_longest(ids, metadatas or (), embeddings or ()):
            self.wrapped.update(
                id=id,
                values=embedding,
                set_metadata=metadata,
                **self._add_default_namespace(kwargs)
            )
        if update_document_db and self.document_db_collection:        
            self.document_db_collection.update(ids, metadatas, texts)
        return self
    
    @convert_exceptions
    def upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]|Embeddings] = None,
            update_document_db: bool = True,
            **kwargs
                ) -> Self:

        if texts and not embeddings:
            embeddings = self._prepare_embeddings("documents", texts)     
        if not embeddings:
            raise ValueError("Either texts or embeddings must be provided")
                
        vectors = [
            {
                "id": id,
                "values": embedding,
                "metadata": metadata
            }
            for id, metadata, embedding in zip_longest(ids, metadatas or (), embeddings)
        ]
        self.wrapped.upsert(
            vectors=vectors,
            **self._add_default_namespace(kwargs)
        )
        if update_document_db and self.document_db_collection:        
            self.document_db_collection.upsert(ids, metadatas, texts)
        return self
    
    @convert_exceptions
    def delete(self, 
               ids: Optional[list[str]] = None,
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               update_document_db: bool = True,
               **kwargs
               ) -> Self:
        if where_document:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone")
        self.wrapped.delete(ids=ids, filter=where, **self._add_default_namespace(kwargs))
        if update_document_db and self.document_db_collection:
            self.document_db_collection.delete(ids=ids, where=where, where_document=where_document)
        return self

    @convert_exceptions
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
        
        if where_document and not self.document_db_collection:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")
        
        result = self.wrapped.fetch(ids=ids, **self._add_default_namespace(kwargs))
        
        result_ids = []
        metadatas = [] if "metadatas" in include else None
        texts = [] if "texts" in include else None
        embeddings = [] if "embeddings" in include else None
        added, current_offset = 0, 0

        for vector in result.vectors.values():
            current_offset += 1
            if offset is not None and current_offset <= offset:
                continue

            # Pinecone Fetch does not support 'where' metadata filtering so need to do it here
            if where and not check_metadata_filters(where, vector.metadata):
                continue
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db_collection:
                document = self.document_db_collection.get_document(vector.id)
                if not check_filter(where_document, document.text):
                    continue
                if texts is not None: # "documents" in include
                    texts.append(document.text)

            # Append result after filtering
            result_ids.append(vector.id)
            if embeddings is not None:
                embeddings.append(vector.values)
            if metadatas is not None:
                metadatas.append(vector.metadata)
            added += 1

        if result_ids and texts is not None and not where_document and self.document_db_collection:
            # Get texts for all results if not already done when checking where_document
            texts.extend(self.document_db_collection.get(result_ids, include=["texts"]).texts or ())

        return GetResult(
            ids=result_ids,
            metadatas=metadatas,
            texts=texts,
            embeddings=embeddings,
            included=["ids", *include]
        )
    
    @convert_exceptions
    def query(
            self,              
            query_input: str|Embedding,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts", "embeddings", "distances"]] = ["metadatas", "texts", "embeddings", "distances"],
            **kwargs
              ) -> QueryResult:        
        
        if where_document and not self.document_db_collection:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")

        if isinstance(query_input, str):
            query_embedding = self._prepare_embeddings("queries", [query_input])[0]
        else:
            query_embedding = query_input

        result_ids = []
        metadatas = [] if "metadatas" in include else None        
        texts = [] if "texts" in include else None
        embeddings = [] if "embeddings" in include else None        
        distances = [] if "distances" in include else None
        
        result = self.wrapped.query(
            vector=query_embedding,
            top_k=top_k,
            filter=where,
            include_values=(embeddings is not None),
            include_metadata=(include_metadata := (metadatas is not None)),
            **self._add_default_namespace(kwargs)
        )

        for match in result["matches"]:
            if where and include_metadata:
                metadata = match["metadata"]
                # Preforms any additional metadata filtering not supported by Pinecone
                if not check_metadata_filters(where, metadata):
                    continue

            id = match["id"]            
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db_collection:
                document = self.document_db_collection.get_document(id)
                if not check_filter(where_document, document.text):
                    continue
                if texts is not None: # "documents" in include
                    texts.append(document.text)

            # Append result after filtering
            result_ids.append(id)
            if embeddings is not None:
                embeddings.append(match["values"])
            if metadatas is not None:
                metadatas.append(match["metadata"])
            if distances is not None:
                distances.append(match["score"])

        if texts is not None and not where_document and self.document_db_collection:
            # Get texts for all results if not already done when checking where_document
            texts.extend(self.document_db_collection.get(result_ids, include=["texts"]).texts or ())

        return QueryResult(
            ids=result_ids,
            metadatas=metadatas,
            texts=texts,
            embeddings=embeddings,
            distances=distances,
            included=["ids", *include],
            query=query_input
        )
            
    @convert_exceptions
    def list_ids(self, **kwargs) -> list[str]:
        return list(chain(*self.wrapped.list(**self._add_default_namespace(kwargs))))
    
    def delete_all(self, **kwargs) -> None:
        self.wrapped.delete(delete_all=True, **self._add_default_namespace(kwargs))     


class PineconeVectorDB(PineconeAdapter, VectorDB[PineconeVectorDBCollection, "GRPCIndex"]):
    index_type = PineconeVectorDBCollection
    default_spec = ServerlessSpec(cloud="aws", region="us-west-1")
    collection_class = PineconeVectorDBCollection

    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> str:
        if distance_metric in ("cosine", None):
            return "cosine"
        if distance_metric in ("dotproduct", "ip"):
            return "dotproduct"
        if distance_metric == "euclidean":
            return "euclidean"
        if distance_metric == "l2":
            raise ProviderUnsupportedFeatureError(
                "Squared L2 distance is not supported by Pinecone. Use 'euclidean' instead. "
                "Note: l2 is squared euclidean distance which is the most similar to euclidean but still slightly different. "
                "'l2': Squared L2 distance: ∑(Ai−Bi)² vs 'euclidean': Euclidean distance: sqrt(∑(Ai−Bi)²)"
                )
        raise ValueError(f"Invalid distance_metric: {distance_metric}")

    @convert_exceptions
    def _create_wrapped_collection(self, config: VectorDBCollectionConfig) -> "GRPCIndex":
        
        if spec := config.init_kwargs.get("spec"): 
            spec_type = None
        elif spec := config.init_kwargs.get("serverless_spec"): 
            spec_type = "serverless"
        elif spec := config.init_kwargs.get("pod_spec"):
            spec_type = "pod"
        else:
            spec = self.default_spec
            # raise ValueError("No spec provided for index creation. Must provide either 'spec', 'serverless_spec', or 'pod_spec' with either dict ServerlessSpec or PodSpec")

        if isinstance(spec, dict): # not a ServerlessSpec or PodSpec instance 
            if spec_type is None:              
                if not (spec_type := spec.get("type")):
                    raise KeyError("No spec type provided. Must provide 'type' key with either 'serverless' or 'pod' when spec is a dict")
            if spec_type == "serverless":
                spec = ServerlessSpec(**spec)
            elif spec_type == "pod":
                spec = PodSpec(**spec) 
            else:
                raise ValueError(f"Invalid spec type: {spec_type}. Must be either 'serverless' or 'pod'")
 
        self.client.create_index(
            name=config.name, 
            dimension=config.dimensions, # type: ignore (Does not need to be re-validated since done when initializing the embedder)
            spec=spec,
            metric=config.distance_metric, # same as above
            timeout=config.init_kwargs.get("timeout"),
            deletion_protection=config.init_kwargs.get("deletion_protection"),
        )        
        return self._get_wrapped_collection(config)
    
    @convert_exceptions
    def _get_wrapped_collection(self, config: VectorDBCollectionConfig) -> "GRPCIndex":
        return self.client.Index(name=config.name, host=config.init_kwargs.get("host", ""))

    @convert_exceptions
    def list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
    ) -> list[str]:
        return [index.name for index in self.client.list_indexes()[limit_offset_slice(limit, offset)]]
    
    @convert_exceptions
    def count_collections(self) -> int:
        return len(self.client.list_indexes())
    
    @convert_exceptions
    def delete_collection(self, name: CollectionName) -> None:
        self.collections.pop(name, None)
        return self.client.delete_index(name)

