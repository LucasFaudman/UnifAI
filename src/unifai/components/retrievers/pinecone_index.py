from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Literal, Self
from itertools import zip_longest, chain

if TYPE_CHECKING:
    from pinecone.grpc import GRPCIndex

from ...exceptions import ProviderUnsupportedFeatureError
from ...types import Embedding, Embeddings, EmbeddingProvider, VectorDBGetResult, VectorDBQueryResult
from .._base_component import convert_exceptions
from ..base_adapters.pinecone_base import PineconeExceptionConverter
from ._base_vector_db_index import VectorDBIndex

class PineconeIndex(PineconeExceptionConverter, VectorDBIndex):
    provider = "pinecone"
    default_namespace = ""
    _wrapped: GRPCIndex
    
    def _add_default_namespace(self, kwargs: dict) -> dict:
        if "namespace" not in kwargs:
            kwargs["namespace"] = self.default_namespace
        return kwargs

    @convert_exceptions
    def count(self, **kwargs) -> int:
        return self._wrapped.describe_index_stats(**kwargs).total_vector_count
    

    # @convert_exceptions
    # def modify(self, 
    #            new_name: Optional[str]=None, 
    #            new_metadata: Optional[dict]=None,
    #            embedding_provider: Optional[EmbeddingProvider] = None,
    #            embedding_model: Optional[str] = None,
    #            dimensions: Optional[int] = None,
    #            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,               
    #            metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",
    #            **kwargs
    #            ) -> Self:        
    #     raise ProviderUnsupportedFeatureError("modify is not supported by Pinecone. See: https://docs.pinecone.io/guides/indexes/configure-an-index")

            
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            update_document_db: bool = True,
            **kwargs
            ) -> Self:
        
        self.upsert(ids, metadatas, documents, embeddings, update_document_db, **kwargs)
        return self


    @convert_exceptions
    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                update_document_db: bool = True,
                **kwargs
                ) -> Self:

        if documents and not embeddings:
            embeddings = self._prepare_embeddings("documents", documents)     

        for id, metadata, embedding in zip_longest(ids, metadatas or (), embeddings or ()):
            self._wrapped.update(
                id=id,
                values=embedding,
                set_metadata=metadata,
                **self._add_default_namespace(kwargs)
            )
        if update_document_db:
            self.update_document_db(ids, documents)
        return self
    

    @convert_exceptions
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                update_document_db: bool = True,
                **kwargs
                ) -> Self:

        if documents and not embeddings:
            embeddings = self._prepare_embeddings("documents", documents)  
        if not embeddings:
            raise ValueError("Either documents or embeddings must be provided")
                
        vectors = [
            {
                "id": id,
                "values": embedding,
                "metadata": metadata
            }
            for id, metadata, embedding in zip_longest(ids, metadatas or (), embeddings)
        ]
        self._wrapped.upsert(
            vectors=vectors,
            **self._add_default_namespace(kwargs)
        )
        if update_document_db:
            self.update_document_db(ids, documents)        
        return self
    

    @convert_exceptions
    def delete(self, 
               ids: Optional[list[str]] = None,
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               update_document_db: bool = True,
               **kwargs
               ) -> None:
        if where_document:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone")
        self._wrapped.delete(ids=ids, filter=where, **self._add_default_namespace(kwargs))
        if update_document_db and ids:
            self.delete_from_document_db(ids)        



    @convert_exceptions
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
        
        if where_document and not self.document_db:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")
        
        result = self._wrapped.fetch(ids=ids, **self._add_default_namespace(kwargs))
        
        result_ids = []
        embeddings = [] if "embeddings" in include else None
        metadatas = [] if "metadatas" in include else None
        documents = [] if "documents" in include else None

        for vector in result.vectors.values():
            metadata = vector.metadata
            # Pinecone Fetch does not support 'where' metadata filtering so need to do it here
            if where and not self._check_metadata_filters(where, metadata):
                continue
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db:
                document = self.document_db.get_document(vector.id)
                if not self._check_filter(where_document, document):
                    continue
                if documents is not None: # "documents" in include
                    documents.append(document)

            # Append result after filtering
            result_ids.append(vector.id)
            if embeddings is not None:
                embeddings.append(vector.values)
            if metadatas is not None:
                metadatas.append(metadata)

        if documents is not None and not where_document and self.document_db:
            # Get documents for all results if not already done when checking where_document
            documents.extend(self.document_db.get_documents(result_ids))

        return VectorDBGetResult(
            ids=result_ids,
            metadatas=metadatas,
            documents=documents,
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
            include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
            **kwargs
              ) -> VectorDBQueryResult:   
        
        if where_document and not self.document_db:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")

        query_embedding = self._prepare_embeddings("queries", [query_input])[0]

        result_ids = []
        embeddings = [] if "embeddings" in include else None
        metadatas = [] if "metadatas" in include else None        
        distances = [] if "distances" in include else None
        documents = [] if "documents" in include else None
        
        result = self._wrapped.query(
            vector=query_embedding,
            top_k=top_k,
            filter=where,
            include_values=(embeddings is not None),
            include_metadata=(include_metadata:=(metadatas is not None)),
            **self._add_default_namespace(kwargs)
        )

        for match in result["matches"]:
            if where and include_metadata:
                metadata = match["metadata"]
                # Preforms any additional metadata filtering not supported by Pinecone
                if not self._check_metadata_filters(where, metadata):
                    continue

            id = match["id"]            
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db:
                document = self.document_db.get_document(id)
                if not self._check_filter(where_document, document):
                    continue
                if documents is not None: # "documents" in include
                    documents.append(document)

            # Append result after filtering
            result_ids.append(id)
            if embeddings is not None:
                embeddings.append(match["values"])
            if metadatas is not None:
                metadatas.append(match["metadata"])
            if distances is not None:
                distances.append(match["score"])

        if documents is not None and not where_document and self.document_db:
            # Get documents for all results if not already done when checking where_document
            documents.extend(self.document_db.get_documents(result_ids))  

        return VectorDBQueryResult(
            ids=result_ids,
            metadatas=metadatas,
            documents=documents,
            embeddings=embeddings,
            distances=distances,
            included=["ids", *include]
        )
            

    @convert_exceptions
    def query_many(
            self,              
            query_inputs: list[str] | list[Embedding] | Embeddings,
            top_k: int = 10,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
            **kwargs
              ) -> list[VectorDBQueryResult]: 
        
        query_embeddings = self._prepare_embeddings("queries", query_inputs)
        return [
            self.query(query_embedding, top_k, where, where_document, include, **kwargs)
            for query_embedding in query_embeddings
        ]


    def list_ids(self, **kwargs) -> list[str]:
        return list(chain(*self._wrapped.list(**self._add_default_namespace(kwargs))))
    

    def delete_all(self, **kwargs) -> None:
        self._wrapped.delete(delete_all=True, **self._add_default_namespace(kwargs))     
