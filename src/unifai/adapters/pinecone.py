from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_vector_db_client import VectorDBIndex, VectorDBClient, DocumentDB

from unifai.types import ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, BadRequestError
from unifai.adapters._base_adapter import UnifAIExceptionConverter, convert_exceptions

from pinecone.grpc import PineconeGRPC as Pinecone, GRPCIndex  
from pinecone import ServerlessSpec, PodSpec, Index
from pinecone.exceptions import PineconeException, PineconeApiException

from json import loads as json_loads, JSONDecodeError


from itertools import zip_longest, chain

def convert_spec(spec: dict, spec_type: Literal["pod", "serverless", None]) ->  PodSpec | ServerlessSpec:
    if not spec_type:
        spec_type = spec.get("type")
    if spec_type == "pod":
        return PodSpec(**spec)
    elif spec_type == "serverless":
        return ServerlessSpec(**spec)
    else:
        raise ValueError(f"Unknown spec type {spec['type']}")
    

def limit_offest_slice(limit: Optional[int], offset: Optional[int]) -> slice:
    if limit is None:
        return slice(offset, None)
    if offset is None:
        return slice(limit)
    return slice(offset, limit + offset)

def add_default_namespace(kwargs: dict) -> dict:
    if "namespace" not in kwargs:
        kwargs["namespace"] = ""
    return kwargs
 
class PineconeExceptionConverter(UnifAIExceptionConverter):
    def convert_exception(self, exception: PineconeException) -> UnifAIError:
        if not isinstance(exception, PineconeApiException):
            return UnifAIError(
                message=str(exception),
                original_exception=exception
            )
        status_code=exception.status
        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError
                    
        error_code = None
        if body := getattr(exception, "body", None):
            message = body
            try:
                decoded_body = json_loads(body)
                error = decoded_body["error"]
                message = error.get("message") or body
                error_code = error.get("code")
            except (JSONDecodeError, KeyError, AttributeError):
                pass
        else:
            message = str(exception)
        
        return unifai_exception_type(
            message=message,
            error_code=error_code,
            status_code=status_code,
            original_exception=exception,
        )   


class PineconeIndex(VectorDBIndex, PineconeExceptionConverter):
    provider = "pinecone"
    wrapped: GRPCIndex


    @convert_exceptions
    def count(self, **kwargs) -> int:
        return self.wrapped.describe_index_stats(**kwargs).total_vector_count
    

    @convert_exceptions
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[LLMProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",
               **kwargs
               ) -> Self:        
        raise ProviderUnsupportedFeatureError("modify is not supported by Pinecone. See: https://docs.pinecone.io/guides/indexes/configure-an-index")

            
    @convert_exceptions
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> Self:
        
        self.upsert(ids, metadatas, documents, embeddings, **kwargs)
        return self


    @convert_exceptions
    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
        metadatas = metadatas or []

        # if embeddings and documents:
        #     raise ValueError("Cannot provide both documents and embeddings")
        if not embeddings and documents and self.embedding_function:
            embeddings = self.embedding_function(documents)
        if not embeddings:
            raise ValueError("Must provide either documents or embeddings")
        if documents and self.document_db:
            self.document_db.set_documents(ids, documents)
                    
        for id, metadata, embedding in zip_longest(ids, metadatas, embeddings):
            self.wrapped.update(
                id=id,
                values=embedding,
                set_metadata=metadata,
                **add_default_namespace(kwargs)
            )
        return self
    

    @convert_exceptions
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
        metadatas = metadatas or []

        # if embeddings and documents:
        #     raise ValueError("Cannot provide both documents and embeddings")
        if not embeddings and documents and self.embedding_function:
            embeddings = self.embedding_function(documents)
        if not embeddings:
            raise ValueError("Must provide either documents or embeddings")
        if documents and self.document_db:
            self.document_db.set_documents(ids, documents)        
        
        vectors = []        
        for id, metadata, embedding in zip_longest(ids, metadatas, embeddings):
            vectors.append({
                "id": id,
                "values": embedding,
                "metadata": metadata
            })
        self.wrapped.upsert(
            vectors=vectors,
            **add_default_namespace(kwargs)
        )
        return self
    

    @convert_exceptions
    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        if where_document:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone")
        if self.document_db:
            self.document_db.delete_documents(ids)
        self.wrapped.delete(ids=ids, filter=where, **add_default_namespace(kwargs))


    @convert_exceptions
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        
        if where_document and not self.document_db:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")
        
        result = self.wrapped.fetch(ids=ids, **add_default_namespace(kwargs))
        
        result_ids = []
        embeddings = [] if "embeddings" in include else None
        metadatas = [] if "metadatas" in include else None
        documents = [] if "documents" in include else None

        for vector in result.vectors.values():
            metadata = vector.metadata
            # Pinecone Fetch does not support 'where' metadata filtering so need to do it here
            if where and not self.check_metadata_filters(where, metadata):
                continue
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db:
                document = self.document_db.get_document(vector.id)
                if not self.check_filter(where_document, document):
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
    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:   
        
        if where_document and not self.document_db:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")

        if query_text is not None and self.embedding_function:
            query_embedding = self.embedding_function(query_text)[0]            
        elif query_embedding is None:
            raise ValueError("Either (query_text and embedding_function) or query_embedding must be provided")

        result_ids = []
        embeddings = [] if "embeddings" in include else None
        metadatas = [] if "metadatas" in include else None        
        distances = [] if "distances" in include else None
        documents = [] if "documents" in include else None
        
        result = self.wrapped.query(
            vector=query_embedding,
            top_k=n_results,
            filter=where,
            include_values=(embeddings is not None),
            include_metadata=(include_metadata:=(metadatas is not None)),
            **add_default_namespace(kwargs)
        )

        for match in result["matches"]:
            if where and include_metadata:
                metadata = match["metadata"]
                # Preforms any additional metadata filtering not supported by Pinecone
                if not self.check_metadata_filters(where, metadata):
                    continue

            id = match["id"]            
            # Same for 'where_document' filtering but only if document_db is provided to get the documents 
            if where_document and self.document_db:
                document = self.document_db.get_document(id)
                if not self.check_filter(where_document, document):
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
    def query_many(self,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]: 
        
        if where_document and not self.document_db:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone directly. A DocumentDB subclass must be provided to support this feature")
        
        if query_embeddings is None: 
            if query_texts is None:
                raise ValueError("Must provide either query_texts or query_embeddings not both")
            if self.embedding_function:
                query_embeddings = self.embedding_function(query_texts)
        if not query_embeddings:
            raise ValueError("Must provide either query_texts or query_embeddings")

        return [
            self.query(None, query_embedding, n_results, where, where_document, include, **kwargs)
            for query_embedding in query_embeddings
        ]


    def list_ids(self, **kwargs) -> list[str]:
        return list(chain(*self.wrapped.list(**add_default_namespace(kwargs))))
    

    def delete_all(self, **kwargs) -> None:
        self.wrapped.delete(delete_all=True, **add_default_namespace(kwargs))     


class PineconeClient(VectorDBClient, PineconeExceptionConverter):
    client: Pinecone
    default_embedding_provider = "pinecone"

    def import_client(self) -> Callable:
        from pinecone.grpc import PineconeGRPC
        return PineconeGRPC

                                       
    @convert_exceptions                           
    def create_index(self, 
                     name: str,
                     metadata: Optional[dict] = None,
                     embedding_provider: Optional[LLMProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     document_db: Optional[DocumentDB] = None,         
                     **kwargs
                     ) -> PineconeIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric

        if metadata is None:
            metadata = {}
        # if "_unifai_embedding_config" not in metadata:
        #     metadata["_unifai_embedding_config"] = ",".join((
        #         str(embedding_provider),
        #         str(embedding_model),
        #         str(dimensions),
        #         str(distance_metric)
        #     ))


        index_kwargs = {**self.default_index_kwargs, **kwargs}
        index_kwargs["dimension"] = dimensions
        index_kwargs["metric"] = distance_metric

        
        if spec := index_kwargs.get("spec", None): 
            spec_type = None
        elif spec := index_kwargs.pop("serverless_spec", None): 
            spec_type = "serverless"
        elif spec := index_kwargs.pop("pod_spec", None):
            spec_type = "pod"
        else:
            raise ValueError("No spec provided for index creation. Must provide either 'spec', 'serverless_spec', or 'pod_spec' with either dict ServerlessSpec or PodSpec")

        if isinstance(spec, dict):
            spec = convert_spec(spec, spec_type)
            index_kwargs["spec"] = spec

        self.client.create_index(name=name, **index_kwargs)
        pinecone_index = self.client.Index(name=name)

        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
        )

        index = PineconeIndex(
            wrapped=pinecone_index,
            name=name,
            metadata=metadata,
            embedding_function=embedding_function,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            **index_kwargs
        )
        self.indexes[name] = index
        return index
    

    @convert_exceptions
    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[LLMProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  document_db: Optional[DocumentDB] = None,
                  **kwargs                                    
                  ) -> PineconeIndex:
        if index := self.indexes.get(name):
            return index
        
        index_kwargs = {**self.default_index_kwargs, **kwargs}
        pinecone_index = self.client.Index(name=name)
        
        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
        )

        index = PineconeIndex(
            wrapped=pinecone_index,
            name=name,
            metadata={},
            embedding_function=embedding_function,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            **index_kwargs
        )
        self.indexes[name] = index
        return index        

    
    @convert_exceptions
    def count_indexes(self) -> int:
        return len(self.list_indexes())


    @convert_exceptions
    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop
                     ) -> list[str]:
        return [index.name for index in self.client.list_indexes()][limit_offest_slice(limit, offset)]



    @convert_exceptions
    def delete_index(self, name: str, **kwargs):
        self.indexes.pop(name, None)
        self.client.delete_index(name, **kwargs)


