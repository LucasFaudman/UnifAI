from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_vector_db_client import VectorDBIndex, VectorDBClient

from unifai.types import ResponseInfo, Embedding, Embeddings, Usage, AIProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError, BadRequestError
from unifai.wrappers._base_client_wrapper import UnifAIExceptionConverter, convert_exceptions

from pinecone.grpc import PineconeGRPC as Pinecone, GRPCIndex  
from pinecone import ServerlessSpec, PodSpec, Index
from pinecone.exceptions import PineconeException, PineconeApiException




from itertools import zip_longest

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
 
class PineconeExceptionConverter(UnifAIExceptionConverter):
    def convert_exception(self, exception: PineconeException) -> UnifAIError:
        if not isinstance(exception, PineconeApiException):
            return UnifAIError(
                message=exception.args[0],
                original_exception=exception
            )
        status_code=exception.status
        if status_code is not None:
            unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
            unifai_exception_type = UnknownAPIError

        return unifai_exception_type(
            message=exception.args[0], 
            status_code=status_code,
            original_exception=exception
        )   


class PineconeIndex(VectorDBIndex, PineconeExceptionConverter):



    def __init__(self,
                 wrapped: GRPCIndex,
                 name: str,
                 metadata: Optional[dict] = None,
                 embedding_provider: Optional[AIProvider] = None,
                 embedding_model: Optional[str] = None,
                 embedding_function: Optional[Callable] = None,
                 dimensions: Optional[int] = None,
                 distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                #  spec = dict|ServerlessSpec|PodSpec,
                #  spec_type: Literal["pod", "serverless", None] = None,
                 **kwargs
                 ):
        
        self.wrapped = wrapped
        self.name = name
        self.metadata = metadata or {}
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_function = embedding_function
        self.dimensions = dimensions
        self.distance_metric = distance_metric  
        self.kwargs = kwargs


    @convert_exceptions
    def count(self) -> int:
        return self.wrapped.describe_index_stats().num_docs
    
    @convert_exceptions
    def modify(self, 
               new_name: Optional[str]=None, 
               new_metadata: Optional[dict]=None,
               embedding_provider: Optional[AIProvider] = None,
               embedding_model: Optional[str] = None,
               dimensions: Optional[int] = None,
               distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
               
               metadata_update_mode: Optional[Literal["replace", "merge"]] = "replace",

               ) -> Self:
        
        if new_name is not None:
            self.name = new_name
        if new_metadata is not None:
            if metadata_update_mode == "replace":
                self.metadata = new_metadata
            else:
                self.metadata.update(new_metadata)
                
        self.wrapped.modify(name=self.name, metadata=self.metadata)
        return self

            
    @convert_exceptions
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            ) -> Self:
        
        self.update(ids, metadatas, documents, embeddings)
        return self


    @convert_exceptions
    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
        metadata = metadatas or []
        documents = documents or []
        embeddings = embeddings or []

        for id, metadata, document, embedding in zip_longest(ids, metadata, documents, embeddings):
            if document and not embedding and self.embedding_function:
                embedding = self.embedding_function(document)
            self.wrapped.update(
                id=id,
                set_metadata=metadata,
                embedding=embedding
            )

        return self
    

    @convert_exceptions
    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                ) -> Self:
        
        self.update(ids, metadatas, documents, embeddings)
        return self
    

    @convert_exceptions
    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               ) -> None:
        if where_document:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone")

        self.wrapped.delete(ids=ids, filters=where)


    def _query()

    @convert_exceptions
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"]
            ) -> VectorDBGetResult:
        
        if not ids:
            raise ValueError("Must provide ids to get")
        
        if where_document:
            raise ProviderUnsupportedFeatureError("where_document is not supported by Pinecone")
        
        
        if "embeddings" in include:
            include_values = True            
            embeddings = []
        else:
            include_values = False
            embeddings = None

        if "metadata" in include:
            include_metadatas = True
            metadatas = []
        else:
            include_metadatas = False
            metadatas = None

        # if "documents" in include:
        #     include_values = True
        #     documents = []
        # else:
        #     include_values = False
        #     documents = None

        for id in ids:
            result = self.wrapped.query(
                id=id, 
                top_k=1,
                include_values=include_values, 
                include_metadatas=include_metadatas
            )
            matches = result["matches"]
            if not matches:
                continue
            match = matches[0]

            if embeddings is not None:
                embeddings.append(match["values"])
            if metadatas is not None:
                metadatas.append(match["metadata"])
            

        get_result = self.wrapped.query(
            ids=ids, 
            where=where, 
            limit=limit, 
            offset=offset, 
            where_document=where_document, 
            include=include
        )
        return VectorDBGetResult(
            ids=get_result["ids"],
            metadatas=get_result["metadatas"],
            documents=get_result["documents"],
            embeddings=get_result["embeddings"],
            included=get_result["included"] 
        )
    
    @convert_exceptions
    def query(self,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              ) -> list[VectorDBQueryResult]:
        
        query_result = self.wrapped.query(
            query_embeddings=query_embeddings, 
            query_texts=query_texts, 
            n_results=n_results, 
            where=where, 
            where_document=where_document, 
            include=include
        )

        included = query_result["included"]
        empty_tuple = ()
        return [
            VectorDBQueryResult(
                ids=ids,
                metadatas=metadatas,
                documents=documents,
                embeddings=embeddings,
                distances=distances,
                included=included
            ) for ids, metadatas, documents, embeddings, distances in zip_longest(
                query_result["ids"],
                query_result["metadatas"] or empty_tuple,
                query_result["documents"] or empty_tuple,
                query_result["embeddings"] or empty_tuple,
                query_result["distances"] or empty_tuple,
                fillvalue=None
            )
        ]    



        

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
                     embedding_provider: Optional[AIProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                    #  spec = dict|ServerlessSpec|PodSpec,
                    #  spec_type: Literal["pod", "serverless", None] = None,                     
                     **kwargs
                     ) -> PineconeIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        # dimensions = dimensions or self.default_dimensions
        # distance_metric = distance_metric or self.default_distance_metric

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
        index_kwargs["dimension"] = dimensions or self.default_dimensions
        index_kwargs["metric"] = distance_metric or self.default_distance_metric

        if spec := (
            index_kwargs.get("spec", None) 
            or index_kwargs.pop("serverless_spec", None) 
            or index_kwargs.pop("pod_spec", None)
            ) is None:
            raise ValueError("No spec provided for index creation. Must provide either 'spec', 'serverless_spec', or 'pod_spec' with either dict ServerlessSpec or PodSpec")

        if isinstance(spec, dict):
            spec = convert_spec(spec, index_kwargs.pop("spec_type", None))
            index_kwargs["spec"] = spec

        self.client.create_index(name=name, **index_kwargs)
        pinecone_index = self.client.Index(name=name)

        index = PineconeIndex(
            wrapped=pinecone_index,
            name=name,
            metadata=metadata,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            **index_kwargs
        )
        self.indexes[name] = index
        return index
    

    @convert_exceptions
    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[AIProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  **kwargs                                    
                  ) -> PineconeIndex:
        if index := self.indexes.get(name):
            return index
        
        index_kwargs = {**self.default_index_kwargs, **kwargs}
        pinecone_index = self.client.Index(name=name)
        
        index = PineconeIndex(
            wrapped=pinecone_index,
            name=name,
            metadata={},
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
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
        return self.client.list_indexes.names()[limit_offest_slice(limit, offset)]



    @convert_exceptions
    def delete_index(self, name: str):
        self.indexes.pop(name, None)
        self.client.delete_index(name)
