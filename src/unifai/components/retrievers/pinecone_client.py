from typing import Optional, Literal

from pinecone import ServerlessSpec, PodSpec

from ...types import EmbeddingProvider
from .._base_component import convert_exceptions
from ..document_dbs._base_document_db import DocumentDB
from ..base_adapters.pinecone_base import PineconeAdapter
from ._base_vector_db_client import VectorDBClient
from .pinecone_index import PineconeIndex

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


class PineconeClient(PineconeAdapter, VectorDBClient):
                                       
    @convert_exceptions                           
    def create_index(self, 
                     name: str,
                     embedding_provider: Optional[EmbeddingProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     document_db: Optional[DocumentDB] = None,         
                     metadata: Optional[dict] = None,
                     **kwargs
                     ) -> PineconeIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric
        document_db = document_db or self.default_document_db
        

        # if metadata is None:
        #     metadata = {}
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
            _wrapped=pinecone_index,
            name=name,
            embedding_function=embedding_function,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            metadata=metadata,
            **index_kwargs
        )
        self.indexes[name] = index
        return index
    

    @convert_exceptions
    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[EmbeddingProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  document_db: Optional[DocumentDB] = None,
                  **kwargs                                    
                  ) -> PineconeIndex:
        if index := self.indexes.get(name):
            return index
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric
        document_db = document_db or self.default_document_db

        index_kwargs = {**self.default_index_kwargs, **kwargs}
        pinecone_index = self.client.Index(name=name)
        
        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
        )

        index = PineconeIndex(
            _wrapped=pinecone_index,
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


