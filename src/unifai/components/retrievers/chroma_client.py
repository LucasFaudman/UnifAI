from typing import Optional, Literal

from ...types import EmbeddingProvider
from .._base_component import convert_exceptions
from ..document_dbs._base_document_db import DocumentDB
from ..base_adapters.chroma_base import ChromaAdapter
from ._base_vector_db_client import VectorDBClient
from .chroma_index import ChromaIndex


class ChromaClient(ChromaAdapter, VectorDBClient):
                                       
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
                     ) -> ChromaIndex:
        
        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric
        document_db = document_db or self.default_document_db

        if metadata is None:
            metadata = {}
        if "_unifai_embedding_config" not in metadata:
            metadata["_unifai_embedding_config"] = ",".join((
                str(embedding_provider),
                str(embedding_model),
                str(dimensions),
                str(distance_metric)
            ))

        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions
        )

        index_kwargs = {**self.default_index_kwargs, **kwargs}                
        collection = self.client.create_collection(
            name=name, 
            metadata=metadata,
            embedding_function=embedding_function,
            **index_kwargs
        )
        index = ChromaIndex(
            wrapped=collection,
            name=name,
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
                  ) -> ChromaIndex:
        if index := self.indexes.get(name):
            return index
        
        index_kwargs = {**self.default_index_kwargs, **kwargs}

        if not (embedding_provider or embedding_model or dimensions or distance_metric):
            # get by name, extract metadata, and use that to create the index
            collection = self.client.get_collection(name=name, **index_kwargs)
            if not (embedding_config := collection.metadata.get("_unifai_embedding_config")):
                raise ValueError(f"Index {name} does not have an embedding config and kwargs are not provided")            
            embedding_provider, embedding_model, dimensions, distance_metric = (
                config_val if config_val != "None" else None for config_val in embedding_config.split(",")
            )
            dimensions = int(dimensions) if dimensions else None


        embedding_provider = embedding_provider or self.default_embedding_provider
        embedding_model = embedding_model or self.default_embedding_model
        dimensions = dimensions or self.default_dimensions
        distance_metric = distance_metric or self.default_distance_metric 
        document_db = document_db or self.default_document_db               

        embedding_function = self.get_embedding_function(
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions
        )

        collection = self.client.get_collection(
            name=name,
            embedding_function=embedding_function,
            **index_kwargs
        )
        
        index = ChromaIndex(
            wrapped=collection,
            name=name,
            metadata=collection.metadata,
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
