from typing import TYPE_CHECKING, Optional, Literal

from pinecone import ServerlessSpec, PodSpec

from ...exceptions import ProviderUnsupportedFeatureError
from ...types import EmbeddingProvider, EmbeddingTaskTypeInput
from .._base_component import convert_exceptions
from ..document_dbs._base_document_db import DocumentDB
from ..base_adapters.pinecone_base import PineconeAdapter
from ._base_vector_db_client import VectorDBClient
from .pinecone_index import PineconeIndex

if TYPE_CHECKING:
    from pinecone.grpc import GRPCIndex

class PineconeClient(PineconeAdapter, VectorDBClient):
    index_type = PineconeIndex
    default_spec = ServerlessSpec(cloud="aws", region="us-west-1")

    def _validate_distance_metric(self, distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]]) -> str:
        if distance_metric in ("cosine", None):
            return "cosine"
        if distance_metric in ("dotproduct", "ip"):
            return "ip"
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
    def _create_wrapped_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> "GRPCIndex":
        
        if spec := index_kwargs.get("spec"): 
            spec_type = None
        elif spec := index_kwargs.pop("serverless_spec", None): 
            spec_type = "serverless"
        elif spec := index_kwargs.pop("pod_spec", None):
            spec_type = "pod"
        else:
            spec = self.default_spec
            # raise ValueError("No spec provided for index creation. Must provide either 'spec', 'serverless_spec', or 'pod_spec' with either dict ServerlessSpec or PodSpec")

        if isinstance(spec, dict):
            if spec_type is None:                
                if (spec_type := spec.get("type")) is None:
                    raise KeyError("No spec type provided. Must provide 'type' key with either 'serverless' or 'pod' when spec is a dict")
            if spec_type == "serverless":
                spec = ServerlessSpec(**spec)
            elif spec_type == "pod":
                spec = PodSpec(**spec) 
            else:
                raise ValueError(f"Invalid spec type: {spec_type}. Must be either 'serverless' or 'pod'")

        index_kwargs["spec"] = spec
        index_kwargs["dimension"] = dimensions or self.default_dimensions
        index_kwargs["metric"] = distance_metric or self.default_distance_metric        
        self.client.create_index(name=name, **index_kwargs)
        return self.client.Index(name=name)
    
    @convert_exceptions
    def _get_wrapped_index(
            self, 
            name: str,
            dimensions: Optional[int] = None,
            distance_metric: Optional[Literal["cosine", "dotproduct",  "euclidean", "ip", "l2"]] = None,
            document_db: Optional[DocumentDB] = None,
            embedding_provider: Optional[EmbeddingProvider] = None,
            embedding_model: Optional[str] = None,
            embed_document_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_query_task_type: Optional[EmbeddingTaskTypeInput] = None,
            embed_kwargs: Optional[dict] = None,
            **index_kwargs
                     ) -> "GRPCIndex":
        return self.client.Index(name=name)


    @convert_exceptions
    def count_indexes(self) -> int:
        return len(self.list_indexes())


    @convert_exceptions
    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop
                     ) -> list[str]:
        if limit is not None and offset is not None:
            limit_offset_slice = slice(offset, limit + offset)
        elif limit is None and offset is not None:
            limit_offset_slice = slice(offset, None)
        elif offset is None:
            limit_offset_slice = slice(limit)
        return [index.name for index in self.client.list_indexes()][limit_offset_slice]

    @convert_exceptions
    def delete_index(self, name: str, **kwargs):
        self.indexes.pop(name, None)
        self.client.delete_index(name, **kwargs)


