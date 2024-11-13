from typing import TYPE_CHECKING, Optional, Literal

from ...exceptions import ProviderUnsupportedFeatureError

from ...types import EmbeddingProvider, EmbeddingTaskTypeInput, Embeddings
from .._base_component import convert_exceptions
from ..document_dbs._base_document_db import DocumentDB
from ..base_adapters.chroma_base import ChromaAdapter
from ._base_vector_db_client import VectorDBClient
from .chroma_index import ChromaIndex
if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection as ChromaCollection

class ChromaClient(ChromaAdapter, VectorDBClient):
    index_type = ChromaIndex

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
            ) -> "ChromaCollection":
        metadata = index_kwargs.pop("metadata", {})
        metadata["hnsw:space"] = distance_metric
        return self.client.create_collection(
            name=name, 
            metadata=metadata,
            embedding_function=None,
            **index_kwargs
        )

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
        ) -> "ChromaCollection":
        return self.client.get_collection(
            name=name, 
            embedding_function=None,
            **index_kwargs
        )      


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
