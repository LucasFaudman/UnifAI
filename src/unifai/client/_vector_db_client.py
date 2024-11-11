from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.retrievers._base_vector_db_index import VectorDBIndex
    from ..components.retrievers._base_vector_db_client import VectorDBClient
    from ..components.document_dbs._base_document_db import DocumentDB


from ._embed_client import UnifAIEmbedClient
from ._document_db_client import UnifAIDocumentDBClient

class UnifAIVectorDBClient(UnifAIEmbedClient, UnifAIDocumentDBClient):

    def get_vector_db(self, 
                      provider: Optional[str] = None, 
                      default_embedding_provider: Optional[str] = None,
                      default_embedding_model: Optional[str] = None,
                      default_dimensions: int = 768,
                      default_distance_metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine",
                      default_index_kwargs: Optional[dict] = None,
                      default_document_db: Optional["DocumentDB"] = None,                                                  
                      **client_kwargs) -> "VectorDBClient":
        provider = provider or self.config.default_providers["vector_db"]
        client_kwargs.update(
            default_embedding_provider=default_embedding_provider,
            default_embedding_model=default_embedding_model,
            default_dimensions=default_dimensions,
            default_distance_metric=default_distance_metric,
            default_index_kwargs=default_index_kwargs,
            default_document_db=default_document_db
        )
        if "embed" not in client_kwargs:
            client_kwargs["embed"] = self.embed
        return self.get_component(provider, "vector_db", **client_kwargs)

    def get_or_create_index(self, 
                            name: str,
                            vector_db_provider: Optional[str] = None, 
                            document_db_provider: Optional[str] = None,                           
                            embedding_provider: Optional[str] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None, 
                            index_metadata: Optional[dict] = None,
                            **kwargs
                            ) -> "VectorDBIndex":
        if dimensions is None:
            dimensions = self.get_embedder(embedding_provider).get_model_dimensions(embedding_model)
        document_db = self.get_document_db(document_db_provider) if document_db_provider else None
        
        return self.get_vector_db(vector_db_provider).get_or_create_index(
            name=name,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            dimensions=dimensions,
            distance_metric=distance_metric,
            document_db=document_db,
            metadata=index_metadata,
            **kwargs
        )  
    
