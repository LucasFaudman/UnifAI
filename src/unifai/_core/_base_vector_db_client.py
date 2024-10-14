from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_adapter import UnifAIAdapter
from ._base_vector_db_index import VectorDBIndex, DocumentDB

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, EmbeddingProvider, Usage, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError, NotFoundError


T = TypeVar("T")

class VectorDBCompatibleEmbeddingFunction:
    """
    Chroma style EmbeddingFunction object that takes a list of documents and returns a list of embeddings.    
    - Stores the a reference to the embedding function, the embedding provider, model, and dimensions used to generate the embeddings.
    - Stores a list of response infos for the embeddings. (usage, model, etc)
    Args:
        embed (Callable[..., Embeddings]): Embedding function that takes a list of documents and returns a list of embeddings.
        embedding_provider (Optional[str]): The embedding provider to use.
        model (Optional[str]): The embedding model to use.
        dimensions (Optional[int]): The number of dimensions to use.
        response_infos (Optional[list[ResponseInfo]]): List of response infos (usage, model, etc) for the embeddings.
    """

    def __init__(
            self,
            embed: Callable[..., Embeddings],
            embedding_provider: Optional[EmbeddingProvider] = None,
            model: Optional[str] = None,
            dimensions: Optional[int] = None,
            response_infos: Optional[list[ResponseInfo]] = None,
    ):
        self.embed = embed
        self.embedding_provider = embedding_provider
        self.model = model
        self.dimensions = dimensions
        self.response_infos = response_infos or []

    def __call__(self, input: list[str]) -> list[Embedding]:
        """
        Embed a list of documents.
        
        Args:
            input (list[str]): List of documents to embed.

        Returns:
            list[Embedding]: List of embeddings for the input documents. (list of lists of floats)           
        """
        print(f"Embedding {len(input)} documents")
        embed_result = self.embed(
            input=input,
            model=self.model,
            provider=self.embedding_provider,
            dimensions=self.dimensions
        )
        if embed_result.response_info:
            self.response_infos.append(embed_result.response_info)
        return embed_result.list()
    

class VectorDBClient(UnifAIAdapter):
    provider = "base_vector_db"

    def __init__(self, 
                 embed: Callable[..., Embeddings],
                 default_embedding_provider: EmbeddingProvider,
                 default_embedding_model: Optional[str] = None,
                 default_dimensions: int = 1536,
                 default_distance_metric: Literal["cosine", "euclidean", "dotproduct"] = "cosine",
                 default_index_kwargs: Optional[dict] = None,
                 default_document_db: Optional[DocumentDB] = None,
                 **client_kwargs
                 ):
        super().__init__(**client_kwargs)

        self.embed = embed
        self.indexes = {}

        self.default_embedding_provider = default_embedding_provider
        self.default_embedding_model = default_embedding_model
        self.default_dimensions = default_dimensions
        self.default_distance_metric = default_distance_metric
        self.default_index_kwargs = default_index_kwargs or {}
        self.default_document_db = default_document_db


    def get_embedding_function(self,
                               embedding_provider: Optional[EmbeddingProvider] = None,
                               embedding_model: Optional[str] = None,
                               dimensions: Optional[int] = None,
                               response_infos: Optional[list[ResponseInfo]] = None,
                                 ) -> VectorDBCompatibleEmbeddingFunction:
        return VectorDBCompatibleEmbeddingFunction(
            embed=self.embed,
            embedding_provider=embedding_provider or self.default_embedding_provider,
            model=embedding_model or self.default_embedding_model,
            dimensions=dimensions or self.default_dimensions,
            response_infos=response_infos
        )


    def create_index(self, 
                     name: str,
                     embedding_provider: Optional[EmbeddingProvider] = None,
                     embedding_model: Optional[str] = None,
                     dimensions: Optional[int] = None,
                     distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                     document_db: Optional[DocumentDB] = None,
                     metadata: Optional[dict] = None,
                     **kwargs
                     ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def get_index(self, 
                  name: str,
                  embedding_provider: Optional[EmbeddingProvider] = None,
                  embedding_model: Optional[str] = None,
                  dimensions: Optional[int] = None,
                  distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                  document_db: Optional[DocumentDB] = None,
                  **kwargs  
                  ) -> VectorDBIndex:
        raise NotImplementedError("This method must be implemented by the subclass")


    def get_or_create_index(self, 
                            name: str,
                            embedding_provider: Optional[EmbeddingProvider] = None,
                            embedding_model: Optional[str] = None,
                            dimensions: Optional[int] = None,
                            distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                            document_db: Optional[DocumentDB] = None,
                            metadata: Optional[dict] = None,                            
                            **kwargs
                            ) -> VectorDBIndex:
        try:
            return self.get_index(
                name=name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                dimensions=dimensions,
                distance_metric=distance_metric,
                document_db=document_db,
                **kwargs
            )
        except (BadRequestError, NotFoundError):
            return self.create_index(
                name=name,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                dimensions=dimensions,
                distance_metric=distance_metric,
                document_db=document_db,
                metadata=metadata,
                **kwargs
            )
    

    def list_indexes(self,
                     limit: Optional[int] = None,
                     offset: Optional[int] = None, # woop woop,
                     **kwargs
                     ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_index(self, name: str, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_indexes(self, names: Collection[str], **kwargs) -> None:
        for name in names:
            self.delete_index(name, **kwargs)


    def delete_all_indexes(self, **kwargs) -> None:
        self.delete_indexes(self.list_indexes(), **kwargs)


    def count_indexes(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")


    def count(self, name: str, **kwargs) -> int:
        return self.get_index(name).count(**kwargs)
    

    def modify_index(self, 
                     name: str, 
                     new_name: Optional[str]=None,
                     new_metadata: Optional[dict]=None,
                     **kwargs
                     ) -> VectorDBIndex:
        return self.get_index(name).modify(new_name, new_metadata, **kwargs)   


    def add(self,
            name: str,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> VectorDBIndex:
        return self.get_index(name).add(ids, metadatas, documents, embeddings, **kwargs)
    

    def update(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               **kwargs
               ) -> VectorDBIndex:
        return self.get_index(name).update(ids, metadatas, documents, embeddings, **kwargs)
    

    def upsert(self,
               name: str,
               ids: list[str],
               metadatas: Optional[list[dict]] = None,
               documents: Optional[list[str]] = None,
               embeddings: Optional[list[Embedding]] = None,
               **kwargs
               ) -> VectorDBIndex:
          return self.get_index(name).upsert(ids, metadatas, documents, embeddings, **kwargs)
    

    def delete(self, 
               name: str,
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        return self.get_index(name).delete(ids, where, where_document, **kwargs)


    def get(self,
            name: str,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["embeddings", "metadatas", "documents"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        return self.get_index(name).get(ids, where, limit, offset, where_document, include, **kwargs)


    def query(self,
              name: str,
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,         
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:
        return self.get_index(name).query(query_text, query_embedding, n_results, where, where_document, include, **kwargs)    
    

    def query_many(self,
              name: str,
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,              
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["embeddings", "metadatas", "documents", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]:
        return self.get_index(name).query_many(query_texts, query_embeddings, n_results, where, where_document, include, **kwargs)        