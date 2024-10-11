from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from pydantic import BaseModel

T = TypeVar("T")

class DocumentDB:
    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        raise NotImplementedError
    
    def get_document(self, id: str) -> str:
        return next(iter(self.get_documents([id])))
    
    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        raise NotImplementedError
    
    def set_document(self, id: str, document: str) -> None:
        self.set_documents([id], [document])
    
    def delete_documents(self, ids: Collection[str]) -> None:
        raise NotImplementedError
    
    def delete_document(self, id: str) -> None:
        self.delete_documents([id])
    

class DictDocumentDB(DocumentDB):
    def __init__(self, documents: dict[str, str]):
        self.documents = documents

    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        return (self.documents[id] for id in ids)
    
    def get_document(self, id: str) -> str:
        return self.documents[id]

    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        for id, document in zip(ids, documents):
            self.documents[id] = document

    def set_document(self, id: str, document: str) -> None:
        self.documents[id] = document

    def delete_documents(self, ids: Collection[str]) -> None:
        for id in ids:
            del self.documents[id]
    
    def delete_document(self, id: str) -> None:
        del self.documents[id]


class VectorDBIndex(UnifAIExceptionConverter):
    provider = "base_vector_db"

    def __init__(self,
                 wrapped: Any,
                 name: str,
                 metadata: Optional[dict] = None,
                 embedding_function: Optional[Callable] = None,
                 embedding_provider: Optional[LLMProvider] = None,
                 embedding_model: Optional[str] = None,
                 dimensions: Optional[int] = None,
                 distance_metric: Optional[Literal["cosine", "euclidean", "dotproduct"]] = None,
                 document_db: Optional[DocumentDB] = None,
                 **kwargs
                 ):
        
        self.wrapped = wrapped
        self.name = name
        self.metadata = metadata or {}
        self.embedding_function = embedding_function
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.dimensions = dimensions
        self.distance_metric = distance_metric
        self.document_db = document_db
        self.kwargs = kwargs


    def count(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")
    

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
        
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def add(self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            documents: Optional[list[str]] = None,
            embeddings: Optional[list[Embedding]] = None,
            **kwargs
            ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")


    def update(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
          
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def upsert(self,
                ids: list[str],
                metadatas: Optional[list[dict]] = None,
                documents: Optional[list[str]] = None,
                embeddings: Optional[list[Embedding]] = None,
                **kwargs
                ) -> Self:
        
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete(self, 
               ids: list[str],
               where: Optional[dict] = None,
               where_document: Optional[dict] = None,
               **kwargs
               ) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")

        
    def get(self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "documents", "embeddings"]] = ["metadatas", "documents"],
            **kwargs
            ) -> VectorDBGetResult:
        raise NotImplementedError("This method must be implemented by the subclass")


    def query(self,              
              query_text: Optional[str] = None,
              query_embedding: Optional[Embedding] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> VectorDBQueryResult:
        
        raise NotImplementedError("This method must be implemented by the subclass")

    def query_many(self,              
              query_texts: Optional[list[str]] = None,
              query_embeddings: Optional[list[Embedding]] = None,
              n_results: int = 10,
              where: Optional[dict] = None,
              where_document: Optional[dict] = None,
              include: list[Literal["metadatas", "documents", "embeddings", "distances"]] = ["metadatas", "documents", "distances"],
              **kwargs
              ) -> list[VectorDBQueryResult]:
        
        raise NotImplementedError("This method must be implemented by the subclass")    


    def list_ids(self, **kwargs) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def delete_all(self, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")