from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Iterable, Generator, Self

# from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
# from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from .._base_component import UnifAIComponent
from ...types import Document, Documents

T = TypeVar("T")
def _next(iterable: Iterable[T]) -> T:
    return next(iter(iterable))

class DocumentDB(UnifAIComponent):
    provider = "document_db"

    def get_documents(self, ids: Iterable[str]) -> Iterable[Document]:
        return (Document(id=id, text=text, metadata=metadata) for id, (text, metadata) in zip(ids, self.get_texts_and_metadatas(ids)))
    
    def get_document(self, id: str) -> Document:
        return _next(self.get_documents([id]))
    
    def upsert_documents(self, documents: Iterable[Document]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def upsert_document(self, document: Document) -> None:
        self.upsert_documents([document])
    
    def delete_documents(self, ids: Iterable[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def delete_document(self, id: str) -> None:
        self.delete_documents([id])
        
        
    def get_texts(self, ids: Iterable[str]) -> Iterable[str]:
        if self.get_texts_and_metadatas is DocumentDB.get_texts_and_metadatas:
            raise NotImplementedError("This method must be implemented by the subclass")
        return (text for text, metadata in self.get_texts_and_metadatas(ids))

    def get_text(self, id: str) -> str:
        return _next(self.get_texts([id]))

    def get_texts_and_metadatas(self, ids: Iterable[str]) -> Iterable[tuple[str, Optional[dict]]]:
        if self.get_texts is DocumentDB.get_texts:
            raise NotImplementedError("This method must be implemented by the subclass")
        return ((text, None) for text in self.get_texts(ids))
    
    def get_text_and_metadata(self, id: str) -> tuple[str, Optional[dict]]:
        return _next(self.get_texts_and_metadatas([id]))
    
    def upsert_texts(self, ids: Iterable[str], texts: Iterable[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def upsert_text(self, id: str, text: str) -> None:
        self.upsert_texts([id], [text])

    def upsert_metadatas(self, ids: Iterable[str], metadatas: Optional[Iterable[Optional[dict]]] = None) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def upsert_metadata(self, id: str, metadata: Optional[dict] = None) -> None:
        self.upsert_metadatas([id], [metadata] if metadata is not None else None)
        
    def upsert_texts_and_metadatas(self, ids: Iterable[str], texts: Iterable[str], metadatas: Optional[Iterable[Optional[dict]]] = None) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def upsert_text_and_metadata(self, id: str, text: str, metadata: Optional[dict] = None) -> None:
        self.upsert_texts_and_metadatas([id], [text], [metadata] if metadata else None)

    def delete_ids(self, ids: Iterable[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def delete_id(self, id: str) -> None:
        self.delete_ids([id])




