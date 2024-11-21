from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Collection, Generator, Self

from unifai.exceptions import DocumentNotFoundError, DocumentWriteError, DocumentDeleteError
from ._base_document_db import DocumentDB
from ...types import Document

T = TypeVar("T")

class DictDocumentDB(DocumentDB):
    provider = "dict"

    def __init__(self, documents: Optional[dict[str, str]] = None):
        self.documents = documents if documents is not None else {}

    def get_documents(self, ids: Iterable[str]) -> Iterable[Document]:
        return (Document(id=id, text=text, metadata=metadata) for id, (text, metadata) in zip(ids, self.get_texts_and_metadatas(ids)))
    
    def upsert_documents(self, documents: Iterable[Document]) -> None:
        for document in documents:
            self.upsert_text(document.id, document.text)
        
    def get_text(self, id: str) -> str:
        try:
            return self.documents[id]
        except KeyError as e:
            raise DocumentNotFoundError(f"Document with id '{id}' not found", original_exception=e)

    def get_texts(self, ids: Collection[str]) -> Iterable[str]:
        yield from map(self.get_text, ids)

    def upsert_text(self, id: str, document: str) -> None:
        try:
            self.documents[id] = document
        except Exception as e:
            raise DocumentWriteError(f"Error writing document with id '{id}'", original_exception=e)

    def upsert_texts(self, ids: Collection[str], documents: Collection[str]) -> None:
        for id, document in zip(ids, documents):
            self.upsert_text(id, document)

    def delete_id(self, id: str) -> None:
        try:
            del self.documents[id]
        except Exception as e:
            raise DocumentDeleteError(f"Error deleting document with id '{id}'", original_exception=e)

    def delete_ids(self, ids: Collection[str]) -> None:
        for id in ids:
            del self.documents[id]       