from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from unifai.exceptions import DocumentNotFoundError, DocumentWriteError, DocumentDeleteError
from ._base_document_db import DocumentDB

T = TypeVar("T")


class DictDocumentDB(DocumentDB):
    provider = "dict"


    def __init__(self, documents: dict[str, str]):
        self.documents = documents


    def get_document(self, id: str) -> str:
        try:
            return self.documents[id]
        except KeyError as e:
            raise DocumentNotFoundError(f"Document with id '{id}' not found", original_exception=e)


    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        yield from map(self.get_document, ids)


    def set_document(self, id: str, document: str) -> None:
        try:
            self.documents[id] = document
        except Exception as e:
            raise DocumentWriteError(f"Error writing document with id '{id}'", original_exception=e)


    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        for id, document in zip(ids, documents):
            self.set_document(id, document)


    def delete_document(self, id: str) -> None:
        try:
            del self.documents[id]
        except Exception as e:
            raise DocumentDeleteError(f"Error deleting document with id '{id}'", original_exception=e)


    def delete_documents(self, ids: Collection[str]) -> None:
        for id in ids:
            del self.documents[id]        