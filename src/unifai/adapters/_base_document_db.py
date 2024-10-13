from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_adapter import BaseAdapter

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from pydantic import BaseModel

T = TypeVar("T")

class DocumentDB(BaseAdapter):
    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def get_document(self, id: str) -> str:
        return next(iter(self.get_documents([id])))
    
    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def set_document(self, id: str, document: str) -> None:
        self.set_documents([id], [document])
    
    def delete_documents(self, ids: Collection[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

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


class SQLITEDocumentDB(DocumentDB):
    
    def import_client(self) -> Callable:
        from sqlite3 import connect
        return connect


    def init_client(self, **client_kwargs) -> Any:
        if client_kwargs:
            self.client_kwargs.update(client_kwargs)
        if "database" not in self.client_kwargs:
            self.client_kwargs["database"] = self.client_kwargs.pop("db_path", ":memory:")
        self.table_name = self.client_kwargs.pop("table_name", "documents")
        self._client = self.import_client()(**self.client_kwargs)
        self._client.cursor().execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (id TEXT PRIMARY KEY, document TEXT)")
        self._client.commit()
        return self._client


    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        cursor = self.client.cursor()
        cursor.execute(f"SELECT id, document FROM {self.table_name} WHERE id IN ({','.join('?' for _ in ids)})", ids)
        documents_dict = {row[0]: row[1] for row in cursor.fetchall()}
        return (documents_dict.get(id, None) for id in ids)


    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        for id, document in zip(ids, documents):
            self.client.cursor().execute("INSERT OR REPLACE INTO documents (id, document) VALUES (?, ?)", (id, document))
        self.client.commit()


    def delete_documents(self, ids: Collection[str]) -> None:
        self.client.cursor().execute(f"DELETE FROM {self.table_name} WHERE id IN ({','.join('?' for _ in ids)})", ids)
        self.client.commit()


    def close_connection(self):
        self.client.close()


    def __del__(self):
        self.close_connection()