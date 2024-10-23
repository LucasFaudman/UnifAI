from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self


from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, DocumentDBAPIError, DocumentNotFoundError, DocumentReadError, DocumentWriteError, DocumentDeleteError
from ._base_document_db import DocumentDB

from sqlite3 import connect as sqlite_connect, Error as SQLiteError

T = TypeVar("T")

class SQLiteDocumentDB(DocumentDB):
    provider = "sqlite"
    
    def __init__(self, db_path: str = ":memory:", table_name: str = "documents", **connection_kwargs):
        self.db_path = db_path
        self.table_name = table_name
        self.connection_kwargs = connection_kwargs
        if "database" in self.connection_kwargs:
            self.db_path = self.connection_kwargs.pop("database")


    def connect(self):
        try:
            self.connection = sqlite_connect(self.db_path, **self.connection_kwargs)
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error connecting to SQLite database at '{self.db_path}'", original_exception=e)


    def close(self):
        try:
            self.connection.close()
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error closing connection to SQLite database at '{self.db_path}'", original_exception=e)


    def create_document_table(self, table_name: Optional[str] = None):
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name or self.table_name} (id TEXT PRIMARY KEY, document TEXT)")
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error creating document table '{table_name or self.table_name}'", original_exception=e)



    def get_documents(self, ids: list[str]) -> Iterable[str]:
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT id, document FROM {self.table_name} WHERE id IN ({','.join('?' for _ in ids)})", ids)
            documents_dict = {row[0]: row[1] for row in cursor.fetchall()}
        except SQLiteError as e:
            raise DocumentReadError(f"Error reading documents with ids '{ids}'", original_exception=e)
        
        for id in ids:
            if id in documents_dict:
                yield documents_dict[id]
            else:
                raise DocumentNotFoundError(f"Document with id '{id}' not found")


    def set_documents(self, ids: list[str], documents: list[str]) -> None:
        try:
            for id, document in zip(ids, documents):
                self.connection.cursor().execute("INSERT OR REPLACE INTO documents (id, document) VALUES (?, ?)", (id, document))
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentWriteError(f"Error writing documents with ids '{ids}'", original_exception=e)
        

    def delete_documents(self, ids: list[str]) -> None:
        try:
            self.connection.cursor().execute(f"DELETE FROM {self.table_name} WHERE id IN ({','.join('?' for _ in ids)})", ids)
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentDeleteError(f"Error deleting documents with ids '{ids}'", original_exception=e)
