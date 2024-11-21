from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from json import dumps as json_dumps, loads as json_loads

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, GetResult, QueryResult
from unifai.exceptions import UnifAIError, DocumentDBAPIError, DocumentNotFoundError, DocumentReadError, DocumentWriteError, DocumentDeleteError
from ._base_document_db import DocumentDB, Document

from sqlite3 import connect as sqlite_connect, Error as SQLiteError

T = TypeVar("T")

class SQLiteDocumentDB(DocumentDB):
    provider = "sqlite"
    
    def __init__(
            self,
            db_path: str = ":memory:", 
            table_name: str = "documents", 
            connect_on_init: bool = True,
            create_table_on_init: bool = True,
            **connection_kwargs
            ):
        self.db_path = db_path
        self.table_name = table_name
        self.connection_kwargs = connection_kwargs
        if "database" in self.connection_kwargs:
            self.db_path = self.connection_kwargs.pop("database")
        if connect_on_init:
            self.connect()
            if create_table_on_init:
                self.create_document_table()

    def connect(self):
        try:
            self._connection = sqlite_connect(self.db_path, **self.connection_kwargs)
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error connecting to SQLite database at '{self.db_path}'", original_exception=e)

    @property
    def connection(self):
        if not hasattr(self, "_connection") or not self._connection:
            self.connect()
        return self._connection

    def close(self):
        try:
            self.connection.close()
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error closing connection to SQLite database at '{self.db_path}'", original_exception=e)

    def __del__(self):
        self.close()

    def create_document_table(self, table_name: Optional[str] = None):
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name or self.table_name} (id TEXT PRIMARY KEY, document TEXT)")
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentDBAPIError(f"Error creating document table '{table_name or self.table_name}'", original_exception=e)



    def get_texts(self, ids: list[str]) -> Iterable[str]:
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


    def upsert_texts(self, ids: list[str], documents: list[str]) -> None:
        try:
            for id, document in zip(ids, documents):
                self.connection.cursor().execute("INSERT OR REPLACE INTO documents (id, document) VALUES (?, ?)", (id, document))
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentWriteError(f"Error writing documents with ids '{ids}'", original_exception=e)
        

    def delete_ids(self, ids: list[str]) -> None:
        try:
            self.connection.cursor().execute(f"DELETE FROM {self.table_name} WHERE id IN ({','.join('?' for _ in ids)})", ids)
            self.connection.commit()
        except SQLiteError as e:
            raise DocumentDeleteError(f"Error deleting documents with ids '{ids}'", original_exception=e)

    # def create_document_table(self, table_name: Optional[str] = None):
    #     try:
    #         cursor = self.connection.cursor()
    #         cursor.execute(f"""
    #             CREATE TABLE IF NOT EXISTS {table_name or self.table_name} (
    #                 id TEXT PRIMARY KEY,
    #                 text TEXT,
    #                 metadata TEXT
    #             )
    #         """)
    #         self.connection.commit()
    #     except SQLiteError as e:
    #         raise DocumentDBAPIError(f"Error creating document table '{table_name or self.table_name}'", original_exception=e)

    # def get_documents(self, ids: Collection[str]) -> Iterable[Document]:
    #     try:
    #         cursor = self.connection.cursor()
    #         placeholders = ','.join('?' for _ in ids)
    #         cursor.execute(
    #             f"SELECT id, text, metadata FROM {self.table_name} WHERE id IN ({placeholders})",
    #             list(ids)
    #         )
    #         for row in cursor.fetchall():
    #             yield Document(
    #                 id=row[0],
    #                 text=row[1],
    #                 metadata=json_loads(row[2]) if row[2] else None
    #             )
    #     except SQLiteError as e:
    #         raise DocumentReadError(f"Error reading documents with ids '{ids}'", original_exception=e)

    # def update_documents(self, documents: Collection[Document]) -> None:
    #     try:
    #         cursor = self.connection.cursor()
    #         for doc in documents:
    #             cursor.execute(
    #                 f"INSERT OR REPLACE INTO {self.table_name} (id, text, metadata) VALUES (?, ?, ?)",
    #                 (
    #                     doc.id,
    #                     doc.text,
    #                     json_dumps(doc.metadata) if doc.metadata else None
    #                 )
    #             )
    #         self.connection.commit()
    #     except SQLiteError as e:
    #         raise DocumentWriteError(f"Error writing documents", original_exception=e)

    # def delete_documents(self, ids: Collection[str]) -> None:
    #     try:
    #         cursor = self.connection.cursor()
    #         placeholders = ','.join('?' for _ in ids)
    #         cursor.execute(
    #             f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
    #             tuple(ids)
    #         )
    #         self.connection.commit()
    #     except SQLiteError as e:
    #         raise DocumentDeleteError(f"Error deleting documents with ids '{ids}'", original_exception=e)