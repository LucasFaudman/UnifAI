from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection, Callable, Iterator, Iterable, Generator, Self
from json import dumps as json_dumps, loads as json_loads

from ...types import Document, Documents, GetResult
from ...exceptions import UnifAIError, DBAPIError, DocumentNotFoundError, DocumentAlreadyExistsError, CollectionNotFoundError, CollectionAlreadyExistsError
from .._base_components._base_document_db import DocumentDB, DocumentDBCollection
from ...utils import combine_dicts, as_lists, limit_offset_slice
from ...configs.document_db_config import DocumentDBConfig, DocumentDBCollectionConfig
from ...types.annotations import CollectionName

from sqlite3 import connect as sqlite_connect, Error as SQLiteError, Connection

class ConnectionContextManager:
    def __init__(self, db: "SQLiteDocumentDB"):
        self.db = db

    def __enter__(self) -> Connection:
        return self.db.connection

    def __exit__(self, exc_type, exc_value, traceback):
        if self.db.db_path != ":memory:":  # Don't close in-memory databases
            self.db.close()

WrappedConnectionContextManager = Callable[[], ConnectionContextManager]

class SQLiteDocumentDBCollection(DocumentDBCollection[WrappedConnectionContextManager]):
    provider = "sqlite"

    def _setup(self):
        super()._setup()
        self.get_connection = self.wrapped
        self.table_name = self.name
        self.table_created = self.config.init_kwargs.get("table_created", False)
        if not self.table_created and self.config.init_kwargs.get("create_table_on_init", True):
            self.create_document_table()

    def create_document_table(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} (id TEXT PRIMARY KEY, text TEXT, metadata TEXT)")
            conn.commit()
        self.table_created = True

    def _build_filter_query(self, where: Optional[dict] = None, where_document: Optional[dict] = None) -> tuple[str, list]:
        """
        Construct SQL WHERE clause and parameters based on filter conditions.
        Supports nested filter operators like $eq, $ne, $gt, $lt, $and, $or, etc.
        """
        def parse_condition(key, condition):
            if not isinstance(condition, dict):
                return f"{key} = ?", [condition]
            
            conditions = []
            params = []
            for op, val in condition.items():
                if op == "$eq":
                    conditions.append(f"{key} = ?")
                    params.append(val)
                elif op == "$ne":
                    conditions.append(f"{key} != ?")
                    params.append(val)
                elif op == "$gt":
                    conditions.append(f"{key} > ?")
                    params.append(val)
                elif op == "$gte":
                    conditions.append(f"{key} >= ?")
                    params.append(val)
                elif op == "$lt":
                    conditions.append(f"{key} < ?")
                    params.append(val)
                elif op == "$lte":
                    conditions.append(f"{key} <= ?")
                    params.append(val)
                elif op == "$in":
                    placeholders = ','.join(['?'] * len(val))
                    conditions.append(f"{key} IN ({placeholders})")
                    params.extend(val)
                elif op == "$nin":
                    placeholders = ','.join(['?'] * len(val))
                    conditions.append(f"{key} NOT IN ({placeholders})")
                    params.extend(val)
                elif op == "$exists":
                    conditions.append(f"({key} IS {'NOT ' if not val else ''}NULL)")
                elif op == "$contains":
                    conditions.append(f"{key} LIKE ?")
                    params.append(f"%{val}%")
                elif op == "$not_contains":
                    conditions.append(f"{key} NOT LIKE ?")
                    params.append(f"%{val}%")
            
            return ' AND '.join(conditions), params

        where_clauses = []
        where_params = []

        # Handle metadata filters
        if where:
            for key, condition in where.items():
                if key == "$and":
                    sub_clauses = []
                    for sub_filter in condition:
                        sub_key, sub_condition = list(sub_filter.items())[0]
                        sub_where = {sub_key: sub_condition}
                        clause, params = self._build_filter_query(sub_where)
                        sub_clauses.append(clause)
                        where_params.extend(params)
                    where_clauses.append(f"({' AND '.join(sub_clauses)})")
                elif key == "$or":
                    sub_clauses = []
                    for sub_filter in condition:
                        sub_key, sub_condition = list(sub_filter.items())[0]
                        sub_where = {sub_key: sub_condition}
                        clause, params = self._build_filter_query(sub_where)
                        sub_clauses.append(clause)
                        where_params.extend(params)
                    where_clauses.append(f"({' OR '.join(sub_clauses)})")
                else:
                    clause, params = parse_condition(f"json_extract(metadata, '$.{key}')", condition)
                    where_clauses.append(clause)
                    where_params.extend(params)

        # Handle document text filters
        if where_document:
            for key, condition in where_document.items():
                if key == "$contains":
                    where_clauses.append("text LIKE ?")
                    where_params.append(f"%{condition}%")
                # Add more text filter conditions as needed

        return ' WHERE ' + ' AND '.join(where_clauses) if where_clauses else '', where_params

    def _count(self, **kwargs) -> int:
        """Count total number of documents in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT COUNT(*) FROM {self.table_name}"
            
            # Apply optional filters
            where_clause, where_params = self._build_filter_query(
                kwargs.get('where'), 
                kwargs.get('where_document')
            )
            query += where_clause
            
            cursor.execute(query, where_params)
            return cursor.fetchone()[0]

    def _list_ids(self, **kwargs) -> list[str]:
        """List all document IDs in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = f"SELECT id FROM {self.table_name}"
            
            # Apply optional filters and pagination
            where_clause, where_params = self._build_filter_query(
                kwargs.get('where'), 
                kwargs.get('where_document')
            )
            query += where_clause
            
            # Add optional limit and offset
            limit = kwargs.get('limit')
            offset = kwargs.get('offset', 0)
            if limit is not None:
                query += f" LIMIT {limit} OFFSET {offset}"
            
            cursor.execute(query, where_params)
            return [row[0] for row in cursor.fetchall()]

    def _add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
            ) -> Self:
        """Add documents to the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            ids, metadatas, texts = as_lists(ids, metadatas, texts)
            
            for id, metadata, text in zip(ids, metadatas or [None]*len(ids), texts or [None]*len(ids)):
                # Check if document already exists
                cursor.execute(f"SELECT 1 FROM {self.table_name} WHERE id = ?", (id,))
                if cursor.fetchone():
                    raise DocumentAlreadyExistsError(f"Document with ID {id} already exists")
                
                # Insert new document
                cursor.execute(
                    f"INSERT INTO {self.table_name} (id, text, metadata) VALUES (?, ?, ?)", 
                    (id, text, json_dumps(metadata) if metadata else None)
                )
            
            conn.commit()
        return self

    def _add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:
        """Add documents to the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for document in documents:
                # Check if document already exists
                cursor.execute(f"SELECT 1 FROM {self.table_name} WHERE id = ?", (document.id,))
                if cursor.fetchone():
                    raise DocumentAlreadyExistsError(f"Document with ID {document.id} already exists")
                
                # Insert new document
                cursor.execute(
                    f"INSERT INTO {self.table_name} (id, text, metadata) VALUES (?, ?, ?)", 
                    (document.id, document.text, json_dumps(document.metadata))
                )
            
            conn.commit()
        return self

    def _update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        """Update existing documents in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            ids, metadatas, texts = as_lists(ids, metadatas, texts)
            
            for id, metadata, text in zip(ids, metadatas or [None]*len(ids), texts or [None]*len(ids)):
                # Verify document exists
                cursor.execute(f"SELECT 1 FROM {self.table_name} WHERE id = ?", (id,))
                if not cursor.fetchone():
                    raise DocumentNotFoundError(f"Document with ID {id} not found")
                
                # Prepare update query dynamically based on provided fields
                update_parts = []
                update_params = []
                if text is not None:
                    update_parts.append("text = ?")
                    update_params.append(text)
                if metadata is not None:
                    update_parts.append("metadata = ?")
                    update_params.append(json_dumps(metadata))
                
                if update_parts:
                    update_params.append(id)
                    cursor.execute(
                        f"UPDATE {self.table_name} SET {', '.join(update_parts)} WHERE id = ?", 
                        update_params
                    )
            
            conn.commit()
        return self

    def _update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        """Update existing documents in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for document in documents:
                # Verify document exists
                cursor.execute(f"SELECT 1 FROM {self.table_name} WHERE id = ?", (document.id,))
                if not cursor.fetchone():
                    raise DocumentNotFoundError(f"Document with ID {document.id} not found")
                
                # Update document
                cursor.execute(
                    f"UPDATE {self.table_name} SET text = ?, metadata = ? WHERE id = ?", 
                    (document.text, json_dumps(document.metadata), document.id)
                )
            
            conn.commit()
        return self

    def _upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        """Upsert documents in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            ids, metadatas, texts = as_lists(ids, metadatas, texts)
            
            for id, metadata, text in zip(ids, metadatas or [None]*len(ids), texts or [None]*len(ids)):
                # Upsert using INSERT OR REPLACE
                cursor.execute(
                    f"INSERT OR REPLACE INTO {self.table_name} (id, text, metadata) VALUES (?, ?, ?)", 
                    (id, text, json_dumps(metadata) if metadata else None)
                )
            
            conn.commit()
        return self

    def _upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        """Upsert documents in the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for document in documents:
                # Upsert using INSERT OR REPLACE
                cursor.execute(
                    f"INSERT OR REPLACE INTO {self.table_name} (id, text, metadata) VALUES (?, ?, ?)", 
                    (document.id, document.text, json_dumps(document.metadata))
                )
            
            conn.commit()
        return self

    def _delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> Self:
        """Delete documents from the collection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # If specific IDs are provided, delete those
            if ids:
                placeholders = ','.join(['?'] * len(ids))
                cursor.execute(f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})", ids)
            else:
                # Build filter query for flexible deletion
                where_clause, where_params = self._build_filter_query(where, where_document)
                cursor.execute(f"DELETE FROM {self.table_name}{where_clause}", where_params)
            
            conn.commit()
        return self

    def _delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
                ) -> Self:
        """Delete specific documents or those matching filters."""
        if not documents:
            return self._delete(where=where, where_document=where_document)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for document in documents:
                # Check optional metadata and text filters
                if where or where_document:
                    # Construct a query to check if the document matches filters
                    filter_clause, filter_params = self._build_filter_query(where, where_document)
                    check_query = f"SELECT 1 FROM {self.table_name} WHERE id = ?{filter_clause}"
                    filter_params.insert(0, document.id)
                    
                    cursor.execute(check_query, filter_params)
                    if not cursor.fetchone():
                        continue
                
                # Delete matching document
                cursor.execute(f"DELETE FROM {self.table_name} WHERE id = ?", (document.id,))
            
            conn.commit()
        return self
    
    def _get(
        self,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
        limit: Optional[int] = None,
        offset: Optional[int] = None,            
        **kwargs
        ) -> GetResult:
        """Retrieve documents from the collection based on optional filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Base query
            query = f"SELECT id, text, metadata FROM {self.table_name}"
            
            # Build filter query
            where_clause, where_params = self._build_filter_query(where, where_document)
            query += where_clause
            
            # Add optional id filtering if provided
            if ids:
                placeholders = ','.join(['?'] * len(ids))
                if where_clause:
                    query += f" AND id IN ({placeholders})"
                else:
                    query += f" WHERE id IN ({placeholders})"
                where_params.extend(ids)
            
            # Add pagination
            if limit is not None:
                query += f" LIMIT {limit}"
            if offset is not None:
                query += f" OFFSET {offset}"
            
            # Execute query
            cursor.execute(query, where_params)
            results = cursor.fetchall()
            
            # Process results
            result_ids = []
            texts = [] if "texts" in include else None
            metadatas = [] if "metadatas" in include else None
            
            for row in results:
                result_ids.append(row[0])
                if texts is not None:
                    texts.append(row[1])
                if metadatas is not None:
                    metadatas.append(json_loads(row[2]) if row[2] else None)
            
            return GetResult(
                ids=result_ids, 
                texts=texts, 
                metadatas=metadatas, 
                included=["ids", *include]
            )    
        

class SQLiteDocumentDB(DocumentDB[SQLiteDocumentDBCollection, WrappedConnectionContextManager]):
    provider = "sqlite"
    collection_class = SQLiteDocumentDBCollection
    
    def _setup(self):
        super()._setup()
        self.db_path = self.init_kwargs.pop("database", None) or self.init_kwargs.pop("db_path", ":memory:")            
        self.connection_kwargs = self.init_kwargs.pop("connection_kwargs", {})
        if self.init_kwargs.pop("connect_on_init", True):
            self.connect()

    def connect(self, **connection_kwargs) -> Connection:
        try:
            self._connection = sqlite_connect(self.db_path, **combine_dicts(self.connection_kwargs, connection_kwargs))
            return self._connection
        except SQLiteError as e:
            raise DBAPIError(f"Error connecting to SQLite database at '{self.db_path}'", original_exception=e)

    @property
    def connection(self) -> Connection:
        if not hasattr(self, "_connection") or not self._connection:
            return self.connect()
        return self._connection

    def close(self):
        try:
            self.connection.close()
            self._connection = None
        except SQLiteError as e:
            raise DBAPIError(f"Error closing connection to SQLite database at '{self.db_path}'", original_exception=e)

    def __del__(self):
        self.close()
    
    def get_connection(self) -> ConnectionContextManager:
        return ConnectionContextManager(self)
    
    def _create_wrapped_collection(
            self,
            config: DocumentDBCollectionConfig,
            **init_kwargs
    ) -> WrappedConnectionContextManager:
        """Create a new collection based on the provided configuration."""
        # Check if collection already exists
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (config.name,))
            if cursor.fetchone():
                raise CollectionAlreadyExistsError(f"Collection {config.name} already exists")
        
        config.init_kwargs['create_table_on_init'] = True        
        return self.get_connection

    def _get_wrapped_collection(
            self,
            config: DocumentDBCollectionConfig,
            **init_kwargs
    ) -> WrappedConnectionContextManager:
        """Retrieve an existing collection based on the provided configuration."""
                
        # Check if collection exists
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (config.name,))
            if not cursor.fetchone():
                raise CollectionNotFoundError(f"Collection {config.name} not found")

        config.init_kwargs['create_table_on_init'] = False
        return self.get_connection

    def _list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            **kwargs
    ) -> list[str]:
        """List all collections in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Apply pagination
            if offset is not None or limit is not None:
                tables = tables[limit_offset_slice(limit, offset)]
            
            return tables

    def _count_collections(self, **kwargs) -> int:
        """Count the number of collections (tables) in the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(name) FROM sqlite_master WHERE type='table'")
            return cursor.fetchone()[0]

    def _delete_collection(self, name: CollectionName, **kwargs) -> None:
        """Delete a specific collection (table) from the database."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {name}")
                conn.commit()
            except SQLiteError as e:
                raise CollectionNotFoundError(f"Error deleting collection {name}: {e}")