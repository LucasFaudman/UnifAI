from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Collection, Generator, Self, TypeAlias
from itertools import zip_longest


from ...exceptions.db_errors import CollectionAlreadyExistsError, CollectionNotFoundError, DocumentAlreadyExistsError, DocumentNotFoundError
from ...types import Document, Documents, GetResult
from ...types.annotations import CollectionName
from ...utils import check_filter, check_metadata_filters, limit_offset_slice, as_lists, as_list
from .._base_components._base_document_db import DocumentDB, DocumentDBCollection
from ...configs.document_db_config import DocumentDBConfig, DocumentDBCollectionConfig

T = TypeVar("T")
DataDict = dict[str, dict[Literal["text", "metadata"], Any]]

class EphemeralDocumentDBCollection(DocumentDBCollection[DataDict]):
    provider = "ephemeral"

    def _count(self, **kwargs) -> int:
        return len(self.wrapped)
    
    def _list_ids(self, **kwargs) -> list[str]:
        return list(self.wrapped.keys())
    
    def _check_id_already_exists(self, id: str) -> None:
        if id in self.wrapped:
            raise DocumentAlreadyExistsError(f"Document with ID {id} already exists in collection {self.name}")
    
    def _try_get_document_data(self, id: str) -> dict[Literal["text", "metadata"], Any]:
        if (data := self.wrapped.get(id)) is None:
            raise DocumentNotFoundError(f"Document with ID {id} not found in collection {self.name}")
        return data

    def _add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
            ) -> Self:
        ids, metadatas, texts = as_lists(ids, metadatas, texts)
        for id, metadata, text in zip_longest(ids, metadatas, texts):
            self._check_id_already_exists(id)
            self.wrapped[id] = {"text": text, "metadata": metadata}
        return self
                
    def _add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:
        for document in documents:
            self._check_id_already_exists(document.id)
            self.wrapped[document.id] = {"text": document.text, "metadata": document.metadata}
        return self

    def _update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:

        _update_metadata = metadatas is not None
        _update_text = texts is not None
        ids, metadatas, texts = as_lists(ids, metadatas, texts)        
        for id, metadata, text in zip_longest(ids, metadatas, texts):
            data = self._try_get_document_data(id)            
            if _update_metadata:
                data["metadata"] = metadata
            if _update_text:
                data["text"] = text
        return self
    
    def _update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        for document in documents:
            data = self._try_get_document_data(document.id)
            data["text"] = document.text
            data["metadata"] = document.metadata
        return self
                 
    def _upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:        
        ids, metadatas, texts = as_lists(ids, metadatas, texts)        
        for id, metadata, text in zip_longest(ids, metadatas, texts):
            self.wrapped[id] = {"text": text, "metadata": metadata}
        return self
    
    def _upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        for document in documents:
            self.wrapped[document.id] = {"text": document.text, "metadata": document.metadata}
        return self

    def _delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> Self:
        if ids and not where and not where_document:
            ids_to_delete = ids if isinstance(ids, list) else [ids]
        else:
            ids_to_delete = self.get(ids, where, where_document).ids
        for id in ids_to_delete:
            self.wrapped.pop(id, None)
        return self

    def _delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
                ) -> Self:    
        if not documents:
            return self._delete(where=where, where_document=where_document)
        
        for document in documents:
            if not self.check_filters(where, document.metadata, where_document, document.text):
                continue
            self.wrapped.pop(document.id, None)
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
        
        result_ids = []
        metadatas = [] if "metadatas" in include else None
        texts = [] if "texts" in include else None
        added = 0
        for id in (as_list(ids) or self.list_ids()[offset:]):
            data = self._try_get_document_data(id)
            metadata = data["metadata"]
            text = data["text"]            
            
            if not self.check_filters(where, metadata, where_document, text):
                continue

            result_ids.append(id)
            if metadatas is not None:
                metadatas.append(metadata)
            if texts is not None:
                texts.append(text)
            added += 1
            if limit is not None and added >= limit:
                break
        
        return GetResult(ids=result_ids, metadatas=metadatas, texts=texts, included=["ids", *include])
      
    
class EphemeralDocumentDB(DocumentDB[EphemeralDocumentDBCollection, DataDict]):
    provider = "ephemeral"
    collection_class: Type[EphemeralDocumentDBCollection] = EphemeralDocumentDBCollection 

    def _setup(self) -> None:
        super()._setup()
        self._data = self.init_kwargs.get("data", {})
    
    def _list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
            **kwargs
    ) -> list[str]:
        return list(self.collections.keys())[limit_offset_slice(limit, offset)]
    
    def _count_collections(self, **kwargs) -> int:
        return len(self.collections)

    def _delete_collection(self, name: CollectionName, **kwargs) -> None:
        try:
            del self.collections[name]
            del self._data[name]
        except KeyError:
            raise CollectionNotFoundError(f"Collection with name {name} not found in database {self.name}")

    def _create_collection_from_config(
            self,
            config: DocumentDBCollectionConfig,
            **collection_kwargs
    ) -> EphemeralDocumentDBCollection:
        collection_name = config.name
        if collection_name in self.collections:
            raise CollectionAlreadyExistsError(f"Collection with name {collection_name} already exists in database {self.name}")
        wrapped = {}
        self._data[collection_name] = wrapped
        return self.init_collection_from_wrapped(config, wrapped, **collection_kwargs)

    def _get_collection_from_config(
            self,
            config: DocumentDBCollectionConfig,
            **collection_kwargs
    ) -> EphemeralDocumentDBCollection:
        collection_name = config.name
        if (wrapped := self._data.get(collection_name)) is None:
            raise CollectionNotFoundError(f"Collection with name {collection_name} not found in database {self.name}")
        return self.init_collection_from_wrapped(config, wrapped, **collection_kwargs)

        
        