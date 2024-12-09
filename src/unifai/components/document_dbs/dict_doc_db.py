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

class DictDocumentDBCollection(DocumentDBCollection[DataDict]):
    provider = "dict_document_db"

    def count(self, **kwargs) -> int:
        return len(self.wrapped)
    
    def list_ids(self, **kwargs) -> list[str]:
        return list(self.wrapped.keys())
    
    def _check_id_already_exists(self, id: str) -> None:
        if id in self.wrapped:
            raise DocumentAlreadyExistsError(f"Document with ID {id} already exists in collection {self.name}")
    
    def _try_get_document_data(self, id: str) -> dict[Literal["text", "metadata"], Any]:
        if (data := self.wrapped.get(id)) is None:
            raise DocumentNotFoundError(f"Document with ID {id} not found in collection {self.name}")
        return data

    def add(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
            ) -> Self:
        ids, metadatas, texts = as_lists(ids, metadatas, texts)
        for id, metadata, text in zip_longest(ids, metadatas, texts):
            self._check_id_already_exists(id)
            self.wrapped[id] = {"text": text, "metadata": metadata}
        return self
                
    def add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:
        for document in documents:
            self._check_id_already_exists(document.id)
            self.wrapped[document.id] = {"text": document.text, "metadata": document.metadata}
        return self

    def update(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
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
    
    def update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        for document in documents:
            data = self._try_get_document_data(document.id)
            data["text"] = document.text
            data["metadata"] = document.metadata
        return self
                 
    def upsert(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
                ) -> Self:        
        ids, metadatas, texts = as_lists(ids, metadatas, texts)        
        for id, metadata, text in zip_longest(ids, metadatas, texts):
            self.wrapped[id] = {"text": text, "metadata": metadata}
        return self
    
    def upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        for document in documents:
            self.wrapped[document.id] = {"text": document.text, "metadata": document.metadata}
        return self
        
    def get(
            self,
            ids: Optional[list[str]|str] = None,
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
            
            if where and not check_metadata_filters(where, metadata):
                continue
            if where_document and not check_filter(where_document, text):
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
    
    def get_documents(
            self,
            ids: Optional[list[str]|str] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> list[Document]:
        return self.get(ids, where, where_document, include, limit, offset, **kwargs).to_documents()

    def delete_all(self, **kwargs) -> None:
        for id in self.list_ids():
            del self.wrapped[id]
  
    def delete(
            self, 
            ids: Optional[list[str]|str] = None,
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

    def delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
                ) -> Self:    
        if not documents:
            return self.delete(where=where, where_document=where_document)
        for document in documents:
            if where and document.metadata and not check_metadata_filters(where, document.metadata):
                continue
            if where_document and document.text and not check_filter(where_document, document.text):
                continue
            self.wrapped.pop(document.id, None)
        return self
    
class DictDocumentDB(DocumentDB[DictDocumentDBCollection, DataDict]):
    provider = "dict_document_db"
    collection_class: Type[DictDocumentDBCollection] = DictDocumentDBCollection 

    def _setup(self) -> None:
        super()._setup()
        self._data = self.init_kwargs.get("data", {})    

    def _create_wrapped_collection(self, config: DocumentDBCollectionConfig, **collection_kwargs) -> DataDict:
        collection_name = config.name
        if collection_name in self.collections:
            raise CollectionAlreadyExistsError(f"Collection with name {collection_name} already exists in database {self.name}")
        wrapped = {}
        self._data[collection_name] = wrapped
        return wrapped

    def _get_wrapped_collection(self, config: DocumentDBCollectionConfig, **collection_kwargs) -> DataDict:
        collection_name = config.name
        if (wrapped := self._data.get(collection_name)) is None:
            raise CollectionNotFoundError(f"Collection with name {collection_name} not found in database {self.name}")
        return wrapped
    
    def create_collection_from_config(
            self,
            config: DocumentDBCollectionConfig,
            **collection_kwargs
    ) -> DictDocumentDBCollection:
        wrapped = self._create_wrapped_collection(config, **collection_kwargs)
        return self._init_collection(config, wrapped, **collection_kwargs)

    def get_collection_from_config(
            self,
            config: DocumentDBCollectionConfig,
            **collection_kwargs
    ) -> DictDocumentDBCollection:
        wrapped = self._get_wrapped_collection(config, **collection_kwargs)
        return self._init_collection(config, wrapped, **collection_kwargs)
    
    def list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
            **kwargs
    ) -> list[str]:
        return list(self.collections.keys())[limit_offset_slice(limit, offset)]
    
    def count_collections(self, **kwargs) -> int:
        return len(self.collections)

    def delete_collection(self, name: CollectionName, **kwargs) -> None:
        try:
            del self.collections[name]
            del self._data[name]
        except KeyError:
            raise CollectionNotFoundError(f"Collection with name {name} not found in database {self.name}")
        
        