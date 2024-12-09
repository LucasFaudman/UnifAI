from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self, Generic, TypeAlias

from ._base_component import UnifAIComponent
from ._base_adapter import UnifAIAdapter

from ...types.annotations import CollectionName
from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, EmbeddingTaskTypeInput, Usage, GetResult, QueryResult, Document, Documents
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError, NotFoundError, CollectionNotFoundError

from ...type_conversions import documents_to_lists, iterables_to_documents
from ...utils import _next, combine_dicts, as_lists
from ...configs._base_configs import BaseDBConfig, BaseDBCollectionConfig

T = TypeVar("T")
DBConfigT = TypeVar("DBConfigT", bound=BaseDBConfig)
CollectionConfigT = TypeVar("CollectionConfigT", bound=BaseDBCollectionConfig)
WrappedT = TypeVar("WrappedT")

class BaseDBCollection(UnifAIComponent[CollectionConfigT], Generic[CollectionConfigT, WrappedT]):
    component_type = "base_db_collection"
    provider = "base"
    config_class: Type[CollectionConfigT]
    
    wrapped_type: Type[WrappedT]
    
    _document_attrs = ("id", "metadata", "text")

    def __init__(self, config: CollectionConfigT, wrapped: WrappedT):        
        super().__init__(config)
        self.wrapped: WrappedT = wrapped
        self.response_infos = []

    def count(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def list_ids(self, **kwargs) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    # def delete_all(self, **kwargs) -> None:
    #     raise NotImplementedError("This method must be implemented by the subclass")    
    
    def add(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
            ) -> Self:
        return self.add_documents(iterables_to_documents(*as_lists(ids, metadatas, texts)), **kwargs)
        
    def add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:            
        return self.add(*documents_to_lists(documents, self._document_attrs), **kwargs)

    def update(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
                ) -> Self:
        if self.update_documents is BaseDBCollection.update_documents:
            raise NotImplementedError("This method must be implemented by the subclass")
        return self.update_documents(iterables_to_documents(*as_lists(ids, metadatas, texts)), **kwargs)
    
    def update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        return self.update(*documents_to_lists(documents, self._document_attrs), **kwargs)  
                 
    def upsert(
            self,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
                ) -> Self:
        
        return self.upsert_documents(iterables_to_documents(*as_lists(ids, metadatas, texts)), **kwargs)
    
    def upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:                
        return self.upsert(*documents_to_lists(documents, self._document_attrs), **kwargs)
  
    def delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> Self:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def delete_all(self, **kwargs) -> Self:
        kwargs["ids"] = kwargs.pop("ids", None) or self.list_ids()
        return self.delete(**kwargs)

    def delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
                ) -> Self:    
        ids = [doc.id for doc in documents] if documents else None
        return self.delete(ids, where, where_document, **kwargs)
    
    def get(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
            ) -> GetResult:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def get_all(
            self,
            **kwargs
    ) -> GetResult:
        kwargs["ids"] = kwargs.pop("ids", None) or self.list_ids()
        return self.get(**kwargs)
    
    def get_documents(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> list[Document]:
        return self.get(ids, where, where_document, include, limit, offset, **kwargs).to_documents()

    def get_document(
            self,
            id: str,
            **kwargs
    ) -> Document:
        return self.get_documents([id], **kwargs)[0]
    
    def get_all_documents(
            self,
            **kwargs
    ) -> list[Document]:
        return self.get_all(**kwargs).to_documents()

CollectionT = TypeVar("CollectionT", bound=BaseDBCollection)

class BaseDB(UnifAIAdapter[DBConfigT], Generic[DBConfigT, CollectionConfigT, CollectionT, WrappedT]):
    component_type = "base_db"
    provider = "base"    
    config_class: Type[DBConfigT]    

    collection_class: Type[CollectionT]
    collection_config_class: Type[CollectionConfigT]

    def _setup(self) -> None:
        super()._setup()
        self.collections = {}
    
    def _create_wrapped_collection(self, config: CollectionConfigT, **collection_kwargs) -> WrappedT:
        raise NotImplementedError("This method must be implemented by the subclass")

    def _get_wrapped_collection(self, config: CollectionConfigT, **collection_kwargs) -> WrappedT:
        raise NotImplementedError("This method must be implemented by the subclass")

    def _kwargs_to_collection_config(self, **collection_kwargs)-> CollectionConfigT:
        config = self.config.default_collection.model_copy(deep=True)
        config.provider = self.provider
        
        config_fields = config.model_fields
        for key in list(collection_kwargs.keys()):
            if key in config_fields:
                value = collection_kwargs.pop(key)
                # if (value is not None) ^ (getattr(config, key) is not None):
                if value is not None or (value is None and key in config.model_fields_set):
                    setattr(config, key, value)
        config.init_kwargs.update(collection_kwargs)
        return config
            
    def _init_collection(
            self, 
            config: CollectionConfigT, 
            wrapped: WrappedT,
            **kwargs
    ) -> CollectionT:
        collection = self.collection_class(config, wrapped, **kwargs)
        self.collections[config.name] = collection
        return collection
    
    def create_collection_from_config(
            self,
            config: CollectionConfigT,
            **kwargs
    ) -> CollectionT:  
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def get_collection_from_config(
            self,
            config: CollectionConfigT,
            **kwargs
    ) -> CollectionT:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def get_or_create_collection_from_config(
            self,
            config: CollectionConfigT,
            **kwargs
    ) -> CollectionT:        
        try:
            return self.get_collection_from_config(config)
        except (CollectionNotFoundError, NotFoundError, BadRequestError):
            return self.create_collection_from_config(config)
        
    def create_collection(
            self,
            name: CollectionName = "default_collection",
            **collection_kwargs
    ) -> CollectionT:
        return self.create_collection_from_config(self._kwargs_to_collection_config(name=name, **collection_kwargs))
    
    def get_collection(
            self,
            name: CollectionName = "default_collection",
            **collection_kwargs
    ) -> CollectionT:
        return self.get_collection_from_config(self._kwargs_to_collection_config(name=name, **collection_kwargs))
    
    def get_or_create_collection(
            self,
            name: CollectionName = "default_collection",
            **collection_kwargs
    ) -> CollectionT:
        return self.get_or_create_collection_from_config(self._kwargs_to_collection_config(name=name, **collection_kwargs))
            
    def list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
            **kwargs
    ) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def count_collections(self, **kwargs) -> int:
        raise NotImplementedError("This method must be implemented by the subclass")

    def delete_collection(self, name: CollectionName, **kwargs) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")
    
    def delete_collections(self, names: Iterable[CollectionName], **kwargs) -> None:
        for name in names:
            self.delete_collection(name, **kwargs)

    def delete_all_collections(self, **kwargs) -> None:
        self.delete_collections(self.list_collections(), **kwargs)

    def pop_collection(self, name: CollectionName, default: T=None) -> CollectionT|T:
        return self.collections.pop(name)

    def _resolve_collection(self, collection: CollectionName | CollectionT) -> CollectionT:
        if isinstance(collection, str):
            return self.get_or_create_collection(collection)
        return collection

    def count(
            self, 
            collection: CollectionName | CollectionT, 
            **kwargs
    ) -> int:
        return self._resolve_collection(collection).count(**kwargs)
    
    def add(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).add(*as_lists(ids, metadatas, texts), **kwargs)
    
    def delete(
            self, 
            collection: CollectionName | CollectionT,
            ids: list[str],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).delete(ids, where, where_document, **kwargs)

    def update(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).update(*as_lists(ids, metadatas, texts), **kwargs)
    
    def upsert(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str]|str,
            metadatas: Optional[list[dict]|dict] = None,
            texts: Optional[list[str]|str] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).upsert(*as_lists(ids, metadatas, texts), **kwargs)
    
    def get(
            self,
            collection: CollectionName | CollectionT,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
            ) -> GetResult:
        return self._resolve_collection(collection).get(ids, where, where_document, include, limit, offset, **kwargs)
    

    def add_documents(
            self, 
            collection: CollectionName | CollectionT,
            documents: Iterable[Document] | Documents,
    ) -> CollectionT:
        return self._resolve_collection(collection).add_documents(documents)
    
    def delete_documents(
            self, 
            collection: CollectionName | CollectionT,
            documents: Iterable[Document] | Documents,
    ) -> CollectionT:
        return self._resolve_collection(collection).delete_documents(documents)
    
    def update_documents(
            self, 
            collection: CollectionName | CollectionT,
            documents: Iterable[Document] | Documents,
    ) -> CollectionT:
        return self._resolve_collection(collection).update_documents(documents)
    
    def upsert_documents(
            self, 
            collection: CollectionName | CollectionT,
            documents: Iterable[Document] | Documents,
    ) -> CollectionT:
        return self._resolve_collection(collection).upsert_documents(documents)
    
    def get_documents(
            self, 
            collection: CollectionName | CollectionT,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            limit: Optional[int] = None,
            offset: Optional[int] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            **kwargs
    ) -> list[Document]:
        return self._resolve_collection(collection).get_documents(ids, where, where_document, include, limit, offset, **kwargs)
    
    def get_document(
            self, 
            collection: CollectionName | CollectionT,
            id: str,
            **kwargs
    ) -> Document:
        return self._resolve_collection(collection).get_document(id, **kwargs)