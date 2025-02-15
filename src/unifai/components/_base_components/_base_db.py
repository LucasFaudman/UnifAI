from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self, Generic, TypeAlias
from abc import abstractmethod

from .__base_component import UnifAIComponent
from ._base_adapter import UnifAIAdapter

from ...types.annotations import CollectionName
from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, EmbeddingTaskTypeInput, Usage, GetResult, QueryResult, Document, Documents
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError, BadRequestError, NotFoundError, CollectionNotFoundError

from ...type_conversions import documents_to_lists, iterables_to_documents
from ...utils import _next, combine_dicts, as_lists, check_filter, check_metadata_filters, limit_offset_slice
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
    _is_abstract = True    
    _abstract_methods = ("_add", "_update", "_upsert", "_delete", "_get")
    _abstract_method_suffixes = ("_documents",)

    def _setup(self) -> None:
        self.wrapped: WrappedT = self.init_kwargs.pop("wrapped", None)
        if self.wrapped is None:
            raise ValueError(f"No wrapped {self.wrapped_type.__name__} provided")
        self.response_infos = []
    
    @staticmethod
    def check_filters(
            where: Optional[dict] = None,
            metadata: Optional[dict] = None,
            where_document: Optional[dict] = None,            
            text: Optional[str] = None,
            ) -> bool:
        if where and metadata and not check_metadata_filters(where, metadata):
            return False
        if where_document and text and not check_filter(where_document, text):
            return False
        return True

    ## Abstract methods
    @abstractmethod
    def _count(self, **kwargs) -> int:
        pass
    
    @abstractmethod
    def _list_ids(self, **kwargs) -> list[str]:
        pass
    
    def _add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
            ) -> Self:
        return self._add_documents(iterables_to_documents(ids, metadatas, texts), **kwargs)
        
    def _add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:            
        return self._add(*documents_to_lists(documents, self._document_attrs), **kwargs)

    def _update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        return self._update_documents(iterables_to_documents(ids, metadatas, texts), **kwargs)
    
    def _update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        return self._update(*documents_to_lists(documents, self._document_attrs), **kwargs)  
                 
    def _upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        
        return self._upsert_documents(iterables_to_documents(ids, metadatas, texts), **kwargs)
    
    def _upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:                
        return self._upsert(*documents_to_lists(documents, self._document_attrs), **kwargs)
  
    def _delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> Self:
        return self._delete_documents(iterables_to_documents(ids, attrs=("id",)), where, where_document, **kwargs)
    
    def _delete_documents(
            self,
            documents: Optional[Iterable[Document] | Documents],
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
                ) -> Self:    
        ids = [doc.id for doc in documents] if documents else None
        return self._delete(ids, where, where_document, **kwargs)
    
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
        return GetResult.from_documents(self._get_documents(ids, where, where_document, include, limit, offset, **kwargs))
        
    def _get_documents(
            self,
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            include: list[Literal["metadatas", "texts"]] = ["metadatas", "texts"],
            limit: Optional[int] = None,
            offset: Optional[int] = None,            
            **kwargs
    ) -> Documents:
        return self.get(ids, where, where_document, include, limit, offset, **kwargs).to_documents()


    ## Concrete methods
    def count(self, **kwargs) -> int:
        return self._run_func(self._count, **kwargs)
    
    def list_ids(self, **kwargs) -> list[str]:
        return self._run_func(self._list_ids, **kwargs)
    
    def add(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
            ) -> Self:
        return self._run_func(self._add, ids, metadatas, texts, **kwargs)
        
    def add_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
            ) -> Self:            
        return self._run_func(self._add_documents, documents, **kwargs)

    def update(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        return self._run_func(self._update, ids, metadatas, texts, **kwargs)
    
    def update_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:
        return self._run_func(self._update_documents, documents, **kwargs)
                 
    def upsert(
            self,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
                ) -> Self:
        
        return self._run_func(self._upsert, ids, metadatas, texts, **kwargs)
    
    def upsert_documents(
            self,
            documents: Iterable[Document] | Documents,
            **kwargs
                ) -> Self:                
        return self._run_func(self._upsert_documents, documents, **kwargs)
  
    def delete(
            self, 
            ids: Optional[list[str]] = None,
            where: Optional[dict] = None,
            where_document: Optional[dict] = None,
            **kwargs
               ) -> Self:
        return self._run_func(self._delete, ids, where, where_document, **kwargs)
    
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
        return self._run_func(self._delete_documents, documents, where, where_document, **kwargs)
    
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
        return self._run_func(self._get, ids, where, where_document, include, limit, offset, **kwargs)
    
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
    ) -> Documents:
        return self._run_func(self._get_documents, ids, where, where_document, include, limit, offset, **kwargs)

    def get_document(
            self,
            id: str,
            **kwargs
    ) -> Document:
        return self.get_documents([id], **kwargs)[0]
    
    def get_all_documents(
            self,
            **kwargs
    ) -> Documents:
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

    @abstractmethod
    def _create_wrapped_collection(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> WrappedT:  
        ...
    
    @abstractmethod
    def _get_wrapped_collection(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> WrappedT:
        ...        

    @abstractmethod
    def _list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
            **kwargs
    ) -> list[str]:
        ...
    
    @abstractmethod
    def _count_collections(self, **kwargs) -> int:
        ...

    @abstractmethod
    def _delete_collection(self, name: CollectionName, **kwargs) -> None:
        ...        

    
    # Concrete Methods
    def _init_collection_from_wrapped(
            self, 
            config: CollectionConfigT, 
            wrapped: WrappedT,
            **init_kwargs
    ) -> CollectionT:
        collection = self.collection_class(config, wrapped=wrapped, **init_kwargs)
        self.collections[config.name] = collection
        return collection

    def _init_collection_config_components(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> CollectionConfigT:
        return config

    def create_collection_from_config(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> CollectionT:
        config = self._run_func(self._init_collection_config_components, config, **init_kwargs)
        wrapped = self._run_func(self._create_wrapped_collection, config, **init_kwargs)
        return self._run_func(self._init_collection_from_wrapped, config, wrapped, **init_kwargs)
    
    def get_collection_from_config(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> CollectionT:
        config = self._run_func(self._init_collection_config_components, config, **init_kwargs)
        wrapped = self._run_func(self._get_wrapped_collection, config, **init_kwargs)
        return self._run_func(self._init_collection_from_wrapped, config, wrapped, **init_kwargs)
    
    def get_or_create_collection_from_config(
            self,
            config: CollectionConfigT,
            **init_kwargs
    ) -> CollectionT:        
        try:
            return self.get_collection_from_config(config, **init_kwargs)
        except (CollectionNotFoundError, NotFoundError, BadRequestError):
            return self.create_collection_from_config(config, **init_kwargs)

    def _kwargs_to_collection_config(self, **init_kwargs)-> CollectionConfigT:
        config = self.config.default_collection.model_copy(deep=True)
        config.provider = self.provider
        
        config_fields = config.model_fields
        for key in list(init_kwargs.keys()):
            if key in config_fields:
                value = init_kwargs.pop(key)
                if value is not None or (value is None and key in config.model_fields_set):
                    setattr(config, key, value)
        config.init_kwargs.update(init_kwargs)
        return config

    def create_collection(
            self,
            name: CollectionName = "default_collection",
            **init_kwargs
    ) -> CollectionT:
        return self.create_collection_from_config(self._kwargs_to_collection_config(name=name, **init_kwargs))
    
    def get_collection(
            self,
            name: CollectionName = "default_collection",
            **init_kwargs
    ) -> CollectionT:
        return self.get_collection_from_config(self._kwargs_to_collection_config(name=name, **init_kwargs))
    
    def get_or_create_collection(
            self,
            name: CollectionName = "default_collection",
            **init_kwargs
    ) -> CollectionT:
        return self.get_or_create_collection_from_config(self._kwargs_to_collection_config(name=name, **init_kwargs))
            
    def list_collections(
            self,
            limit: Optional[int] = None,
            offset: Optional[int] = None, # woop woop,
            **kwargs
    ) -> list[str]:
        return self._run_func(self._list_collections, limit, offset, **kwargs)
    
    def count_collections(self, **kwargs) -> int:
        return self._run_func(self._count_collections, **kwargs)

    def delete_collection(self, name: CollectionName, **kwargs) -> None:
        return self._run_func(self._delete_collection, name, **kwargs)
    
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

    # Collection methods
    def count(
            self, 
            collection: CollectionName | CollectionT, 
            **kwargs
    ) -> int:
        return self._resolve_collection(collection).count(**kwargs)
    
    def add(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).add(ids, metadatas, texts, **kwargs)
    
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
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).update(ids, metadatas, texts, **kwargs)
    
    def upsert(
            self,
            collection: CollectionName | CollectionT,
            ids: list[str],
            metadatas: Optional[list[dict]] = None,
            texts: Optional[list[str]] = None,
            **kwargs
    ) -> CollectionT:
        return self._resolve_collection(collection).upsert(ids, metadatas, texts, **kwargs)
    
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
    ) -> Documents:
        return self._resolve_collection(collection).get_documents(ids, where, where_document, include, limit, offset, **kwargs)
    
    def get_document(
            self, 
            collection: CollectionName | CollectionT,
            id: str,
            **kwargs
    ) -> Document:
        return self._resolve_collection(collection).get_document(id, **kwargs)