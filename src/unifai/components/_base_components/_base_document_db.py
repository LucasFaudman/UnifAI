from typing import TypeVar, ClassVar, Generic, Type
from ...configs.document_db_config import DocumentDBConfig, DocumentDBCollectionConfig
from ._base_db import BaseDB, BaseDBCollection

WrappedT = TypeVar("WrappedT")

class DocumentDBCollection(BaseDBCollection[DocumentDBCollectionConfig, WrappedT], Generic[WrappedT]):
    component_type = "document_db_collection"
    provider = "base"    
    config_class = DocumentDBCollectionConfig

CollectionT = TypeVar("CollectionT", bound=DocumentDBCollection)

class DocumentDB(BaseDB[DocumentDBConfig, DocumentDBCollectionConfig, CollectionT, WrappedT], Generic[CollectionT, WrappedT]):
    component_type = "document_db"
    provider = "base"    
    config_class = DocumentDBConfig

    collection_class: Type[CollectionT]
    