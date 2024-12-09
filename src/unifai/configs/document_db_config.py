from typing import Any, Callable, Collection, Literal, Optional, ClassVar

from ._base_configs import BaseDBCollectionConfig, BaseDBConfig

class DocumentDBCollectionConfig(BaseDBCollectionConfig):
    component_type: ClassVar = "document_db_collection"
    extra_kwargs: Optional[dict[Literal[
        "add",
        "update",
        "upsert",
        "delete",
        "get",      
        ], dict[str, Any]]] = None    

class DocumentDBConfig(BaseDBConfig):
    component_type: ClassVar = "document_db"
    default_collection: DocumentDBCollectionConfig = DocumentDBCollectionConfig(provider="default")
    extra_kwargs: Optional[dict[Literal[
        "create_collection", 
        "get_collection",
        "list_collections",
        "count_collections",
        "delete_collection",
        "add",
        "update",
        "upsert",
        "delete",
        "get",      
        ], dict[str, Any]]] = None    


    
