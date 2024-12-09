from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar
from ._base_configs import ComponentConfig, BaseDocumentCleanerConfig, BaseModel, Field

class _BaseDocumentLoaderConfig(ComponentConfig):
    pass
    # encoding: str = "utf-8"
    # add_to_metadata: Optional[list[Literal["source", "mimetype"]]] = ["source"]
    # metadata_load_func: Literal["json", "yaml"]|Callable[[IO], dict] = "json"
    # source_id_func: Literal["stringify_source", "hash_source"]|Callable[[Any], str] = "stringify_source"
    # mimetype_func: Literal["builtin_mimetypes", "magic"]|Callable[[Any], str | None] = "builtin_mimetypes"    

class DocumentLoaderConfig(BaseDocumentCleanerConfig):
    component_type: ClassVar = "document_loader"
    encoding: str = "utf-8"
    add_to_metadata: Optional[list[Literal["source", "mimetype"]]] = ["source"]
    
    metadata_load_func: Literal["json", "yaml"]|Callable[[IO], dict] = "json"
    source_id_func: Literal["stringify_source", "hash_source"]|Callable[[Any], str] = "stringify_source"
    mimetype_func: Literal["builtin_mimetypes", "magic"]|Callable[[Any], str | None] = "builtin_mimetypes"       
    error_handling: dict[Literal["source_load_error", "metadata_load_error", "processor_error"], Literal["skip", "raise"]] = {
        "source_load_error": "raise",
        "metadata_load_error": "raise",
        "processor_error": "raise"
    }
    error_retries: dict[Literal["source_load_error", "metadata_load_error", "processor_error"], int] = {
        "source_load_error": 0,
        "metadata_load_error": 0,
        "processor_error": 0
    }
    extra_kwargs: Optional[dict[Literal["load_documents", "clean_text"], dict[str, Any]]] = None

DocumentLoaderConfig(provider="default")
