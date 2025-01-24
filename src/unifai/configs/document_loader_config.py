from typing import Any, Callable, ParamSpec, Generic, TypeVar, Iterable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

# from ..types.annotations import InputP
from ..types.documents import Document, Documents
from ..types.response_info import ResponseInfo
from ._base_configs import ComponentConfig, BaseDocumentCleanerConfig, BaseModel, Field


DocumentT = TypeVar("DocumentT", bound=Document)
DocumentsT = TypeVar("DocumentsT", bound=Documents)
SourceT = TypeVar("SourceT")
LoadedSourceT = TypeVar("LoadedSourceT")
LoaderInputP = ParamSpec("LoaderInputP")

class _DocumentLoaderConfig(ComponentConfig, Generic[LoaderInputP]):
    component_type: ClassVar = "document_loader"
    load_documents: Callable[LoaderInputP, Documents | Iterable[Document] | Generator[Document, None, ResponseInfo | None]] | Literal["default"] = "default"
    encoding: str = "utf-8"


class DocumentLoaderConfig(BaseDocumentCleanerConfig, _DocumentLoaderConfig[LoaderInputP], Generic[LoaderInputP]):
    component_type: ClassVar = "document_loader"
    add_to_metadata: Optional[list[Literal["source", "mimetype"]]] = ["source"]
    
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

# DocumentLoaderConfig(provider="default")

def iload_documents_mock(
        self, 
        sources: Iterable[SourceT], 
        *args, 
        metadatas: Optional[Iterable[SourceT|dict|None]] = None,  
        **kwargs
        ) -> Iterable[Document]:
    ...

class _FileIODocumentLoaderConfig(ComponentConfig, Generic[LoaderInputP, SourceT, LoadedSourceT]):
    component_type: ClassVar = "document_loader"
    load_documents: Callable[LoaderInputP, Documents | Iterable[Document] | Generator[Document, None, ResponseInfo | None]] | Literal["default"] = Field(default=iload_documents_mock)
    metadata_load_func: Literal["json", "yaml"]|Callable[[IO], dict] = "json"
    source_id_func: Literal["stringify_source", "hash_source"]|Callable[[Any], str] = "stringify_source"
    mimetype_func: Literal["builtin_mimetypes", "magic"]|Callable[[Any], str | None] = "builtin_mimetypes"     

class FileIODocumentLoaderConfig(DocumentLoaderConfig[LoaderInputP], _FileIODocumentLoaderConfig[LoaderInputP, SourceT, LoadedSourceT], Generic[LoaderInputP, SourceT, LoadedSourceT]):
    component_type: ClassVar = "document_loader"
    load_documents: Callable[LoaderInputP, Documents | Iterable[Document] | Generator[Document, None, ResponseInfo | None]] | Literal["default"] = Field(default=iload_documents_mock)
    
    add_to_metadata: Optional[list[Literal["source", "mimetype"]]] = ["source"]
    
# FileIODocumentLoaderConfig()