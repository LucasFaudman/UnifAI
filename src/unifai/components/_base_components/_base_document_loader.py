from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, IO
from abc import abstractmethod

from .__base_component import UnifAIComponent
from ...utils import stringify_content, sha256_hash, clean_text, _next
from ...types import Document, Documents
from ...configs.document_loader_config import DocumentLoaderConfig

from copy import deepcopy
from itertools import zip_longest
import re
from typing import Dict, Optional, Union, Pattern, Generic, Type

T = TypeVar("T")
SourceT = TypeVar("SourceT")
LoadedSourceT = TypeVar("LoadedSourceT")

class DocumentLoader(UnifAIComponent[DocumentLoaderConfig], Generic[SourceT, LoadedSourceT]):
    component_type = "document_loader"
    provider = "base"    
    config_class = DocumentLoaderConfig

    @staticmethod
    def get_mimetype_with_builtin_mimetypes(uri: str) -> Optional[str]:
        import mimetypes
        return mimetypes.guess_type(uri)[0]

    @staticmethod
    def get_mimetype_with_magic(uri: str) -> Optional[str]:
        import magic
        return magic.from_file(uri, mime=True)
    
    def _setup(self) -> None:
        super()._setup()
        if self.config.metadata_load_func == "json":
            from json import load
            self._metadata_load_func = load
        elif self.config.metadata_load_func == "yaml":
            from yaml import safe_load
            self._metadata_load_func = safe_load
        else:
            self._metadata_load_func = self.config.metadata_load_func

        if self.config.source_id_func == "stringify_source":
            self._source_id_func = stringify_content
        elif self.config.source_id_func == "hash_source":
            self._source_id_func = sha256_hash
        else:
            self._source_id_func = self.config.source_id_func

        if self.config.mimetype_func == "builtin_mimetypes":
            self._mimetype_func = self.get_mimetype_with_builtin_mimetypes
        elif self.config.mimetype_func == "magic":
            self._mimetype_func = self.get_mimetype_with_magic
        else:
            self._mimetype_func = self.config.mimetype_func                
    
    @abstractmethod            
    def _load_source(self, source: SourceT, *args, **kwargs) -> LoadedSourceT:
        ...
    
    @abstractmethod    
    def _load_metadata(self, source: SourceT, loaded_source: LoadedSourceT, metadata: SourceT|dict|None, *args, **kwargs) -> dict|None:
        ...

    def _add_to_metadata(self, source: SourceT, loaded_source: LoadedSourceT, loaded_metadata: dict, *args, **kwargs) -> dict:
        if not (add_to_metadata := self.config.add_to_metadata):
            return loaded_metadata
        if "source" in add_to_metadata:
            loaded_metadata["source"] = source
        return loaded_metadata    

    def _process_text(self, source: SourceT, loaded_source: LoadedSourceT, loaded_metadata: dict|None, *args, **kwargs) -> str:
        return str(loaded_source) if not isinstance(loaded_source, str) else loaded_source
            
    def _process_metadata(self, source: SourceT, loaded_source: LoadedSourceT, loaded_metadata: dict|None, *args, **kwargs) -> dict|None:
        return loaded_metadata
        
    def _process_id(self, source: SourceT, loaded_source: LoadedSourceT, loaded_metadata: dict|None, *args, **kwargs) -> str:
        return self._source_id_func(source)
    
    def _load_document(self, source: SourceT, metadata: SourceT|dict|None, *args, **kwargs) -> Document|None|Exception:
        loaded_source = load_source_exception = None
        for _ in range(max(self.config.error_retries["source_load_error"] + 1, 1)):
            try:
                loaded_source = self._load_source(source, *args, **kwargs)
                load_source_exception = None
                break
            except Exception as e:
                load_source_exception = e
                
        if loaded_source is None: 
            if self.config.error_handling["source_load_error"] == "skip":
                return None
            elif load_source_exception is not None:
                return load_source_exception
            else:
                return ValueError("Source could not be loaded")
        

        loaded_metadata = load_metadata_exception = None
        for _ in range(max(self.config.error_retries["metadata_load_error"] + 1, 1)):
            try:
                loaded_metadata = self._load_metadata(source, loaded_source, metadata, *args, **kwargs)
                if self.config.add_to_metadata:
                    if loaded_metadata is None:
                        loaded_metadata = {}
                    loaded_metadata = self._add_to_metadata(source, loaded_source, loaded_metadata, *args, **kwargs)
                load_metadata_exception = None
                break
            except Exception as e:
                load_metadata_exception = e
        
        if load_metadata_exception is not None:
            if self.config.error_handling["metadata_load_error"] == "skip":
                return None
            else:
                return load_metadata_exception
                
        processor_exception = None
        for _ in range(max(self.config.error_retries["processor_error"] + 1, 1)):
            try:
                processed_text = self._process_text(source, loaded_source, loaded_metadata, *args, **kwargs)
                final_text = clean_text(processed_text, self.config.replacements, self.config.strip_chars)                
                processed_metadata = self._process_metadata(source, loaded_source, loaded_metadata, *args, **kwargs)
                _id = self._process_id(source, loaded_source, loaded_metadata, *args, **kwargs)                
                processor_exception = None
                break
            except Exception as e:
                processor_exception = e

        if processor_exception is not None:
            if self.config.error_handling["processor_error"] == "skip":
                return None
            else:
                return processor_exception

        return Document(id=_id, text=final_text, metadata=processed_metadata)
    
    def iload_documents(
            self, 
            sources: Iterable[SourceT], 
            *args, 
            metadatas: Optional[Iterable[SourceT|dict|None]] = None,  
            **kwargs
            ) -> Iterable[Document]:
        
        deepcopy_metadata = self.config.deepcopy_metadata
        for source, metadata in zip_longest(sources, metadatas or ()):            
            if deepcopy_metadata and metadata is not None:
                metadata = deepcopy(metadata) # Deepcopy the metadata before processing and potentially modifying it
            if not (load_result := self._load_document(source, metadata, *args, **kwargs)):
                continue # Skip the document since error_handling is "skip"
            if isinstance(load_result, Document):
                yield load_result # Yield the document
            else:
                raise load_result # Raise the exception

    def load_documents(
            self, 
            sources: Iterable[SourceT], 
            *args, 
            metadatas: Optional[Iterable[SourceT|dict|None]] = None,  
            **kwargs
            ) -> list[Document]|Documents:
        return list(self.iload_documents(sources, *args, metadatas=metadatas, **kwargs))

    def load_document(
            self, 
            source: SourceT, 
            *args, 
            metadata: SourceT|dict|None = None,  
            **kwargs
            ) -> Document|None:
        return _next(self.iload_documents([source], *args, metadatas=[metadata], **kwargs))