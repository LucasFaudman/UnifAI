from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, IO, Pattern 

from .._base_component import UnifAIComponent
from ...types import Document, Documents
from ._base_document_loader import DocumentLoader

from itertools import zip_longest
from pathlib import Path
import json


T = TypeVar("T")

class TextFileDocumentLoader(DocumentLoader):
    provider = "text_file_loader"

    @staticmethod
    def get_mimetype_with_builtin_mimetypes(path: Path) -> Optional[str]:
        import mimetypes
        return mimetypes.guess_type(path.as_uri())[0]

    @staticmethod
    def get_mimetype_with_magic(Path: Path) -> Optional[str]:
        import magic
        return magic.from_file(str(Path), mime=True)

    @staticmethod
    def get_id_from_path_leave_text_and_metadata_as_is(path: Path, text: str, metadata: Optional[dict]) -> tuple[str, str, Optional[dict]]:
        return str(path), text, metadata    

    def __init__(
            self, 
            processor_func: Optional[Callable[[Path, str, Optional[dict]], tuple[str, str, Optional[dict]]]] = get_id_from_path_leave_text_and_metadata_as_is,
            metadata_load_func: Optional[Callable[[IO], dict]] = json.load,   
            mimetype_func: Optional[Callable[[Path], str|None]] = get_mimetype_with_builtin_mimetypes,
            include_path_in_metadata: bool = True,
            include_mimetype_in_metadata: bool = False,
            encoding: str = "utf-8",
            open_kwargs: Optional[dict] = None,           
            raise_on_file_read_error: bool = True,
            raise_on_metadata_load_error: bool = True,
            replacements: Optional[dict[str|Pattern, str]|Literal[False]] = {
                r'.\x08': '', # Remove backspace formatting
                r'[\x00-\x08\x0B-\x1F\x7F-\x9F]+': ' ', # Replace common control chars with space
                r'[\t\n\r]+': ' ', # Replace tabs and newlines with space            
                r'\s+': ' ', # Normalize whitespace (including Unicode whitespace)
            },             
            **kwargs
            ):
        self.processor_func = processor_func
        self.metadata_load_func = metadata_load_func      
        self.mimetype_func = mimetype_func  
        self.include_path_in_metadata = include_path_in_metadata
        self.include_mimetype_in_metadata = include_mimetype_in_metadata
        self.encoding = encoding
        self.open_kwargs = open_kwargs or {}
        self.raise_on_file_read_error = raise_on_file_read_error
        self.raise_on_metadata_load_error = raise_on_metadata_load_error
        self.replacements = replacements
        

    def iload_documents(
            self,
            paths: Iterable[Path|str],
            metadatas: Optional[Iterable[dict|Path|str]] = None,
            processor_func: Optional[Callable[[Path, str, Optional[dict]], tuple[str, str, Optional[dict]]]] = None,
            metadata_load_func: Optional[Callable[[IO], dict]] = None,    
            mimetype_func: Optional[Callable[[Path], str|None]] = None,        
            include_path_in_metadata: Optional[bool] = None,
            include_mimetype_in_metadata: Optional[bool] = None,
            encoding: Optional[str] = None,
            open_kwargs: Optional[dict] = None,            
            raise_on_file_read_error: Optional[bool] = None,
            raise_on_metadata_load_error: Optional[bool] = None,
            replacements: Optional[dict[str|Pattern, str]|Literal[False]] = None,
            **kwargs
    ) -> Iterable[Document]:

        processor_func = processor_func if processor_func is not None else self.processor_func
        metadata_load_func = metadata_load_func if metadata_load_func is not None else self.metadata_load_func        
        mimetype_func = mimetype_func if mimetype_func is not None else self.mimetype_func
        include_path_in_metadata = include_path_in_metadata if include_path_in_metadata is not None else self.include_path_in_metadata
        include_mimetype_in_metadata = include_mimetype_in_metadata if include_mimetype_in_metadata is not None else self.include_mimetype_in_metadata
        encoding = encoding if encoding is not None else self.encoding
        raise_on_file_read_error = raise_on_file_read_error if raise_on_file_read_error is not None else self.raise_on_file_read_error
        raise_on_metadata_load_error = raise_on_metadata_load_error if raise_on_metadata_load_error is not None else self.raise_on_metadata_load_error        
        replacements = replacements if replacements is not None else self.replacements

        open_kwargs = open_kwargs if open_kwargs is not None else self.open_kwargs
        open_kwargs = {**open_kwargs, "mode": "r", "encoding": encoding}
        init_metadata_if_none = include_path_in_metadata or include_mimetype_in_metadata

        for path, metadata in zip_longest(paths, metadatas or ()):
            if isinstance(path, str):
                path = Path(path)
            try:
                with path.open(**open_kwargs) as f:
                    text = f.read()
            except Exception as e:
                if raise_on_file_read_error:
                    raise e
                continue

            if metadata is None:
                if init_metadata_if_none:
                    metadata = {}
            if isinstance(metadata, str):
                metadata = Path(metadata)
            if isinstance(metadata, Path):
                if metadata_load_func is None:
                    raise ValueError("metadata_load_func must be provided if metadata is a str|Path object")
                try:
                    with metadata.open(**open_kwargs) as f:
                        metadata = metadata_load_func(f)
                except Exception as e:
                    if raise_on_metadata_load_error:
                        raise e
                    metadata = {}
            
            if include_path_in_metadata:
                metadata["path"] = str(path)
            if include_mimetype_in_metadata:       
                if mimetype_func is None:
                    raise ValueError("mimetype_func must be provided if include_mimetype_in_metadata is True")         
                metadata["mimetype"] = mimetype_func(path)
            
            if replacements is not False:
                text = self.clean_text(text, replacements)

            if processor_func is not None:
                _id, text, metadata = processor_func(path, text, metadata)
            else:
                _id = str(path)                    
            yield Document(id=_id, text=text, metadata=metadata)
    