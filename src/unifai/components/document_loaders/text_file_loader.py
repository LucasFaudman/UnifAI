from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, IO, Pattern 

from .._base_components._base_component import UnifAIComponent
from ...types import Document, Documents
from .._base_components._base_document_loader import DocumentLoader

from pathlib import Path

URIIOTuple = tuple[str, IO]
URITextTuple = tuple[str, str]
PathStrOrURIIOTuple = Union[Path, str, URIIOTuple]
class TextFileDocumentLoader(DocumentLoader[PathStrOrURIIOTuple, URITextTuple]):
    provider = "text_file_loader"

    def _load_source(self, source: PathStrOrURIIOTuple, *args, **kwargs) -> URITextTuple:
        if isinstance(source, str):
            source = Path(source)
        if isinstance(source, Path):
            return str(source), source.read_text(encoding=self.config.encoding)
        uri, source_io = source
        out = source_io.read()
        if isinstance(out, bytes):
            out = out.decode(self.config.encoding)
        return uri, out
    
    def _load_metadata(self, source: PathStrOrURIIOTuple, loaded_source: URITextTuple, metadata: PathStrOrURIIOTuple|dict|None, *args, **kwargs) -> dict|None:
        if metadata is None or isinstance(metadata, dict):
            return metadata # metadata already loaded or None
        
        if isinstance(metadata, str):
            metadata = Path(metadata)
        if isinstance(metadata, Path):
            metadata_io = metadata.open(encoding=self.config.encoding)
        else:
            metadata_io = metadata[1] # metadata is a URIIOTuple     

        loaded_metadata = self._metadata_load_func(metadata_io)
        if not metadata_io.closed and not metadata_io.seekable():
            metadata_io.close()
        return loaded_metadata

    def _add_to_metadata(self, source: PathStrOrURIIOTuple, loaded_source: URITextTuple, loaded_metadata: dict, *args, **kwargs) -> dict:
        loaded_metadata = super()._add_to_metadata(source, loaded_source, loaded_metadata, *args, **kwargs)
        if self.config.add_to_metadata and "mimetype" in self.config.add_to_metadata:
            loaded_metadata["mimetype"] = self._mimetype_func(loaded_source[0])
        return loaded_metadata

    def _process_text(self, source: PathStrOrURIIOTuple, loaded_source: URITextTuple, loaded_metadata: dict|None, *args, **kwargs) -> str:
        return loaded_source[1]
            
    def _process_metadata(self, source: PathStrOrURIIOTuple, loaded_source: URITextTuple, loaded_metadata: dict|None, *args, **kwargs) -> dict|None:
        return loaded_metadata
        
    def _process_id(self, source: PathStrOrURIIOTuple, loaded_source: URITextTuple, loaded_metadata: dict|None, *args, **kwargs) -> str:
        return self._source_id_func(loaded_source[0]) 
    