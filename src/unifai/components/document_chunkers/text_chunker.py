from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, IO, Pattern 

from .._base_components._base_document_chunker import DocumentChunker
from ...types import Document, Documents

class TextDocumentChunker(DocumentChunker):
    provider = "text_chunker"