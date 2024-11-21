from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Iterable,  Callable, Iterator, Iterable, Generator, Self, IO
from pathlib import Path

from .._base_component import UnifAIComponent
from ...types import Document, Documents
import re
from typing import Dict, Optional, Union, Pattern

T = TypeVar("T")
class DocumentLoader(UnifAIComponent):
    provider = "base_document_loader"

    processor_func: Optional[Callable[[Path, str, Optional[dict]], tuple[str, str, Optional[dict]]]]
    metadata_load_func: Optional[Callable[[IO], dict]]
    mimetype_func: Optional[Callable[[Path], str|None]]
    default_replacements: dict[str|Pattern, str] = {
            # Remove backspace formatting (common in man pages)
            r'.\x08': '',
            # Replace common control chars with space
            r'[\x00-\x08\x0B-\x1F\x7F-\x9F]+': ' ',
            # Replace tabs and newlines with space
            r'[\t\n\r]+': ' ',            
            # Normalize whitespace (including Unicode whitespace)
            r'\s+': ' ',
    }

    def iload_documents(self, *args, **kwargs) -> Iterable[Document]:
        raise NotImplementedError("This method must be implemented by the subclass")

    def load_documents(self, *args, **kwargs) -> list[Document]|Documents:
        return list(self.iload_documents(*args, **kwargs))

    def clean_text(
        self,
        text: str,
        replacements: Optional[dict[str|Pattern, str]] = {
            # r'.\x08': '', # Remove backspace formatting
            r'[\x00-\x08\x0B-\x1F\x7F-\x9F]+': ' ', # Replace common control chars with space
            r'[\t\n\r]+': ' ', # Replace tabs and newlines with space            
            r'\s+': ' ', # Normalize whitespace (including Unicode whitespace)
        },
        strip_chars: Optional[str|Literal[False]] = None,
        **kwargs
    ) -> str:
        if not text:
            return text

        if replacements is None:
            replacements = self.default_replacements
        
        for pattern, replacement in replacements.items():
            if not isinstance(pattern, Pattern):
                pattern = re.compile(pattern)
            text = pattern.sub(replacement, text)
        
        if strip_chars is not False:
            text = text.strip(strip_chars)
        return text