from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName
from ._base_configs import BaseDocumentCleanerConfig, ComponentConfig, BaseModel, Field
from .tokenizer_config import TokenizerConfig

class _BaseDocumentChunkerConfig(ComponentConfig):
    component_type: ClassVar = "document_chunker"
    chunk_size: int = 4200
    chunk_overlap: int | float = .2
    separators: list[str] = ["\n\n", "\n", " ", ""]
    keep_separator: Literal["start", "end", False] = False            
    regex: bool = False    
    size_function: Literal["tokens", "characters", "words"]|Callable[[str], int] = "tokens"
    tokenizer: TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName] = "default"
    tokenizer_model: Optional[ModelName] = None    


class DocumentChunkerConfig(BaseDocumentCleanerConfig, _BaseDocumentChunkerConfig):
    component_type: ClassVar = "document_chunker"
    default_base_id: str = "doc"
    extra_kwargs: Optional[dict[Literal["chunk_documents", "chunk_texts", "size_function"], dict[str, Any]]] = None

DocumentChunkerConfig(provider="default")