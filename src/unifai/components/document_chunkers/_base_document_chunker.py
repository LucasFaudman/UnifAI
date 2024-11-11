from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self


# from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
# from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ..base_adapters._base_adapter import UnifAIAdapter
from .._base_component import UnifAIComponent

T = TypeVar("T")

class DocumentChunker(UnifAIComponent):
    provider = "document_chunker"

    def chunk_document(self, document: str) -> Iterable[str]:
        raise NotImplementedError("This method must be implemented by the subclass")