from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self


# from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
# from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ._base_component import UnifAIComponent

T = TypeVar("T")

class DocumentDB(UnifAIComponent):
    provider = "document_db"

    def get_documents(self, ids: Collection[str]) -> Iterable[str]:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def get_document(self, id: str) -> str:
        return next(iter(self.get_documents([id])))
    
    def set_documents(self, ids: Collection[str], documents: Collection[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def set_document(self, id: str, document: str) -> None:
        self.set_documents([id], [document])
    
    def delete_documents(self, ids: Collection[str]) -> None:
        raise NotImplementedError("This method must be implemented by the subclass")    

    def delete_document(self, id: str) -> None:
        self.delete_documents([id])