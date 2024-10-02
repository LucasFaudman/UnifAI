from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator

from ._base_client_wrapper import BaseClientWrapper

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embeddings, Usage
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError

T = TypeVar("T")

class EmbeddingClient(BaseClientWrapper):
    provider = "base_embedding"
    default_embedding_model = "llama3.1:8b-instruct-q2_K"


    # Embeddings
    def embed(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> Embeddings:
        
        kwargs["input"] = input
        kwargs["model"] = model or self.default_embedding_model
        kwargs["max_dimensions"] = max_dimensions

        response = self.run_func_convert_exceptions(
            func=self._get_embed_response,
            **kwargs
        )
        return self._extract_embeddings(response, **kwargs)
    

    def _get_embed_response(
            self,            
            input: str | Sequence[str],
            model: Optional[str] = None,
            max_dimensions: Optional[int] = None,
            **kwargs
            ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
    

    def _extract_embeddings(
            self,            
            response: Any,
            **kwargs
            ) -> Embeddings:
        raise NotImplementedError("This method must be implemented by the subclass")


   