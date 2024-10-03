from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter
from ._base_embedding_client import EmbeddingClient
from ._base_llm_client import LLMClient
from ._base_reranker_client import RerankerClient

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, LLMProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError, STATUS_CODE_TO_EXCEPTION_MAP, UnknownAPIError

from cohere import ClientV2
from cohere.core import ApiError as CohereAPIError

T = TypeVar("T")

class CohereWrapper(EmbeddingClient, RerankerClient, LLMClient):
    client: ClientV2

    provider = "cohere"
    default_model = "command"
    default_embedding_model = "embed-multilingual-v3.0"
    default_reranking_model = "rerank-multilingual-v3.0"
    

    def import_client(self):
        from cohere import ClientV2
        return ClientV2
    
    def init_client(self, **client_kwargs) -> ClientV2:
        self.client_kwargs.update(client_kwargs)
        if not (api_key := self.client_kwargs.get("api_key")):
            raise ValueError("Cohere API key is required")
        self._client = self.import_client()(api_key) # Cohere ClientV2 does require an API key as a positional argument
        return self._client


    # Convert Exceptions from AI Provider Exceptions to UnifAI Exceptions
    def convert_exception(self, exception: CohereAPIError) -> UnifAIError:
        message = exception.body
        status_code = exception.status_code
        if status_code is not None:
                unifai_exception_type = STATUS_CODE_TO_EXCEPTION_MAP.get(status_code, UnknownAPIError)
        else:
                unifai_exception_type = UnknownAPIError
        return unifai_exception_type(
            message=message,
            status_code=status_code,
            original_exception=exception
        )        
    

    # Embeddings
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
        

    # Reranking
    def _get_rerank_response(
        self,
        query: str,
        query_result: VectorDBQueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:

        if not query_result.documents:
            raise ValueError("Cannot rerank an empty query result")

        return self.client.rerank(
             model=model,
             query=query,
             documents=query_result.documents,
             top_n=top_n,
             **kwargs
        )
        

    def _extract_reranked_order(
        self,
        response: Any,
        top_n: Optional[int] = None,        
        **kwargs
        ) -> list[int]:
        sorted_results = sorted(response.results, key=lambda result: result.relevance_score, reverse=True)
        if top_n is not None and top_n < len(sorted_results):
            sorted_results = sorted_results[:top_n]
        return [result.index for result in sorted_results]