from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ._base_client_wrapper import BaseClientWrapper, UnifAIExceptionConverter

from unifai.types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, AIProvider, VectorDBGetResult, VectorDBQueryResult
from unifai.exceptions import UnifAIError, ProviderUnsupportedFeatureError
from pydantic import BaseModel

T = TypeVar("T")

class RerankerClient(BaseClientWrapper):
    provider = "base_reranker"
    default_reranking_model = "rerank-english-v3.0"

    def rerank(
        self, 
        query: str, 
        query_result: VectorDBQueryResult,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        **reranker_kwargs
        ) -> VectorDBQueryResult:
        
        rerank_response = self.run_func_convert_exceptions(
            func=self._get_rerank_response,
            query=query,
            query_result=query_result,
            model=model or self.default_reranking_model,
            top_n=top_n,
            **reranker_kwargs
        )
        reranked_order = self._extract_reranked_order(rerank_response)
        query_result.rerank(reranked_order)
        if top_n is not None:
            query_result.reduce_to_top_n(top_n)
        return query_result


    def _get_rerank_response(
        self,
        query: str,
        query_result: VectorDBQueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")
        

    def _extract_reranked_order(
        self,
        response: Any,
        **kwargs
        ) -> list[int]:
        raise NotImplementedError("This method must be implemented by the subclass")    
