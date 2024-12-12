from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self
from abc import abstractmethod

from ._base_adapter import UnifAIAdapter, UnifAIComponent

from ...types import Message, MessageChunk, Tool, ToolCall, Image, ResponseInfo, Embedding, Embeddings, Usage, ProviderName, GetResult, QueryResult, RerankedQueryResult
from ...exceptions import UnifAIError, ProviderUnsupportedFeatureError
from ...configs.reranker_config import RerankerConfig

T = TypeVar("T")

class Reranker(UnifAIAdapter[RerankerConfig]):
    component_type = "reranker"
    provider = "base"    
    config_class = RerankerConfig
    can_get_components = False

    default_reranking_model = "rerank-english-v3.0"

    # Abstract Methods
    @abstractmethod
    def _get_rerank_response(
        self,
        query: str,
        query_result: QueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:
        ...

    def _extract_similarity_scores(
        self,
        response: Any,
        **kwargs
        ) -> list[float]:
        return response       

    def _list_models(self) -> list[str]:
        return [self.default_reranking_model]
    

    # Concrete Methods
    def list_models(self) -> list[str]:
        return self._run_func(self._list_models)
    
    @property
    def default_model(self) -> str:
        return self.config.default_model or self.default_reranking_model

    def rerank(
        self, 
        query: str, 
        query_result: QueryResult,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        min_similarity_score: Optional[float] = None,
        **reranker_kwargs
        ) -> RerankedQueryResult:
        
        rerank_response = self._run_func(
            self._get_rerank_response,
            query=query,
            query_result=query_result,
            model=model or self.default_model,
            top_n=top_n,
            **reranker_kwargs
        )
        similarity_scores = self._extract_similarity_scores(rerank_response)
        reranked_query_result = RerankedQueryResult.from_query_result(query_result, similarity_scores)
        if min_similarity_score is not None:
            reranked_query_result.trim_by_similarity_score(min_similarity_score)
        if top_n is not None:
            reranked_query_result.reduce_to_top_n(top_n)
        return reranked_query_result