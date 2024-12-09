from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self

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

    # List Models
    def list_models(self) -> list[str]:
        raise NotImplementedError("This method must be implemented by the subclass")     

    def _get_rerank_response(
        self,
        query: str,
        query_result: QueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:
        raise NotImplementedError("This method must be implemented by the subclass")

    def _extract_similarity_scores(
        self,
        response: Any,
        **kwargs
        ) -> list[float]:
        return response       

    def rerank(
        self, 
        query: str, 
        query_result: QueryResult,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        min_similarity_score: Optional[float] = None,
        **reranker_kwargs
        ) -> RerankedQueryResult:
        
        rerank_response = self.run_func_convert_exceptions(
            func=self._get_rerank_response,
            query=query,
            query_result=query_result,
            model=model or self.default_reranking_model,
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




    # def _extract_reranked_order_and_similarity_scores(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> tuple[list[int], list[float]]:
    #     raise NotImplementedError("This method must be implemented by the subclass")        
        
    # def _extract_reranked_order(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> list[int]:
    #     raise NotImplementedError("This method must be implemented by the subclass")    

    # def _extract_similarity_scores(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> list[float]:
    #     raise NotImplementedError("This method must be implemented by the subclass")  

    # def rerank(
    #     self, 
    #     query: str, 
    #     query_result: QueryResult,
    #     model: Optional[str] = None,
    #     top_n: Optional[int] = None,
    #     **reranker_kwargs
    #     ) -> QueryResult:
        
    #     rerank_response = self.run_func_convert_exceptions(
    #         func=self._get_rerank_response,
    #         query=query,
    #         query_result=query_result,
    #         model=model or self.default_reranking_model,
    #         top_n=top_n,
    #         **reranker_kwargs
    #     )
    #     reranked_order = self._extract_reranked_order(rerank_response)
    #     query_result.rerank(reranked_order)
    #     if top_n is not None:
    #         query_result.reduce_to_top_n(top_n)
    #     return query_result


    # def _get_rerank_response(
    #     self,
    #     query: str,
    #     query_result: QueryResult,
    #     model: str,
    #     top_n: Optional[int] = None,               
    #     **kwargs
    #     ) -> Any:
    #     raise NotImplementedError("This method must be implemented by the subclass")
        

    # def _extract_reranked_order(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> list[int]:
    #     raise NotImplementedError("This method must be implemented by the subclass")    
