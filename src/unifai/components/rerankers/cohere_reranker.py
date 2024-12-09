from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import QueryResult
from ...exceptions import ProviderUnsupportedFeatureError
from ..base_adapters.cohere_base import CohereAdapter
from .._base_components._base_reranker import Reranker


class CohereReranker(CohereAdapter, Reranker):
    provider = "cohere"
    default_reranking_model = "rerank-multilingual-v3.0"

    # Reranking
    def _get_rerank_response(
        self,
        query: str,
        query_result: QueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> Any:

        if not query_result.texts:
            raise ValueError("Cannot rerank an empty query result")

        return self.client.rerank(
             model=model,
             query=query,
             documents=query_result.texts,
             top_n=top_n,
             return_documents=False,
             **kwargs
        )

    def _extract_similarity_scores(
        self,
        response: Any,
        **kwargs
        ) -> list[float]:
        return [result.relevance_score for result in sorted(response.results, key=lambda result: result.index)]



    # def _extract_reranked_order(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,        
    #     **kwargs
    #     ) -> list[int]:
    #     sorted_results = sorted(response.results, key=lambda result: result.relevance_score, reverse=True)
    #     if top_n is not None and top_n < len(sorted_results):
    #         sorted_results = sorted_results[:top_n]
    #     return [result.index for result in sorted_results]

    
    # def _extract_reranked_order_and_similarity_scores(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> tuple[list[int], list[float]]:
    #     reranked_order, similarity_scores = [], []
    #     for item in sorted(response.results, key=lambda result: result.relevance_score, reverse=True):
    #         reranked_order.append(item.index)
    #         similarity_scores.append(item.relevance_score)
    #     return reranked_order, similarity_scores
