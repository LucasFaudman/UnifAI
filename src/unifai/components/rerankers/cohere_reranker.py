from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import VectorDBQueryResult
from ...exceptions import ProviderUnsupportedFeatureError
from ..base_adapters.cohere_base import CohereAdapter
from ._base_reranker import Reranker


class CohereReranker(CohereAdapter, Reranker):
    provider = "cohere"
    default_reranking_model = "rerank-multilingual-v3.0"

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