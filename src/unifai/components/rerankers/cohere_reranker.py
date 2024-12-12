from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Collection,  Callable, Iterator, Iterable, Generator, Self

from ...types import QueryResult
from ...exceptions import ProviderUnsupportedFeatureError
from ..adapters.cohere_adapter import CohereAdapter
from .._base_components._base_reranker import Reranker


class CohereReranker(Reranker, CohereAdapter):
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
