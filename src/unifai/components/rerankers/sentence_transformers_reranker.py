from __future__ import annotations

from typing import Optional, Any, ClassVar, TYPE_CHECKING
from itertools import product

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from ...types import QueryResult
from ..adapters.sentence_transformers_adapter import SentenceTransformersAdapter
from .._base_components._base_reranker import Reranker
from ...utils import lazy_import

class SentenceTransformersReranker(SentenceTransformersAdapter, Reranker):
    provider = "sentence_transformers"
    default_reranking_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Cache for loaded CrossEncoder models
    ce_model_cache: ClassVar[dict[str, CrossEncoder]] = {}

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

        model = model or self.default_reranking_model        
        model_init_kwargs = {**self.init_kwargs, **kwargs.pop("model_init_kwargs", {})}
        if not (ce_model := self.ce_model_cache.get(model)):
            # ce_model = sentence_transformers.CrossEncoder(
            ce_model = lazy_import("sentence_transformers.CrossEncoder")(
                model_name=model, 
                **model_init_kwargs
            )
            self.ce_model_cache[model] = ce_model     

        pairs = list(product([query], query_result.texts))
        relevance_scores = ce_model.predict(pairs, **kwargs)
        return relevance_scores
    
    # def _extract_reranked_order(
    #     self,
    #     response: Any,
    #     top_n: Optional[int] = None,        
    #     **kwargs
    #     ) -> list[int]:        
    #     return [index for index, score in sorted(enumerate(response), key=lambda x: x[1], reverse=True)[:top_n]]