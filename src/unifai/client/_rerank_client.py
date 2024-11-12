from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components.rerankers._base_reranker import Reranker, VectorDBQueryResult

from ._base_client import BaseClient

class UnifAIRerankClient(BaseClient):

    def get_reranker(self, provider: Optional[str] = None, **client_kwargs) -> "Reranker":
        return self._get_component(provider, "reranker", **client_kwargs)    

    def rerank(
        self, 
        query: str, 
        query_result: "VectorDBQueryResult",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        **reranker_kwargs
        ) -> "VectorDBQueryResult":

        return self.get_reranker(provider).rerank(
            query, 
            query_result, 
            model, 
            top_n, 
            **reranker_kwargs
        )    
    