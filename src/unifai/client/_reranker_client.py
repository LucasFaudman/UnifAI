from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_reranker import Reranker
    from ..types.annotations import ComponentName, ProviderName
    from ..types.db_results import QueryResult, RerankedQueryResult
    from ..configs.reranker_config import RerankerConfig

from ._base_client import BaseClient

class UnifAIRerankClient(BaseClient):

    def reranker(
            self, 
            provider_config_or_name: "ProviderName | RerankerConfig | tuple[ProviderName, ComponentName]" = "default",       
            **init_kwargs
            ) -> "Reranker":
        return self._get_component("reranker", provider_config_or_name, init_kwargs) 

    def rerank(
        self, 
        query: str, 
        query_result: "QueryResult",
        provider: str = "default",
        model: Optional[str] = None,
        top_n: Optional[int] = None,
        **reranker_kwargs
        ) -> "RerankedQueryResult":

        return self.reranker(provider).rerank(
            query, 
            query_result, 
            model, 
            top_n, 
            **reranker_kwargs
        )    
    