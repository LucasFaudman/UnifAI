from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_reranker import Reranker
    from ..types.annotations import ComponentName, ProviderName
    from ..types.db_results import QueryResult, RerankedQueryResult

from ..utils import copy_init_from
from ..configs.reranker_config import RerankerConfig

from ._base_client import BaseClient

class UnifAIRerankClient(BaseClient):

    def _get_reranker(
            self, 
            config_or_name: "RerankerConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",       
            **init_kwargs
            ) -> "Reranker":
        return self._get_component("reranker", config_or_name, init_kwargs) 

    def reranker_from_config(
            self, 
            config: "RerankerConfig",       
            **init_kwargs
            ) -> "Reranker":
        return self._get_reranker(config, **init_kwargs)
    
    def reranker_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "Reranker":
        return self._get_reranker((provider, name), **init_kwargs)
    
    @copy_init_from(RerankerConfig.__init__)
    def reranker(self, *args, **kwargs) -> "Reranker":
        config = RerankerConfig(*args, **kwargs)
        return self._get_reranker(config)