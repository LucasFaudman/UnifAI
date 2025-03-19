from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_tokenizer import Tokenizer
    from ..types.annotations import ComponentName, ProviderName

from ..utils import copy_init_from
from ..configs.tokenizer_config import TokenizerConfig
from ._base_client import BaseClient

class UnifAITokenizerClient(BaseClient):

    def _get_tokenizer(
            self, 
            config_or_name: "TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]" = "default",
            **init_kwargs
            ) -> "Tokenizer":
        return self._get_component("tokenizer", config_or_name, init_kwargs)

    def tokenizer_from_config(
            self, 
            config: "TokenizerConfig",
            **init_kwargs
            ) -> "Tokenizer":
        return self._get_tokenizer(config, **init_kwargs)
    
    def tokenizer_from_name(
            self,
            provider: "ProviderName" = "default",
            name: "ComponentName" = "default",
            **init_kwargs
            ) -> "Tokenizer":
        return self._get_tokenizer((provider, name), **init_kwargs)
    
    @copy_init_from(TokenizerConfig.__init__)
    def tokenizer(self, *args, **kwargs) -> "Tokenizer":
        config = TokenizerConfig(*args, **kwargs)
        return self._get_tokenizer(config)