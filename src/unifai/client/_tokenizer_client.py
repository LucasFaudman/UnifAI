from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

if TYPE_CHECKING:
    from ..components._base_components._base_tokenizer import Tokenizer
    from ..configs.tokenizer_config import TokenizerConfig
    from ..types.annotations import ComponentName, ProviderName

from ._base_client import BaseClient

class UnifAITokenizerClient(BaseClient):
    
    def tokenizer(
            self, 
            provider_config_or_name: "ProviderName | TokenizerConfig | tuple[ProviderName, ComponentName]" = "default",          
            **init_kwargs
            ) -> "Tokenizer":
        return self._get_component("tokenizer", provider_config_or_name, init_kwargs)