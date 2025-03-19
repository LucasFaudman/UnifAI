from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ProviderName
from ._base_configs import ComponentConfigWithDefaultModel
from .tokenizer_config import TokenizerConfig
from ..components._base_components._base_tokenizer import Tokenizer


class RerankerConfig(ComponentConfigWithDefaultModel):
    component_type: ClassVar = "reranker"
    tokenizer: Optional[Tokenizer | TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal["rerank", "rerank_documents"], dict[str, Any]]] = None

RerankerConfig()