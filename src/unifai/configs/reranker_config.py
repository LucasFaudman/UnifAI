from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ProviderName
from .tokenizer_config import TokenizerConfig
from ._base_configs import ComponentWithModelConfig

class RerankerConfig(ComponentWithModelConfig):
    component_type: ClassVar = "reranker"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal["rerank", "rerank_documents"], dict[str, Any]]] = None
    