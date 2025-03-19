from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName
from ._base_configs import ComponentConfigWithDefaultModel
from .tokenizer_config import TokenizerConfig
from ..components._base_components._base_tokenizer import Tokenizer

class LLMConfig(ComponentConfigWithDefaultModel):
    component_type: ClassVar = "llm"
    tokenizer: Optional[Tokenizer | TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal["chat", "chat_stream"], dict[str, Any]]] = None

