from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar

from ..types.annotations import ComponentName, ModelName, ProviderName
from ._base_configs import ComponentWithModelConfig
from .tokenizer_config import TokenizerConfig

class LLMConfig(ComponentWithModelConfig):
    component_type: ClassVar = "llm"
    tokenizer: Optional[TokenizerConfig | ProviderName | tuple[ProviderName, ComponentName]] = None
    extra_kwargs: Optional[dict[Literal["chat", "chat_stream"], dict[str, Any]]] = None

