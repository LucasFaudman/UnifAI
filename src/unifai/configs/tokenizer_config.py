from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar
from ._base_configs import ComponentWithModelConfig

class TokenizerConfig(ComponentWithModelConfig):
    component_type: ClassVar = "tokenizer"
    allowed_special: Literal["all"] | AbstractSet[str] = set()
    disallowed_special: Literal["all"] | list[str] = "all"
    extra_kwargs: Optional[dict[Literal["chunk_documents", "chunk_texts"], dict[str, Any]]] = None

TokenizerConfig(provider="default")