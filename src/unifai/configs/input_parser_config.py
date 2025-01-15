from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, TypeVar, Generic
from ._base_configs import ComponentConfig
from ..types import Message, ToolCall
from ..types.annotations import InputP

InputReturnT = TypeVar('InputReturnT', Message, str)

class InputParserConfig(ComponentConfig, Generic[InputP, InputReturnT]):
    component_type: ClassVar = "input_parser"
    input_parser: Callable[InputP, InputReturnT | Callable[..., InputReturnT]]
    extra_kwargs: Optional[dict[Literal["__call__"], dict[str, Any]]] = None