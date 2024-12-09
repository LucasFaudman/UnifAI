from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, TypeVar, Generic
from ._base_configs import ComponentConfig
from ..types import Message, ToolCall

OutputT = TypeVar('OutputT', Message, ToolCall, str)
ReturnT = TypeVar('ReturnT')
class OutputParserConfig(ComponentConfig, Generic[OutputT, ReturnT]):
    component_type: ClassVar = "output_parser"
    output_type: Type[OutputT]
    return_type: Type[ReturnT]
    extra_kwargs: Optional[dict[Literal["parse_output"], dict[str, Any]]] = None