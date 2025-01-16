from typing import TYPE_CHECKING, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, TypeVar, Generic
from ._base_configs import ComponentConfig
from ..types import Message, ToolCall
from ..types.annotations import OutputT, ReturnT

# if TYPE_CHECKING:
#     from ..components.chats import Chat

# OutputT = TypeVar('OutputT', Message, ToolCall, "Chat", str)

class OutputParserConfig(ComponentConfig, Generic[OutputT, ReturnT]):
    component_type: ClassVar = "output_parser"
    output_type: Type[OutputT]
    return_type: Type[ReturnT]
    extra_kwargs: Optional[dict[Literal["parse_output"], dict[str, Any]]] = None