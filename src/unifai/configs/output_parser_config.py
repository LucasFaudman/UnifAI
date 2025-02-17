from typing import TYPE_CHECKING, get_type_hints, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, TypeVar, Generic
from ._base_configs import ComponentConfigWithCallableNameDefault
from ..types import Message, ToolCall, BaseModel
from ..types.annotations import OutputT, ReturnT, ComponentName, ProviderName
from pydantic import model_validator, PrivateAttr, Field

# if TYPE_CHECKING:
#     from ..components.chats import Chat

# OutputT = TypeVar('OutputT', Message, ToolCall, "Chat", str)

class OutputParserConfig(ComponentConfigWithCallableNameDefault, Generic[OutputT, ReturnT]):
    component_type: ClassVar = "output_parser"
    callable: Optional[Callable[[OutputT], ReturnT]] = None
    output_type: Type[OutputT]
    return_type: Optional[Type[ReturnT]] = None
    extra_kwargs: Optional[dict[Literal["parse_output"], dict[str, Any]]] = None

    def __init__(
            self, 
            *,
            name: ComponentName = "__name__",
            provider: ProviderName = "default",
            callable: Optional[Callable[[OutputT], ReturnT]] = None,
            output_type: Optional[Type[OutputT]] = None,
            return_type: Optional[Type[ReturnT]] = None,
            extra_kwargs: Optional[dict[Literal["parse_output"], dict[str, Any]]] = None,
            **init_kwargs
            ) -> None:
        if callable:
            annos = get_type_hints(callable)
            if not (output_type := output_type or annos.get("output")):
                raise ValueError("Callable must have 'output' type hint or 'output_type' must be provided")        
            if not return_type:
                return_type = annos.get("return")
            if name == "__name__" and (name := callable.__name__) == "<lambda>":
                name += f"_{id(callable)}" # Add unique id to lambda function names ie <lambda>_123456>
        elif not output_type:
            raise ValueError("Either 'callable' or 'output_type' must be provided")
        elif name == "__name__":
            if not return_type:
                raise ValueError("Either 'callable' or 'return_type' must be provided when 'name' is '__name__' so that a name can be generated")
            name = return_type.__name__        
        return BaseModel.__init__(self, **locals())
            