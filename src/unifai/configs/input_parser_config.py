from typing import get_type_hints, Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, AbstractSet, IO, Pattern, Self, ClassVar, TypeVar, Generic
from ._base_configs import ComponentConfigWithCallableNameDefault
from ..types import Message, ToolCall, BaseModel
from ..types.annotations import InputP, InputReturnT


class InputParserConfig(ComponentConfigWithCallableNameDefault, Generic[InputP, InputReturnT]):
    component_type: ClassVar = "input_parser"
    callable: Optional[Callable[InputP, InputReturnT | Callable[..., InputReturnT]]] = None
    return_type: Optional[Type[InputReturnT]] = None
    extra_kwargs: Optional[dict[Literal["parse_input"], dict[str, Any]]] = None

    def __init__(
            self, 
            *,
            name: str = "__name__",
            provider: str = "default",
            callable: Optional[Callable[InputP, InputReturnT | Callable[..., InputReturnT]]] = None,
            return_type: Optional[Type[InputReturnT]] = None,
            extra_kwargs: Optional[dict[Literal["parse_input"], dict[str, Any]]] = None,
            **init_kwargs
            ) -> None:
        if callable:
            if not return_type:
                return_type = get_type_hints(callable).get("return")
            if name == "__name__" and (name := callable.__name__) == "<lambda>":
                name += f"_{id(callable)}" # Add unique id to lambda function names ie <lambda>_123456>                
        elif name == "__name__":
            if not return_type:
                raise ValueError("Either 'callable' or 'return_type' must be provided when 'name' is '__name__' so that a name can be generated")
            name = return_type.__name__
        return BaseModel.__init__(self, **locals())