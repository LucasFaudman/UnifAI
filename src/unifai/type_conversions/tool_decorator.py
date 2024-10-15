from typing import Callable, Type, Optional, Union, overload, Any
from pydantic import BaseModel

from ..types.tool import Tool
from .tool_from_func import tool_from_func
from .tool_from_pydantic import tool_from_pydantic_model, is_type_and_subclass


@overload
def tool(
        func_or_model: Callable|Type[BaseModel],
    ) -> Tool:
    ...

@overload
def tool(
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
        exclude_fields: Optional[list[str]] = None,
    ) -> Callable[[Callable|Type[BaseModel]], Tool]:
    ...

def tool(
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
        exclude_fields: Optional[list[str]] = None,
    ) -> Tool|Callable[[Callable|Type[BaseModel]], Tool]:
    
    def decorator(func_or_model: Union[Callable[..., Any], Type[BaseModel]]) -> Tool:
        if is_type_and_subclass(func_or_model, BaseModel):
            return tool_from_pydantic_model(
                model=func_or_model,
                name=name,
                description=description,
                type=type,
                strict=strict,
                exclude_fields=exclude_fields
            )
        else:
            return tool_from_func(
                func=func_or_model,
                name=name,
                description=description,
                type=type,
                strict=strict,
            )

    if callable(name):
        # If the first argument is callable, it means the decorator was applied directly without parentheses.
        func_or_model = name
        name = None
        return decorator(func_or_model) # @tool
    return decorator # @tool() or @tool(name="my_tool") etc.

