from typing import Optional, Type, TypeVar, Any, Union, Collection, Mapping, List, Tuple, Annotated, Callable, _SpecialForm, Literal, get_args, get_origin
from types import UnionType
from enum import Enum
from pydantic import BaseModel

from unifai.types import (
    StringToolParameter,
    NumberToolParameter,
    IntegerToolParameter,
    BooleanToolParameter,
    NullToolParameter,
    ArrayToolParameter,
    ObjectToolParameter,
    AnyOfToolParameter,
    RefToolParameter,
    Tool,    
)

from .construct_tool_parameter import construct_tool_parameter

def tool_from_pydantic(
        model: Type[BaseModel]|BaseModel, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
        exclude: Optional[Collection[str]] = None,
        ) -> Tool:
    
    if isinstance(model, BaseModel):
        model = model.__class__

    parameters = construct_tool_parameter({"type": model}, exclude=exclude or ())
    if isinstance(parameters, AnyOfToolParameter):
        raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")
    if not isinstance(parameters, ObjectToolParameter):
        raise ValueError("Root parameter must be an object")
    
    name = name or f"return_{parameters.name}"
    description = description or parameters.description or f"Return {parameters.name} object"
    parameters.name = None
    parameters.description = None

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        type=type,
        strict=strict,
        callable=model
    )

# alias for tool_from_pydantic_model so models can be decorated with @tool or @tool_from_model or longform @tool_from_pydantic_model
tool_from_model = tool_from_pydantic