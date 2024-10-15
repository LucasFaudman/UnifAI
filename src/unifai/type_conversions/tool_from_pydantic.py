from typing import Optional, Type, TypeVar, Any, Union, Sequence, Mapping, List, Tuple, Annotated, _SpecialForm
from types import UnionType

from unifai.types import (
    ToolParameter,
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
    ProviderTool,
    PROVIDER_TOOLS,
    
)
from pydantic import BaseModel, Field
from enum import Enum
from typing import Literal, get_args, get_origin, Any
from .tool_from_dict import tool_parameter_from_dict
from .tool_from_func import PY_TYPE_TO_TOOL_PARAMETER_TYPE_MAP
T = TypeVar("T")


def get_enum_or_literal_values(field_type: Any) -> Optional[tuple[type, list]]:
    """
    Extracts possible values for fields that are annotated with Enum or Literal.
    """
    origin = get_origin(field_type)
    if origin is Literal:
        enum = list(get_args(field_type))
    elif isinstance(field_type, type) and issubclass(field_type, Enum):
        # Extract possible values from Enum
        enum = [member.value for member in field_type]
    else:
        return None

    arg_type = type(enum[0])
    return arg_type, enum

def get_origin_arg_type_and_args(field_type: Any) -> tuple[type|None, type, type|None, tuple|None]:
    """
    Extracts the origin, arg type, and args from a field type.
    """
    origin = get_origin(field_type)
    if origin in (None, Literal, Enum, UnionType, Union, list):
        args = get_args(field_type)
        if args:
            arg_type = type(args[0])
            # arg_type, arg_arg_type, arg_args = get_origin_arg_type_and_args(args[0])
            return origin, field_type, arg_type, args
        else:
            return origin, field_type, None, None
    return get_origin_arg_type_and_args(origin)


def is_type_and_subclass(annotation: Optional[type], _class_or_tuple: type|Tuple[type]) -> bool:
    """Checks that the annotation is a type before checking if it is a subclass of _class_or_tuple"""
    return isinstance(annotation, type) and issubclass(annotation, _class_or_tuple)


def is_base_model(annotation: Optional[type]) -> bool:
    """Checks if the annotation is a subclass of pydantic.BaseModel"""
    return is_type_and_subclass(annotation, BaseModel)



def get_field_and_item_origin(annotation: Optional[type]) -> dict:

    if not annotation or is_type_and_subclass(annotation, str):
        return {"type": str, "item_type": None, "args": None}
    # elif annotation in (bool, int, float) or is_type_and_subclass(annotation, (BaseModel, bool, int, float)):
    elif is_type_and_subclass(annotation, (BaseModel, bool, int, float)):
        return {"type": annotation, "item_type": None, "args": None}
    # elif isinstance(annotation, BaseModel):
    #     return {"type": type(annotation), "item_type": None, "args": None}
    elif is_type_and_subclass(annotation, Enum):
        enum = [member.value for member in annotation]
        anno_dict = get_field_and_item_origin(type(enum[0]))
        anno_dict["enum"] = enum
        return anno_dict
        # return {"type": type(annotation), "item_type": None, "args": None, "enum": enum}
    

    origin = get_origin(annotation)
    if origin is None:
        return {"type": annotation, "item_type": None, "args": None}
    elif origin is Literal:
        enum = get_args(annotation)
        anno_dict = get_field_and_item_origin(type(enum[0]))
        anno_dict["enum"] = enum
        return anno_dict
        # return {"type": anno_dict["type"], "item_type": anno_dict["type"], "args": None, "enum": enum}

    if origin in (Annotated, Union, UnionType) or isinstance(type(origin), _SpecialForm):
        arg = None
        args = get_args(annotation)
        for arg in args:
        # for arg in get_args(annotation):
            if arg is not None:
                break
        if arg is None:
            raise ValueError(f"Annotated type {origin} must have at least one non None arg")
        # return get_field_and_item_origin(arg)
        return get_field_and_item_origin(arg)
        # return arg_dict
        # return {"type": arg_dict["type"], "item_type": arg_dict["item_type"], "args": args}

    if is_type_and_subclass(origin, Sequence):
        arg = None
        args = get_args(annotation)
        for arg in args:
        # for arg in get_args(annotation):
            if arg is not None:
                break
        if arg is None:
            raise ValueError(f"Sequence type {origin} must have at least one non None arg")
        # return origin, get_field_and_item_origin(arg)
        return {"type": origin, "item_type": get_field_and_item_origin(arg), "args": args}

    return get_field_and_item_origin(origin)


def tool_parameter_from_anno_dict(
        anno_dict: dict,
        param_name: str,
        param_required: bool=True,
        param_description: Optional[str] = None,
        exclude_fields: Optional[list[str]] = None,
        ):

        field_type = anno_dict["type"]
        enum = anno_dict.get("enum")
        if is_type_and_subclass(field_type, BaseModel):
            return tool_parameter_from_pydantic(model=field_type, param_name=param_name, param_required=param_required, exclude_fields=exclude_fields)
        elif not field_type or issubclass(field_type, str):
            return StringToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)        
        elif is_type_and_subclass(field_type, bool):
            return BooleanToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)
        elif is_type_and_subclass(field_type, float):
            return NumberToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)
        elif is_type_and_subclass(field_type, int):
            return IntegerToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)
        elif is_type_and_subclass(field_type, Sequence):
            items = tool_parameter_from_anno_dict(
                anno_dict=anno_dict["item_type"], 
                param_name=param_name, 
                param_required=param_required,
                exclude_fields=exclude_fields
            )
            return ArrayToolParameter(name=param_name, description=param_description, required=param_required, enum=enum, items=items)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")


def tool_parameter_from_pydantic(
        model: Type[BaseModel]|BaseModel, 
        param_name: Optional[str]=None,
        param_required: bool=True,
        exclude_fields: Optional[list[str]] = None,     
        ) -> ObjectToolParameter:
    """
    Converts a Pydantic model into a ToolParameter object.
    """
    if isinstance(model, BaseModel):
        model = model.__class__
    
    properties = []
    for field_name, field in model.model_fields.items():
        if exclude_fields and field_name in exclude_fields:
            continue

        anno_dict = get_field_and_item_origin(field.annotation)
        field_description = field.description
        field_required = field.is_required()
        properties.append(tool_parameter_from_anno_dict(
            anno_dict=anno_dict,
            param_name=field_name,
            param_required=field_required,
            param_description=field_description,
            exclude_fields=exclude_fields
        ))
        
    name = param_name or model.__name__
    description = model.__doc__
    return ObjectToolParameter(name=name, description=description, properties=properties, required=param_required)



def tool_from_pydantic_model(
        model: Type[BaseModel]|BaseModel, 
        tool_name: Optional[str] = None,
        description: Optional[str] = None,
        exclude_fields: Optional[list[str]] = None,
        ) -> Tool:
    
    tool_type = "function"
    if isinstance(model, BaseModel):
        model = model.__class__

    parameters = tool_parameter_from_pydantic(model, exclude_fields=exclude_fields)
    if isinstance(parameters, AnyOfToolParameter):
        raise ValueError("Root parameter cannot be anyOf: See: https://platform.openai.com/docs/guides/structured-outputs/root-objects-must-not-be-anyof")
    if not isinstance(parameters, ObjectToolParameter):
        raise ValueError("Root parameter must be an object")
    
    name = tool_name or f"return_{parameters.name}" 
    description = description or parameters.description or "Return structured output"
    parameters.name = "parameters"
    parameters.description = None

    return Tool(
        name=name,
        description=description,
        parameters=parameters,
        # callable=model
    )