from typing import Optional, Type, TypeVar, Any, Union, Sequence, Mapping, List, Tuple, Annotated, Callable, _SpecialForm, Literal, get_args, get_origin
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


T = TypeVar("T")


def is_type_and_subclass(annotation: Any, _class_or_tuple: type|Tuple[type]) -> bool:
    """Checks that the annotation is a type before checking if it is a subclass of _class_or_tuple"""
    return isinstance(annotation, type) and issubclass(annotation, _class_or_tuple)


def resolve_annotation(annotation: Optional[type]) -> dict:

    if not annotation or is_type_and_subclass(annotation, str):
        return {"type": str} # reached a concrete type (str) or no annotation so default to str
    elif is_type_and_subclass(annotation, (BaseModel, bool, int, float)):
        return {"type": annotation} # reached a concrete type (BaseModel, bool, int, float)

    is_enum = False
    if ((origin := get_origin(annotation)) is Literal
        or (is_enum := is_type_and_subclass(annotation, Enum))
        ):
        # Get enum values from Enum and/or Literal annotations
        if is_enum:
            enum = [member.value for member in annotation] # Enum
        else:
            enum = get_args(annotation) # Literal
        anno_dict = resolve_annotation(type(enum[0])) # resolved type of first enum value
        anno_dict["enum"] = enum
        return anno_dict

    if origin is None:
        return {"type": annotation} # reached a concrete type

    if origin in (Annotated, Union, UnionType) or isinstance(type(origin), _SpecialForm):
        arg = None
        for arg in get_args(annotation):
            if arg is not None: break # Get the first non-null arg from Annotated, UnionType, etc
        if arg is None:
            raise ValueError(f"_SpecialForm (Union, Annotated, etc) annotations must have at least one non-null arg. Got: {annotation}")        
        return resolve_annotation(arg) # resolved type of first non-null arg

    if is_type_and_subclass(origin, Sequence):
        arg = None
        for arg in get_args(annotation):
            if arg is not None: break # Get the first non-null arg from Sequence
        if arg is None:
            raise ValueError(f"Sequence type annotations must have at least one non-null arg. Got: {annotation}")        
        return {"type": origin, "items": resolve_annotation(arg)} # type=Sequence, items=(resolved type of first non-null arg)

    # Recursively unpack nested annotations until first concrete type, (and optional items) is found
    return resolve_annotation(origin)


def tool_parameter_from_anno_dict(
        anno_dict: dict,
        param_name: str,
        param_required: bool=True,
        param_description: Optional[str] = None,
        exclude: Optional[list[str]] = None,
        ):

        field_type = anno_dict["type"]
        enum = anno_dict.get("enum")
        if is_type_and_subclass(field_type, BaseModel):
            return tool_parameter_from_pydantic_model(model=field_type, param_name=param_name, param_required=param_required, exclude=exclude)
        elif not field_type or is_type_and_subclass(field_type, str):
            return StringToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)        
        elif is_type_and_subclass(field_type, bool):
            return BooleanToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)
        elif is_type_and_subclass(field_type, int):
            return IntegerToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)        
        elif is_type_and_subclass(field_type, float):
            return NumberToolParameter(name=param_name, description=param_description, required=param_required, enum=enum)
        elif is_type_and_subclass(field_type, Sequence):
            items = tool_parameter_from_anno_dict(
                anno_dict=anno_dict["items"], 
                param_name=f"{param_name}_item",
                param_required=param_required,
                exclude=exclude
            )
            return ArrayToolParameter(name=param_name, description=param_description, required=param_required, enum=enum, items=items)
        else:
            raise ValueError(f"Unsupported field type: {field_type}")


def tool_parameter_from_pydantic_model(
        model: Type[BaseModel]|BaseModel, 
        param_name: Optional[str]=None,
        param_required: bool=True,
        exclude: Optional[list[str]] = None,     
        ) -> ObjectToolParameter:
    """
    Converts a Pydantic model into a ToolParameter object.
    """
    if isinstance(model, BaseModel):
        model = model.__class__
    
    properties = []
    for field_name, field in model.model_fields.items():
        if exclude and field_name in exclude:
            continue

        anno_dict = resolve_annotation(field.annotation)
        field_description = field.description
        field_required = field.is_required()
        properties.append(tool_parameter_from_anno_dict(
            anno_dict=anno_dict,
            param_name=field_name,
            param_required=field_required,
            param_description=field_description,
            exclude=exclude
        ))
        
    name = param_name or model.__name__
    description = model.__doc__
    return ObjectToolParameter(name=name, description=description, properties=properties, required=param_required)


def tool_from_pydantic_model(
        model: Type[BaseModel]|BaseModel, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        type: str = "function",
        strict: bool = True,
        exclude: Optional[list[str]] = None,
        ) -> Tool:
    
    if isinstance(model, BaseModel):
        model = model.__class__

    parameters = tool_parameter_from_pydantic_model(model, exclude=exclude)
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
tool_from_model = tool_from_pydantic_model