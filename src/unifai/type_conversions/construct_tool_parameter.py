from typing import Optional, Type, TypeVar, Any, Union, Collection, Sequence, Mapping, List, Tuple, Annotated, Callable, _SpecialForm, Literal, get_args, get_origin
from types import UnionType
from enum import Enum
from pydantic import BaseModel

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


def is_type_and_subclass(annotation: Any, _class_or_tuple: type|Tuple[type]) -> bool:
    """Checks that the annotation is a type before checking if it is a subclass of _class_or_tuple"""
    return isinstance(annotation, type) and issubclass(annotation, _class_or_tuple)

def is_base_model(annotation: Any) -> bool:
    return isinstance(annotation, BaseModel) or is_type_and_subclass(annotation, BaseModel)

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

    if is_type_and_subclass(origin, Collection):
        arg = None
        for arg in get_args(annotation):
            if arg is not None: break # Get the first non-null arg from Collection
        if arg is None:
            raise ValueError(f"Collection type annotations must have at least one non-null arg. Got: {annotation}")        
        return {"type": origin, "items": resolve_annotation(arg)} # type=Collection, items=(resolved type of first non-null arg)

    # Recursively resolve nested annotations until first concrete type, (and optional items) is found
    return resolve_annotation(origin)


def construct_tool_parameter(
        param_dict: dict, 
        param_name: Optional[str]= None,
        param_description: Optional[str] = None,
        exclude: Collection[str] = (),
        ) -> ToolParameter:
    
    if (ref := param_dict.get('$ref')) is not None:
        return RefToolParameter(name=param_name, ref=ref)
    
    if (anyof_param_dicts := param_dict.get('anyOf')) is not None:
        anyOf = [
            construct_tool_parameter(param_dict=anyof_param_dict, param_name=param_name)
            for anyof_param_dict in anyof_param_dicts
        ]
        return AnyOfToolParameter(name=param_name, anyOf=anyOf)
    
    # Everything but AnyOfToolParameter and RefToolParameter should have a 'type' key
    param_type = param_dict['type']    
        
    if is_type_and_subclass(param_type, BaseModel):
        properties = {}
        for field_name, field in param_type.model_fields.items():
            if exclude and field_name in exclude:
                continue

            anno_dict = resolve_annotation(field.annotation)
            field_description = field.description
            properties[field_name] = construct_tool_parameter(
                param_dict=anno_dict,
                param_name=field_name,
                param_description=field_description,
                exclude=exclude
            )            
        name = param_name or param_type.__name__
        description = param_type.__doc__
        return ObjectToolParameter(name=name, description=description, properties=properties)        


    param_name = param_dict.get('name', param_name)
    param_description = param_dict.get('description', param_description)
    param_enum = param_dict.get('enum')

    if param_type == 'string' or param_type is None or is_type_and_subclass(param_type, str):
        return StringToolParameter(name=param_name, description=param_description, enum=param_enum)
    if param_type == 'boolean' or is_type_and_subclass(param_type, bool):
        return BooleanToolParameter(name=param_name, description=param_description, enum=param_enum)
    if param_type == 'number' or is_type_and_subclass(param_type, float):
        return NumberToolParameter(name=param_name, description=param_description, enum=param_enum)
    if param_type == 'integer' or is_type_and_subclass(param_type, int):
        return IntegerToolParameter(name=param_name, description=param_description, enum=param_enum)
    if param_type == 'null':
        return NullToolParameter(name=param_name, description=param_description, enum=param_enum)

    if param_type == 'object':
        if (param_properties := param_dict.get('properties')) is None:
            raise ValueError("Object parameters must have a 'properties' key.")
        properties = {
            prop_name: construct_tool_parameter(param_dict=prop_dict, param_name=prop_name) 
            for prop_name, prop_dict in param_properties.items() if prop_name not in exclude
        }
        additionalProperties = param_dict.get('additionalProperties', False)
        if def_dicts := param_dict.get('$defs'): 
            defs = {
                def_name: construct_tool_parameter(param_dict=def_dict)
                for def_name, def_dict in def_dicts.items() if def_name not in exclude
            }
        else:
            defs = None

        return ObjectToolParameter(name=param_name, description=param_description, enum=param_enum, 
                                   properties=properties, additionalProperties=additionalProperties,
                                   defs=defs)
    
    if param_type == 'array' or is_type_and_subclass(param_type, Collection):
        if not (param_items := param_dict.get('items')):
            raise ValueError("Array parameters must have an 'items' key.")
        
        items = construct_tool_parameter(param_dict=param_items)
        return ArrayToolParameter(name=param_name, description=param_description, 
                                  enum=param_enum,
                                  items=items)    

    
    raise ValueError(f"Invalid parameter type: {param_type}")





