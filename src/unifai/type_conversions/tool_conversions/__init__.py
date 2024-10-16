from typing import Optional, Type, TypeVar, Any, Union, Sequence, Mapping, List, Tuple, Annotated, Callable, _SpecialForm
from types import UnionType

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
from pydantic import BaseModel
from enum import Enum
from typing import Literal, get_args, get_origin

T = TypeVar("T")


def is_type_and_subclass(annotation: Any, _class_or_tuple: type|Tuple[type]) -> bool:
    """Checks that the annotation is a type before checking if it is a subclass of _class_or_tuple"""
    return isinstance(annotation, type) and issubclass(annotation, _class_or_tuple)


def unpack_annotation(annotation: Optional[type]) -> dict:

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
        anno_dict = unpack_annotation(type(enum[0])) # unpacked type of first enum value
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
        return unpack_annotation(arg) # unpacked type of first non-null arg

    if is_type_and_subclass(origin, Sequence):
        arg = None
        for arg in get_args(annotation):
            if arg is not None: break # Get the first non-null arg from Sequence
        if arg is None:
            raise ValueError(f"Sequence type annotations must have at least one non-null arg. Got: {annotation}")        
        return {"type": origin, "item_type": unpack_annotation(arg)} # type=Sequence, item_type=(unpacked type of first non-null arg)

    # Recursively unpack nested annotations until first concrete type, (and optional item_type) is found
    return unpack_annotation(origin)