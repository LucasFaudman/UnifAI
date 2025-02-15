from typing import get_type_hints, TypeGuard, TypeVarTuple, Callable, ParamSpec, TypeVar, Dict, Any, LiteralString, Unpack, Concatenate, TypeVarTuple, ChainMap, cast, Generic, Type, Union, Literal, Optional, ClassVar, TypeVar, Type, List, Tuple, TypeVar, Type, Any, Protocol, ParamSpec, TypeVar, Self, Generic, Union, Literal, Optional, Type, TypeVar, Callable, runtime_checkable, overload
from pydantic import BaseModel as PydanticBaseModel # Any Pydantic BaseModel Subclass not just unifai BaseModel

T = TypeVar('T')
P = ParamSpec('P')
T2 = TypeVar('T2')
P2 = ParamSpec('P2')
Ts = TypeVarTuple('Ts')

def is_type_and_subclass(annotation: Any, _class_or_tuple: Type[T]|Tuple[Type[T],...]) -> TypeGuard[Type[T]]:
    """Prevents raising TypeError: issubclass() arg 1 must be a class"""
    return isinstance(annotation, type) and issubclass(annotation, _class_or_tuple)

def is_base_model(annotation: Any) -> TypeGuard[Type[PydanticBaseModel]]:
    """Check if annotation is a Pydantic BaseModel or subclass including all unifai BaseModel subclasses"""
    return is_type_and_subclass(annotation, PydanticBaseModel)

def copy_signature_from(_origin: Callable[P, T]) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    def decorator(target: Callable[..., Any]) -> Callable[P, T]:
        return cast(Callable[P, T], target)    
    return decorator

def copy_paramspec_from(_origin: Callable[P, Any]) -> Callable[[Callable[..., T]], Callable[P, T]]:
    def decorator(target: Callable[..., T]) -> Callable[P, T]:
        return cast(Callable[P, T], target)    
    return decorator

def concat_signature_from(
    _origin: Callable[P, Any],
    _prepend: Type[T],
    _return: Type[T2]
) -> Callable[[Callable[..., Any]], Callable[Concatenate[T, P], T2]]:
    def decorator(target: Callable[..., Any]) -> Callable[Concatenate[T, P], T2]:
        return cast(Callable[Concatenate[T, P], T2], target)    
    return decorator

# def copy_paramspec_from(
#     _origin: Callable[P, Any],
#     _return: Type[T] | Callable[..., T]
# ) -> Callable[[Callable[..., Any]], Callable[P, T]]:
#     def decorator(target: Callable[..., Any]) -> Callable[P, T]:
#         return cast(Callable[P, T], target)
#     return decorator