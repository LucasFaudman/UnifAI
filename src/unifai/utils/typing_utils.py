from typing import get_type_hints, TypeVarTuple, Callable, ParamSpec, TypeVar, Dict, Any, LiteralString, Unpack, Concatenate, TypeVarTuple, ChainMap, cast, Generic, Type, Union, Literal, Optional, ClassVar, TypeVar, Type, List, Tuple, TypeVar, Type, Any, Protocol, ParamSpec, TypeVar, Self, Generic, Union, Literal, Optional, Type, TypeVar, Callable, runtime_checkable, overload

T = TypeVar('T')
P = ParamSpec('P')
T2 = TypeVar('T2')
P2 = ParamSpec('P2')
Ts = TypeVarTuple('Ts')

def copy_signature_from(_origin: Callable[P, T]) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    def decorator(target: Callable[..., Any]) -> Callable[P, T]:
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

def signature_from_config(
    _config_type: Callable[P, Any],
    _return_type: Type[T]
) -> Callable[[Callable[..., Any]], Callable[P, T]]:
    def decorator(target: Callable[..., Any]) -> Callable[P, T]:
        return cast(Callable[P, T], target)    
    return decorator