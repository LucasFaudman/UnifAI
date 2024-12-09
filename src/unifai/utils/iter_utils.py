from typing import Iterable, Iterator, Optional, TypeVar, Any, overload, Tuple, Union, List, Literal, Type, Generic, Unpack
from itertools import islice, zip_longest

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")

# def chunk_iterable(iterable: Iterable[T], chunk_size: Optional[int]) -> Iterable[Iterable[T]]:
#     if chunk_size is None:
#         yield iterable
#         return
#     iterator = iter(iterable)
#     while chunk := list(islice(iterator, chunk_size)):
#         yield chunk

def chunk_iterable(iterable: Iterable[T], chunk_size: Optional[int]) -> Iterable[tuple[T, ...]]:
    iterator = iter(iterable)
    while chunk := tuple(islice(iterator, chunk_size)):
        yield chunk

def _next(iterable: Iterable[T]) -> T:
    return next(iter(iterable))

def zippable(*iterables: Optional[Iterable[Any]]) -> Iterator[Iterable[Any]]:
    return (iterable or () for iterable in iterables)

def as_list(arg: Any|list[Any]) -> list[Any]:
    return arg if isinstance(arg, list) else [arg] if arg is not None else []

def as_lists(*args: Any|list[Any]) -> Iterator[list[Any]]:
    return map(as_list, args)

def limit_offset_slice(
        limit: Optional[int] = None, 
        offset: Optional[int] = None, # woop woop
    ) -> slice:
    if offset is not None:
        if limit is None:
            return slice(offset, None)
        return slice(offset, limit + offset)
    return slice(limit)