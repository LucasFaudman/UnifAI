from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, Callable, Iterator, Iterable, Generator
from ..exceptions import UnifAIError

yieldT = TypeVar("yieldT")
returnT = TypeVar("returnT")

class UnifAIComponent:
    do_not_convert = (
        UnifAIError,
        AttributeError,
        TypeError,
        ValueError,
        IndexError,
        KeyError,
        ImportError,
        ModuleNotFoundError,
        NameError,
        FileNotFoundError,
        OSError,
        ZeroDivisionError,
        RuntimeError,
        StopIteration,
        AssertionError
    )        

    # Convert Exceptions from Client Exception Types to UnifAI Exceptions for easier handling
    def convert_exception(self, exception: Exception) -> UnifAIError:
        return UnifAIError(message=str(exception), original_exception=exception)


    def run_func_convert_exceptions(self, func: Callable[..., returnT], *args, **kwargs) -> returnT:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, self.do_not_convert):
                raise e
            raise self.convert_exception(e) from e


    def run_func_convert_exceptions_generator(self, func: Callable[..., Generator[yieldT, None, returnT]], *args, **kwargs) ->  Generator[yieldT, None, returnT]:
        try:
            rval = yield from func(*args, **kwargs)
            return rval
        except Exception as e:
            if isinstance(e, self.do_not_convert):
                raise e
            raise self.convert_exception(e) from e 


def convert_exceptions(func: Callable[..., returnT]) -> Callable[..., returnT]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> returnT:
        return instance.run_func_convert_exceptions(func, instance, *args, **kwargs)
    return wrapper


def convert_exceptions_generator(func: Callable[..., Generator[yieldT, None, returnT]]) -> Callable[..., Generator[yieldT, None, returnT]]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> Generator[yieldT, None, returnT]:
        return instance.run_func_convert_exceptions_generator(func, instance, *args, **kwargs)
    return wrapper 