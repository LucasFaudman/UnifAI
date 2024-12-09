from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Callable, Iterator, Iterable, Generator, Generic
from ...exceptions import UnifAIError, UnknownUnifAIError
from ...configs._base_configs import ComponentConfig
from ...types.annotations import ComponentType, ProviderName, ComponentName

YieldT = TypeVar("YieldT")
ReturnT = TypeVar("ReturnT")
ConfigT = TypeVar('ConfigT', bound=ComponentConfig)

class UnifAIComponent(Generic[ConfigT]):
    _do_not_convert = (
        UnifAIError,
        SyntaxError,
        NameError,
        # AttributeError,
        # TypeError,
        # ValueError,
        # IndexError,
        # KeyError,
        # ImportError,
        # ModuleNotFoundError,
        # NameError,
        # FileNotFoundError,
        # OSError,
        # ZeroDivisionError,
        # RuntimeError,
        # StopIteration,
        # AssertionError
    ) 
    component_type = "base_component"
    provider = "base"
    config_class: Type[ConfigT]
    can_get_components = False

    def __init__(
            self, 
            config: Optional[ConfigT] = None,
            # _get_component: Optional[Callable[..., Any]] = None,
            **init_kwargs
            ):
        self.config: ConfigT = config or self.config_class(provider=self.provider)
        self.component_id = f"{self.component_type}:{self.provider}:{self.config.name}:{id(self)}"
        self.__get_component: Optional[Callable[..., Any]] = init_kwargs.pop("_get_component", None)
        self.init_kwargs = init_kwargs
        self._setup()

    def _setup(self) -> None:
        """
        Runs after init to set up the component once self.config and self.init_kwargs are set. 
        Use to avoid avoid needed to override init and call super() to properly set up the component. (handle config & pop _get_component)
        """
        
    @property
    def name(self) -> str:
        return self.config.name
        
    def _get_component(
        self,
        component_type: ComponentType,
        provider_config_or_name: ProviderName | ComponentConfig | tuple[ProviderName, ComponentName],
        **init_kwargs: dict,
    ) -> Any:
        if not self.__get_component:
            raise NotImplementedError(f"{self.component_type} does not support getting components")
        return self.__get_component(component_type, provider_config_or_name, init_kwargs)
            
    # Convert Exceptions from Client Exception Types to UnifAI Exceptions for easier handling
    def convert_exception(self, exception: Exception) -> UnifAIError:
        return UnknownUnifAIError(message=str(exception), original_exception=exception)
        
    def run_func_convert_exceptions(self, func: Callable[..., ReturnT], *args, **kwargs) -> ReturnT:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, UnifAIError):
                e.add_traceback(component=self, func_name=func.__name__, func_args=args, func_kwargs=kwargs)
            if isinstance(e, self._do_not_convert):
                raise e from e
            unifai_exception = self.convert_exception(e)
            unifai_exception.add_traceback(component=self, func_name=func.__name__, func_args=args, func_kwargs=kwargs)
            raise unifai_exception from e

    def run_func_convert_exceptions_generator(self, func: Callable[..., Generator[YieldT, None, ReturnT]], *args, **kwargs) ->  Generator[YieldT, None, ReturnT]:
        try:
            rval = yield from func(*args, **kwargs)
            return rval
        except Exception as e:
            if isinstance(e, UnifAIError):
                e.add_traceback(component=self, func_name=func.__name__, func_args=args, func_kwargs=kwargs)
            if isinstance(e, self._do_not_convert):
                raise e from e
            unifai_exception = self.convert_exception(e)
            unifai_exception.add_traceback(component=self, func_name=func.__name__, func_args=args, func_kwargs=kwargs)
            raise unifai_exception from e


def convert_exceptions(func: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> ReturnT:
        return instance.run_func_convert_exceptions(func, instance, *args, **kwargs)
    return wrapper

def convert_exceptions_generator(func: Callable[..., Generator[YieldT, None, ReturnT]]) -> Callable[..., Generator[YieldT, None, ReturnT]]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> Generator[YieldT, None, ReturnT]:
        return instance.run_func_convert_exceptions_generator(func, instance, *args, **kwargs)
    return wrapper 