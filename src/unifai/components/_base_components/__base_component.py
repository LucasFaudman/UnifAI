from typing import Type, Optional, Sequence, Any, Union, Literal, TypeVar, ClassVar, Callable, Iterator, Iterable, Generator, Generic, Type
from abc import ABC

from ...exceptions import UnifAIError, UnknownUnifAIError
from ...configs._base_configs import ComponentConfig
from ...types.annotations import ComponentType, ProviderName, ComponentName

class AbstractBaseComponent(ABC):
    _is_abstract = True
    _abstract_methods = tuple() # Base abstract methods that must be implemented by the subclass 
    _abstract_method_suffixes = tuple() # Variants that can be implemented by the subclass in place of the base abstract methods
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # _is_abstract=True must explicitly set by the subclass (in __dict__) if the subclass is abstract, 
        # otherwise it is assumed to be concrete and will be validated
        if cls.__dict__.get('_is_abstract'):
            return # Subclass is abstract, allow abstract methods to be unimplemented
        if not (methods := getattr(cls, "_abstract_methods", None)):
            return # No abstract methods to check for

        suffixes = getattr(cls, "_abstract_method_suffixes", ())
        _super = super(cls)
        for method in methods:
            # Check if base method is implemented by the subclass
            if (_implemented_at_least1 := getattr(cls, method, None) is not getattr(_super, method, None)):
                continue
            
            # Check if at least one method+suffix variation is implemented by the subclass
            _methods = [f"{method}_{suffix}" for suffix in suffixes]            
            for _method in _methods:
                if getattr(cls, _method, None) is not getattr(_super, _method, None):
                    _implemented_at_least1 = True
                    break
            
            # Raise error if none of the methods are implemented
            if not _implemented_at_least1:
                missing = ", ".join([method] + _methods[:-1]) + (f" or {_methods[-1]}" if len(_methods) > 1 else "")
                raise NotImplementedError(f"Can't instantiate abstract class {cls.__name__} without at least one of {missing} implemented")


YieldT = TypeVar("YieldT")
ReturnT = TypeVar("ReturnT")
ConfigT = TypeVar('ConfigT', bound=ComponentConfig)

class UnifAIComponent(AbstractBaseComponent, Generic[ConfigT]):
    component_type = "base_component"
    provider = "base"
    config_class: Type[ConfigT]
    can_get_components = False
    setup_on_init = True

    _do_not_convert = (
        UnifAIError,
        SyntaxError,
        NameError,
    )
    _default_exception: Type[UnifAIError] = UnknownUnifAIError


    def __init__(
            self, 
            config: Optional[ConfigT] = None,
            # _get_component: Optional[Callable[..., Any]] = None,
            **init_kwargs
            ):
        self.config: ConfigT = config or self.config_class(provider=self.provider)
        self.component_id = f"{self.component_type}:{self.provider}:{self.config.name}:{id(self)}"
        self.name = self.config.name
        # Args used to initialize the component. Processed first by __init__ and/or _setup, 
        # then passed through to the underlying object being initialized if applicable.
        self.init_kwargs = init_kwargs
        
        # Exception handlers to allow components to handle exceptions and retry functions
        # Handler takes the following arguments:
        #   - e: (Exception) The Exception that was raised converted to a UnifAIError unless it is a do_not_convert exception
        #   - traceback: (dict) The component, function name, args, and kwargs that caused the exception
        # Handler returns a tuple with the following values:
        #   - retry: (bool) True if the function should be retried with new args and kwargs, False to raise the exception (unless handled by another handler due to inheritance)
        #   - new_args: (tuple) New args to use if retry is True
        #   - new_kwargs: (dict) New kwargs to use if retry is True
        self.exception_handlers: dict[Type[Exception], Callable[[Exception, dict], tuple[bool, tuple, dict]]] = {}
        
        # Optional reference to parent to allow components to access other components
        self.__get_component: Optional[Callable[..., Any]] = init_kwargs.pop("_get_component", None)        

        if self.setup_on_init:
            self._setup()

    def _setup(self) -> None:
        """
        Runs after init to set up the component once self.config and self.init_kwargs are set. 
        Used to allow subclasses to setup on init or defer setup until later when component is used 
        to avoid uneccessary work for unused components and/or prevent circular imports, and/or 
        prevent need to override init and call super().__init__ to properly set up the component. (handle config & pop _get_component)
        """
                
    def _get_component(
        self,
        component_type: ComponentType,
        provider_config_or_name: ProviderName | ComponentConfig | tuple[ProviderName, ComponentName],
        **init_kwargs: dict,
    ) -> Any:
        if not self.__get_component:
            raise NotImplementedError(f"{self.__class__.__name__} does not support getting components")
        return self.__get_component(component_type, provider_config_or_name, init_kwargs)
                        
    # Convert Exceptions from Client Exception Types to UnifAI Exceptions for easier handling
    def _convert_exception(self, exception: Exception) -> UnifAIError:
        return self._default_exception(message=str(exception), original_exception=exception)
        
    def _handle_exception(self, e: Exception, func: Callable[..., ReturnT], *args, **kwargs) -> ReturnT:
        traceback = {
            "component": self, 
            "func_name": func.__name__, 
            "func_args": args, 
            "func_kwargs": kwargs
        }
        if isinstance(e, UnifAIError):
            e.add_traceback(**traceback) # add traceback info to existing UnifAIError as it propagates
        elif not isinstance(e, self._do_not_convert):
            e = self._convert_exception(e) # convert exception to UnifAIError and add traceback info
            e.add_traceback(**traceback)
        # Check if exception has a handler registered, if so call it to determine wether to retry the function
        # with new args and kwargs provided by the handler, otherwise raise the exception
        for exception_type, handler in self.exception_handlers.items():
            if isinstance(e, exception_type):
                retry, new_args, new_kwargs = handler(e, traceback)
                if retry:
                    return func(*new_args, **new_kwargs)
        raise e from e
    
    def _run_func(self, func: Callable[..., ReturnT], *args, **kwargs) -> ReturnT:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return self._handle_exception(e, func, *args, **kwargs)

    def _run_generator(self, func: Callable[..., Generator[YieldT, None, ReturnT]], *args, **kwargs) ->  Generator[YieldT, None, ReturnT]:
        try:
            rval = yield from func(*args, **kwargs)
        except Exception as e:
            rval = yield from self._handle_exception(e, func, *args, **kwargs)
        return rval


def convert_exceptions(func: Callable[..., ReturnT]) -> Callable[..., ReturnT]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> ReturnT:
        return instance._run_func(func, instance, *args, **kwargs)
    return wrapper

def convert_exceptions_generator(func: Callable[..., Generator[YieldT, None, ReturnT]]) -> Callable[..., Generator[YieldT, None, ReturnT]]:
    def wrapper(instance: UnifAIComponent, *args, **kwargs) -> Generator[YieldT, None, ReturnT]:
        return instance._run_generator(func, instance, *args, **kwargs)
    return wrapper