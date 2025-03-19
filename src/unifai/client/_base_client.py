from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload, NoReturn

from pathlib import Path
from os import getenv

from ..types.annotations import ComponentType, ComponentName, ProviderName
from ..components._component_importer import import_component
from ..components._globals import COMPONENT_TYPES, PROVIDERS, DEFAULT_PROVIDERS
from ..components._base_components.__base_component import UnifAIComponent
from ..configs._base_configs import ProviderConfig, ComponentConfig
from ..configs import UnifAIConfig, COMPONENT_CONFIGS
from ..utils import _next, combine_dicts, recursive_clear, clean_locals

class BaseClient:
    def __init__(
        self,
        config: Optional[UnifAIConfig|dict[str, Any]|str|Path] = None,
        api_keys: Optional[dict[ProviderName, str]] = None,
        **kwargs
    ):
        # self._config_classes: dict[ComponentType, Type[ComponentConfig]] = COMPONENT_CONFIGS.copy()
        self._config_classes: dict[ComponentType, dict[ProviderName, Type[ComponentConfig]]] = {
            component_type: {"default" : config_class} for component_type, config_class in COMPONENT_CONFIGS.items()
        }      
        self._provider_configs: dict[ProviderName, ProviderConfig] = {} 
        self._default_providers: dict[ComponentType, ProviderName] = {}       
        self._component_configs: dict[ComponentType, dict[ProviderName, dict[ComponentName, ComponentConfig]]] = {}
        # Initialize component_types with empty dict for each built-in component_type.
        # This allows for registering new components for each component type.
        # and for checking if a component_type is valid. (builtin or registered) (_check_component_type_is_valid)
        self._component_types: dict[ComponentType, dict[ProviderName, Type[Any]|Callable[..., Any]]] = {component_type: {} for component_type in COMPONENT_TYPES}
        self._components: dict[ComponentType, dict[ProviderName, dict[ComponentName, Any]]] = {}
        # Update instance with config and api_keys
        self.configure(config, api_keys, **kwargs)

    def configure(
        self,
        config: Optional[UnifAIConfig|dict[str, Any]|str|Path] = None,
        api_keys: Optional[dict[ProviderName, str]] = None,
        **kwargs
    ) -> None:
        # Initialize config from input or create new
        if config is not None:
            if isinstance(config, str):
                config = Path(config)
            if isinstance(config, Path):
                config = UnifAIConfig.model_validate_json(config.read_bytes())
            elif isinstance(config, dict):
                config = UnifAIConfig.model_validate(config)
            elif isinstance(config, UnifAIConfig):
                config = config
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
            if kwargs:
                config = config.model_copy(update=kwargs)
        else:
            config = UnifAIConfig(**kwargs)

        # Update provider configs
        if config.provider_configs:
            self.register_provider_configs(*config.provider_configs)
                        
        # Update default providers
        if config.default_providers:
            self.update_default_providers(config.default_providers)
                
        # Update API keys from config then args.
        # API key priority: args > config > env
        for provider, api_key in combine_dicts(config.api_keys, api_keys).items():
            if not api_key:
                continue
            self.set_api_key(provider, api_key)
        
        # Update component configs
        if config.component_configs:
            self.register_component_configs(*config.component_configs)
        
        # Update component types
        self.config = config

    # Provider level Configurations affect all provider components regardless of component_type ie OpenAILLM, OpenAIEmbedder, etc. 
    def _register_provider_config(self, provider_config: ProviderConfig) -> None:
        provider = provider_config.provider
        # check for env api key for all pre-registered providers
        if provider_config.api_key is None:
            provider_config.api_key = self._check_env_api_key(provider)
        self._provider_configs[provider] = provider_config

    def register_provider_configs(self, *provider_configs: ProviderConfig) -> None:
        for provider_config in provider_configs:
            self._register_provider_config(provider_config)

    def get_provider_config(self, provider: ProviderName) -> ProviderConfig:
        if not (provider_config := self._provider_configs.get(provider)):
            # Create new provider config if it does not exist
            provider_config = ProviderConfig(provider=provider, api_key=self._check_env_api_key(provider))
            self._provider_configs[provider] = provider_config
        return provider_config

    # API keys are provider specific and affect all components of that provider
    def _check_env_api_key(self, provider: ProviderName) -> str | None:
        return getenv(f"{provider.upper()}_API_KEY")

    def set_api_key(self, provider: ProviderName, api_key: str) -> None:
        if provider_config := self._provider_configs.get(provider): 
            provider_config.api_key = api_key
        else:
            # Create new provider config if it does not exist
            self._provider_configs[provider] = ProviderConfig(provider=provider, api_key=api_key)

    def update_api_keys(self, api_keys: dict[ProviderName, str]) -> None:
        for provider, api_key in api_keys.items():
            self.set_api_key(provider, api_key)           

    def get_api_key(self, provider: ProviderName) -> str | None:
        return self.get_provider_config(provider).api_key
    
    def register_config_class(
            self,
            config_class: Type[ComponentConfig],
            component_type: Optional[ComponentType] = None,
            provider: ProviderName = "default",
            override_if_exists: bool = False,
    ):
        if (component_type := component_type or config_class.component_type) == "component":
            raise ValueError("component_type set as a ClassVar of the config_class or passed as an argument. Cannot be 'component' as this is the base class for all component configs")
        
        if not (config_classes_of_type := self._config_classes.get(component_type)):
            config_classes_of_type = self._config_classes[component_type] = {}

        if not override_if_exists and provider in config_classes_of_type:
            return
            # raise ValueError(f"Config class for {component_type} and {provider} already exists. Use override_if_exists=True to override.")
        config_classes_of_type[provider] = config_class

    def get_config_class(self, component_type: ComponentType, provider: ProviderName = "default") -> Type[ComponentConfig]:
        if not (config_classes_of_type := self._config_classes.get(component_type)):
            raise ValueError(f"No config classes found for component_type: {component_type}")
        if provider_config_class := config_classes_of_type.get(provider):
            return provider_config_class # Use a provider specific config class if it exists
        return config_classes_of_type["default"] # Use the default config class if it exists
        
    # Component level Configurations affect only the component of that provider and component_type with the same name (default name is "default") 
    # This is to allow for multiple components of the same provider and component_type with different configurations.
    def _register_component_config(self, component_config: ComponentConfig) -> None:
        component_type = component_config.component_type
        provider = component_config.provider
        component_name = component_config.name
        if component_type not in self._component_configs:
            self._component_configs[component_type] = {provider: {}}
        elif provider not in self._component_configs[component_type]:
            self._component_configs[component_type][provider] = {}

        self._component_configs[component_type][provider][component_name] = component_config

    def register_component_configs(self, *component_configs: ComponentConfig) -> None:
        for component_config in component_configs:
            self._register_component_config(component_config)

    def get_component_config(
            self, 
            component_type: ComponentType, 
            provider: ProviderName, 
            component_name: ComponentName = "default"
        ) -> ComponentConfig:
        if ((configs_of_type := self._component_configs.get(component_type))
             and (configs_of_provider := configs_of_type.get(provider))
             and (component_config_with_name := configs_of_provider.get(component_name))
             ):
            return component_config_with_name

        config_class = self.get_config_class(component_type, provider)
        config = config_class(provider=provider, name=component_name)
        return config
    
    def _config_from_locals(
            self, 
            component_type: ComponentType,
            provider: ProviderName,
            _locals: dict[str, Any],
            exclude: Collection[str] = ('self', 'cls', 'cache', 'create_if_not_exists', 'reuse_if_exists', 'override_config_if_exists'),
            kwargs_key: str = "kwargs"
            ) -> Any:
        config_class = self.get_config_class(component_type, provider)
        config_kwargs = clean_locals(_locals, exclude, kwargs_key)
        config = config_class(**config_kwargs)
        return config

    # Register a new Component Subclasses with the Client
    def register_component(
        self,
        component_class: Type[UnifAIComponent]|Type[Any]|Callable[..., Any],
        *,
        config_class: Optional[Type[ComponentConfig]] = None,
        component_type: Optional[ComponentType] = None,
        provider: Optional[ProviderName] = None,
        can_get_components: Optional[bool] = None
    ) -> None:
        
        if not (component_type := component_type or getattr(component_class, "component_type", None)):
            raise ValueError(f"component_class must have a component_type attribute or pass component_type as an argument")
        if not (provider := provider or getattr(component_class, "provider", None)):
            raise ValueError(f"component_class must have a provider attribute or pass provider as an argument")
        if not (config_class := config_class or getattr(component_class, "config_class", None)):
            raise ValueError(f"component_class must have a config_class attribute or pass config_class as an argument")
        if (can_get_components := getattr(component_class, "can_get_components", can_get_components)) is None:
            raise ValueError(f"component_class must have a can_get_components attribute or pass can_get_components as an argument")
        elif not hasattr(component_class, "can_get_components"):
            setattr(component_class, "can_get_components", can_get_components)
        
        self.register_config_class(config_class, component_type, provider)
        # For registering new component types
        if component_type not in self._component_types:
            self._component_types[component_type] = {}

        self._component_types[component_type][provider] = component_class
        

    # Set the default provider for a component_type when no provider is specified 
    # ie UnifAI().embedder().embed() with use the default provider for the embedder component_type
    def set_default_provider(self, component_type: ComponentType, provider: ProviderName) -> None:
        self._default_providers[component_type] = provider

    def update_default_providers(self, default_providers: dict[ComponentType, ProviderName]) -> None:
        self._default_providers.update(default_providers)  

    def get_default_provider(self, component_type: ComponentType) -> str:
        if config_default_provider := self._default_providers.get(component_type):
            return config_default_provider
        if initialized_providers := self._components.get(component_type):
            return _next(initialized_providers)
        if registered_providers := self._component_types.get(component_type):
            return _next(registered_providers)
        if registered_config_providers := self._component_configs.get(component_type):
            return _next(registered_config_providers)                    
        for provider in self._provider_configs:
            if provider in PROVIDERS[component_type]:
                return provider
        if component_type in DEFAULT_PROVIDERS:
            return DEFAULT_PROVIDERS[component_type]
        raise ValueError(f"No default provider found for component_type: {component_type}")
    
    # Helpers for getting components
    def _check_component_type_is_valid(self, component_type: ComponentType) -> None:
         if component_type not in self._component_types:
            raise ValueError(f"Invalid component_type: {component_type}. Must be one of: {','.join(self._component_types)}. Use register_component to add new component types")
    
    def _unpack_config_provider_component_args(self, config_or_name: ComponentConfig | ProviderName | tuple[ProviderName, ComponentName]) -> tuple[ProviderName, ComponentConfig | ComponentName]:
        if isinstance(config_or_name, ComponentConfig):
            return config_or_name.provider, config_or_name # ProviderName, ComponentConfig
        if isinstance(config_or_name, str):
            return config_or_name, "default" # ProviderName, ComponentName = "default" since just a ProviderName was passed
        if isinstance(config_or_name, tuple):
            if len(config_or_name) != 2:
                raise ValueError(f"Invalid config_or_name tuple: {config_or_name}. Must be of the form (provider, component_name). Got: {config_or_name=}")
            return config_or_name # ProviderName, ComponentName
        raise ValueError(f"Invalid config_or_name type: {type(config_or_name)}. Must be one of: {ComponentConfig, str, tuple}. Got: {config_or_name=} Type: {config_or_name.__class__._name__}")
    
    def _get_existing_component_instance(self, component_type: ComponentType, provider: ProviderName, component_name: ComponentName) -> Any|None:
        if (components_of_type := self._components.get(component_type)) and (components_of_provider := components_of_type.get(provider)):
            return components_of_provider.get(component_name)
        return None
    
    def _get_init_kwargs(self, 
                        can_get_components: bool,
                        component_config: ComponentConfig, 
                        provider_config: ProviderConfig,  
                        init_kwargs: dict
                        ) -> Any:

        _init_kwargs: dict[str, Any] = {}
        if api_key := provider_config.api_key:
            _init_kwargs["api_key"] = api_key
        if provider_config.init_kwargs:
            _init_kwargs.update(provider_config.init_kwargs)
        if component_config.init_kwargs:
            _init_kwargs.update(component_config.init_kwargs)
        if can_get_components:
            # Some components need to get other components. (can_get_components: ClassVar = True)
            _init_kwargs["_get_component"] = self._get_component        
        _init_kwargs.update(init_kwargs)
        return _init_kwargs

    def _init_component(self, 
                        component_type: ComponentType,
                        provider: ProviderName,
                        component_name: ComponentName,
                        component_config: ComponentConfig, 
                        provider_config: ProviderConfig,  
                        init_kwargs: dict,
                        cache: bool = True
                        ) -> Any:
        if (component_class := self._component_types[component_type].get(provider)) is None:
            component_class = import_component(component_type, provider)
            self.register_component(component_class)

        can_get_components = getattr(component_class, "can_get_components", False)
        _init_kwargs = self._get_init_kwargs(can_get_components, component_config, provider_config, init_kwargs)
        
        # Initialize component instance with the combined kwargs        
        component = component_class(config=component_config, **_init_kwargs)

        # Store component instance in cache if cache is True
        if cache: 
            # Ensure there is a dict for the component_type and provider before adding the component
            if component_type not in self._components:
                self._components[component_type] = {provider: {}}
            elif provider not in self._components[component_type]:
                self._components[component_type][provider] = {}
            self._components[component_type][provider][component_name] = component
        return component

    def _get_component(
        self,
        component_type: ComponentType,
        config_or_name: ComponentConfig | ProviderName | tuple[ProviderName, ComponentName],
        init_kwargs: dict,
        cache: bool = True,
        create_if_not_exists: bool = True,
        reuse_if_exists: bool = True,
        override_config_if_exists: bool = True,
    ) -> Any:        
        self._check_component_type_is_valid(component_type)
        provider, config_or_name = self._unpack_config_provider_component_args(config_or_name)
        if provider == "default" or not provider:
            provider = self.get_default_provider(component_type)             
        if isinstance(config_or_name, str):
            component_name = config_or_name
            component_config = self.get_component_config(component_type, provider, component_name)
        else:
            component_config = config_or_name
            component_name = component_config.name

        provider_config = self.get_provider_config(provider)
        
        if reuse_if_exists and (existing_instance := self._get_existing_component_instance(component_type, provider, component_name)):
            # If the resolved init_kwargs are different from the existing instance's passed init_kwargs,
            # or the component_config is different from the existing instance's config, AND override_config_if_exists=True,
            # create a new instance from the existing instance init_kwargs updated with the new init_kwargs. If override_config_if_exists=False, raise ValueError
            _init_kwargs = self._get_init_kwargs(existing_instance.can_get_components, component_config, provider_config, init_kwargs)
            if _init_kwargs != existing_instance._passed_init_kwargs or component_config != existing_instance.config:
                if override_config_if_exists:
                    # Return an instance of the existing component with the new init_kwargs
                    return existing_instance.with_config(config=component_config, **_init_kwargs)
                raise ValueError(
                        f"Cannot reuse existing instance of {component_type} {provider}.{component_name} because the config and/or init_kwargs are different from the existing instance's passed init_kwargs. Use override_config_if_exists=True to override."
                    )

            # If reuse_if_exists=True and init_kwargs are the same as the existing instance's init_kwargs, use the existing instance
            return existing_instance
        
        if not create_if_not_exists:
            raise ValueError(f"Component {component_type} {provider}.{component_name} does not exist or reuse_if_exists=False, and create_if_not_exists=False")

        # If no existing instance or reuse_if_exits=False and create_if_not_exists=True, create a new instance
        return self._init_component(component_type, provider, component_name, component_config, provider_config, init_kwargs, cache)
        

    # Cleanup
    def cleanup(self) -> None:
        """Cleanup all component instances"""
        self._component_types = {
            component_type: recursive_clear(providers) for component_type, providers in self._component_types.items()
        }
        self._components = recursive_clear(self._components)


