from typing import TYPE_CHECKING, Any, Literal, Optional, Type
from typing import Any, Callable, Collection, Literal, Optional, Sequence, Type, Union, Iterable, Generator, overload

from pathlib import Path
from pydantic import BaseModel, Field

from ..components.import_component import import_component


DEFAULT_PROVIDERS = {
    "llm": "openai",
    "embedder": "openai",
    "vector_db": "chroma",
    "reranker": "cohere",
    "document_db": "dict",
    "document_chunker": "unstructured"
}

class ProviderConfig(BaseModel):
    api_key: Optional[str] = None
    client_kwargs: dict[str, Any] = Field(default_factory=dict)
    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_rerank_model: Optional[str] = None
    component_type_client_kwarg_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict)

class Config(BaseModel):
    provider_configs: dict[str, ProviderConfig] = Field(default_factory=dict)
    default_providers: dict[str, str] = Field(default_factory=lambda: DEFAULT_PROVIDERS.copy())


class BaseClient:
    def __init__(
        self,
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_providers: Optional[dict[str, str]] = None,    
        **kwargs
    ):
        # Initialize config
        self.configure(
            config_obj_dict_or_path,
            provider_configs,
            api_keys,
            default_providers,
            **kwargs
        )

        self._component_types: dict[str, dict[str, Type[Any]]] = {component_type: {} for component_type in DEFAULT_PROVIDERS}
        self._components: dict[str, dict[str, Any]] = {}

    def configure(
        self,
        config_obj_dict_or_path: Optional[Config|dict[str, Any]|str|Path] = None,
        provider_configs: Optional[dict[str, ProviderConfig|dict[str, Any]]] = None,
        api_keys: Optional[dict[str, str]] = None,
        default_providers: Optional[dict[str, str]] = None,
        **kwargs
    ) -> Config:
        # Initialize config from input or create new
        if config_obj_dict_or_path is not None:
            if isinstance(config_obj_dict_or_path, str):
                config_obj_dict_or_path = Path(config_obj_dict_or_path)
            if isinstance(config_obj_dict_or_path, Path):
                config = Config.model_validate_json(config_obj_dict_or_path.read_bytes())
            elif isinstance(config_obj_dict_or_path, dict):
                config = Config.model_validate(config_obj_dict_or_path)
            elif isinstance(config_obj_dict_or_path, Config):
                config = config_obj_dict_or_path
            else:
                raise ValueError(f"Invalid config type: {type(config_obj_dict_or_path)}")
        else:
            config = Config()

        # Update provider configs
        if provider_configs is not None:
            for provider, provider_config in provider_configs.items():
                if isinstance(provider_config, dict):
                    provider_config = ProviderConfig.model_validate(provider_config)
                config.provider_configs[provider] = provider_config

        # Update API keys
        if api_keys is not None:
            for provider, api_key in api_keys.items():
                if provider_config := config.provider_configs.get(provider):
                    provider_config.api_key = api_key
                else:
                    config.provider_configs[provider] = ProviderConfig(api_key=api_key)

        # Update default providers
        if default_providers is not None:
            config.default_providers.update(default_providers)

        self.config = config
        return config
    
    # Components
    def register_component(
        self,
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        component_class: Type[Any]
    ) -> None:
        self._component_types[component_type][provider] = component_class

    def init_component(
        self,
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        **client_kwargs
    ) -> Any:
        if component_type not in self._component_types:
            raise ValueError(f"Invalid component_type: {component_type}")
        if component_type not in self._components:
            self._components[component_type] = {}
        
        if (component_class := self._component_types[component_type].get(provider)) is None:
            component_class = import_component(provider, component_type)
            self.register_component(provider, component_type, component_class)
        
        component_kwargs = {}
        if provider_config := self.config.provider_configs.get(provider):
            component_kwargs.update(provider_config.client_kwargs)
            if provider_config.api_key:
                component_kwargs["api_key"] = provider_config.api_key
            if component_type_client_kwargs := provider_config.component_type_client_kwarg_overrides.get(component_type):
                component_kwargs.update(component_type_client_kwargs)
        component_kwargs.update(client_kwargs)
        
        component = component_class(**component_kwargs)
        self._components[component_type][provider] = component
        return component

    def get_component(
        self,
        provider: str,
        component_type: Literal["llm", "embedder", "vector_db", "reranker", "document_db", "document_chunker"],
        **client_kwargs
    ) -> Any:
        components_of_type = self._components.get(component_type)
        if not components_of_type or provider not in components_of_type:
            return self.init_component(provider, component_type, **client_kwargs)
        return self._components[component_type][provider]

    # Cleanup
    def cleanup(self) -> None:
        """Cleanup all component instances"""
        for component_type in self._components:
            for component in self._components[component_type].values():
                if hasattr(component, 'cleanup'):
                    component.cleanup()
        self._components = {k: {} for k in self._components}

